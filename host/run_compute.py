"""Check the tensor core, the FP32 ALU, vector commands and tiled attention; optionally compare an RTX 5090.

    pixi run python host/run_compute.py --gpu cell
"""
import argparse
import json
import math
import random
import struct
import subprocess

import numpy as np

from compute_sim import ComputeSim
from designs import E2M1_MAG2, rand_fp32, same, ulp_distance
from flash_vector import FlashVector, ADD, SUB, MUL, SCALE, AXPY, MAX, SUM, EXP2, RECIP, SOFTMAX, NORMALIZE, SILU
from harness import ROOT
from nvfp4_model import tile
import nvfp4_tc as tc

PRIMITIVE = 15  # vector_check.cu record: add, mul, exp2, recip and max of x[0] and y[0]

def bits(x):
    with np.errstate(all='ignore'):
        return struct.unpack('<I', np.float32(x).tobytes())[0]


def value(x):
    return struct.unpack('<f', struct.pack('<I', x))[0]


def ftz(x):
    return x & 0x8000_0000 if (x >> 23) & 255 == 0 else x


def oracle(records, gpu):
    source = (ROOT / 'host/gpu/vector_check.cu').read_text()
    command = ('mkdir -p /tmp/nvfp4_ws_check && cat > /tmp/nvfp4_ws_check/vector_check.cu && '
               '/usr/local/cuda/bin/nvcc -O2 -arch=sm_120a --ftz=true --fmad=false '
               '/tmp/nvfp4_ws_check/vector_check.cu -o /tmp/nvfp4_ws_check/vector_check')
    compile_result = subprocess.run(['ssh', gpu, command], input=source, text=True, capture_output=True)
    if compile_result.returncode:
        raise RuntimeError(compile_result.stderr)
    lines = []
    for r in records:
        fields = [r['op'], r['mask'], r['scalar'], r['old_max'], r['old_sum'], *r['x'], *r['y']]
        lines.append(' '.join(f'{v:x}' for v in fields))
    process = subprocess.run(['ssh', gpu, '/tmp/nvfp4_ws_check/vector_check'],
                             input='\n'.join(lines)+'\n', capture_output=True, text=True, check=True)
    outputs = [[int(v, 16) for v in line.split()] for line in process.stdout.splitlines()]
    assert len(outputs) == len(records)
    return outputs


def primitives(sim, rng, count):
    records, outputs = [], []
    edges = [0, 0x8000_0000, 1, 0x007f_ffff, 0x0080_0000, 0x0080_0001, 0x7f7f_ffff,
             0x7f80_0000, 0xff80_0000, 0x7fc0_0000, bits(-126), bits(128), bits(-128), bits(1)]
    pairs = [(a,b) for a in edges for b in edges]
    pairs += [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(count)]
    pairs += [(bits(rng.uniform(-126,128)), bits(rng.uniform(.25,256))) for _ in range(count)]
    worst = [0, 0]
    for a,b in pairs:
        output = sim.command(f'F {a:x} {b:x}')
        registered = [sim.command(f'P {op:x} {a:x} {b:x}')[0] for op in (ADD,MUL,EXP2,RECIP,MAX)]
        assert all(same(x,y) for x,y in zip(output,registered)), f'ALU/reference {a:08x} {b:08x}: {output} vs {registered}'
        va, vb = value(ftz(a)), value(ftz(b))
        expected = [ftz(bits(va + vb)), ftz(bits(va * vb))]
        assert same(output[0], expected[0]), f'add {a:08x} {b:08x}: {output[0]:08x} != {expected[0]:08x}'
        assert same(output[1], expected[1]), f'mul {a:08x} {b:08x}: {output[1]:08x} != {expected[1]:08x}'
        if math.isfinite(va) and -126 <= va < 128:
            reference = ftz(bits(2**va))
            distance = ulp_distance(output[2],reference)
            worst[0] = max(worst[0], distance)
            assert distance <= 2, f'exp2 {va}: {distance} ULP'
        if math.isfinite(va) and va != 0:
            reference = ftz(bits(1/va))
            distance = ulp_distance(output[3],reference)
            worst[1] = max(worst[1], distance)
            assert distance <= 2, f'recip {a:08x}: {distance} ULP'
        records.append(dict(op=PRIMITIVE,mask=255,scalar=0,old_max=0,old_sum=0,x=[a]+[0]*7,y=[b]+[0]*7))
        outputs.append(output)
    print(f'FP32 primitives: {len(pairs)} pairs, add/mul exact, exp2/recip max {worst} ULP vs RNE')
    return records, outputs, worst


def matrices(sim, rng, runs):
    core = tc.NVFP4TensorCore(sim)
    assert (core.pm,core.pk,core.pn) == (2,64,4)
    # Memory address mapping: every bank/lane/chunk, including scale-word halves.
    for off,size in [(tc.A,8192),(tc.SFA,1024),(tc.B,8192),(tc.SFB,1024),(tc.C,16384)]:
        blob = rng.randbytes(size)
        core.write_bytes(off,blob)
        assert core.read_bytes(off,size) == blob, f'memory {off:x}'
    _,denied,_ = sim.access(True,core.base+tc.A,0,3)
    assert denied
    core.write(tc.DIMS,2<<24|8<<16|16)
    core.write(tc.CTRL,3)
    assert core.read(tc.STATUS) == 2
    state = np.array(struct.unpack('<4096I',core.read_bytes(tc.C,16384))).reshape(64,64).tolist()
    total = 0
    cases = [(2,8,32,0,True),(2,8,256,0,False),(64,64,256,0,True),
             (6,24,96,1,False),(4,16,160,2,True),(8,32,224,3,False)]
    cases += [(2*rng.randint(1,8),4*rng.randint(1,8),32*rng.randint(1,8),x%4,bool(x%2))
              for x in range(runs)]
    for index,(m,n,k,mode,overwrite) in enumerate(cases):
        a = [[rng.randrange(16) for _ in range(k)] for _ in range(m)]
        b = [[rng.randrange(16) for _ in range(k)] for _ in range(n)]
        scales = lambda: [rng.randrange(128) if mode==0 else rng.randrange(255) for _ in range(k//16)]
        sa,sb = [scales() for _ in range(m)],[scales() for _ in range(n)]
        initial = [[rand_fp32(rng) for _ in range(n)] for _ in range(m)]
        core.set_dims(m,n,k)
        core.write(tc.MODE,mode)
        core.load_a(a,sa); core.load_b(b,sb); core.write_c(initial)
        for i in range(m): state[i][:n] = initial[i]
        expected = tile(a,b,sa,sb,initial,mode,overwrite)
        cycles = core.run(overwrite)
        after = np.array(struct.unpack('<4096I',core.read_bytes(tc.C,16384))).reshape(64,64).tolist()
        for i in range(64):
            for j in range(64):
                want = expected[i][j] if i<m and j<n else state[i][j]
                assert after[i][j] == want, f'tile {index} C[{i}][{j}] {after[i][j]:08x} != {want:08x}'
        # 1 weight-load + M/PM issues + 7 pipeline drain cycles for every N/PN*ceil(K/PK) group.
        assert cycles == (n//core.pn)*(-(-k//core.pk))*(m//core.pm+8), f'schedule cycles {cycles}'
        state=after
        total += m*n
    print(f'WS matrix: {len(cases)} runs, {total} outputs exact; untouched C, modes, bank mapping, cycles pass')
    return dict(runs=len(cases),outputs=total)


def vectors(vector,rng,count):
    records, outputs = [], []
    for i in range(count):
        op = i % 11
        x = [bits(rng.uniform(-16,16)) for _ in range(8)]
        y = [bits(rng.uniform(-4,4)) for _ in range(8)]
        scalar = bits(rng.uniform(.1,2))
        old_max = bits(rng.uniform(-12,12))
        old_sum = 0 if i%3 == 0 else bits(rng.uniform(.1,32))
        mask = 0 if i%19 == 0 else rng.randrange(256)
        if op == NORMALIZE and i%2 == 0: scalar = 0
        r=dict(op=op,mask=mask,scalar=scalar,old_max=old_max,old_sum=old_sum,x=x,y=y)
        out=vector.run(op,x,y,scalar,old_max,old_sum,mask)
        packed=[*out['result'],out['reduction'],out['max'],out['sum'],out['alpha']]
        records.append(r); outputs.append(packed)
        if op == SOFTMAX:
            active = [value(x[j]) for j in range(8) if mask>>j&1]
            maximum = max(active + ([value(old_max)] if old_sum else [-math.inf]))
            assert same(out['max'],bits(maximum))
            p = [2**((value(x[j])-maximum)*value(scalar)) if mask>>j&1 else 0 for j in range(8)]
            alpha = 2**((value(old_max)-maximum)*value(scalar)) if old_sum else 0
            assert np.allclose([value(v) for v in out['result']],p,rtol=5e-6,atol=1e-30)
            assert math.isclose(value(out['sum']),alpha*value(old_sum)+sum(p),rel_tol=5e-6,abs_tol=1e-30)
    print(f'vector commands: {count}, all operations, masking, empty rows and continuation pass')
    return records,outputs


def streams(sim, vector, rng):
    """A command over many groups equals one-group commands; MAX and SUM combine left to right."""
    x_at, y_at, d_at, groups = vector.stage - 0x1000, vector.stage - 0xc00, vector.stage - 0x800, 16
    pack = lambda words: struct.pack(f'<{len(words)}I', *words)
    for op in (ADD, SUB, MUL, SCALE, AXPY, MAX, SUM, EXP2, RECIP, NORMALIZE, SILU):
        x = [bits(rng.uniform(-8, 8)) for _ in range(8 * groups)]
        y = [bits(rng.uniform(-8, 8)) for _ in range(8 * groups)]
        scalar, mask = bits(rng.uniform(.1, 2)), rng.randrange(1, 256)
        sim.write_bytes(x_at, pack(x))
        sim.write_bytes(y_at, pack(y))
        vector.command(op, x_at, y_at, d_at, groups, mask, scalar)
        result = struct.unpack(f'<{8 * groups}I', sim.read_bytes(d_at, 32 * groups))
        reduction = sim.read_word(vector.base + 0x10)
        total = None
        for g in range(groups):
            one = vector.run(op, x[8*g:8*g+8], y[8*g:8*g+8], scalar, mask=mask)
            assert list(result[8*g:8*g+8]) == one['result'], f'op {op} group {g}'
            part = one['reduction']
            total = part if total is None else vector.run(op, [total, part] + [0] * 6, mask=3)['reduction']
        if op in (MAX, SUM): assert same(reduction, total), f'op {op} reduction'
    # Rejected: misaligned or non-scratch operands, SOFTMAX over 2 groups, no groups.
    for args in [(ADD, x_at + 4, y_at, d_at, 1), (ADD, x_at, 0x1000_0000, d_at, 1),
                 (SCALE, x_at, 0, d_at + 8, 1), (SOFTMAX, x_at, 0, d_at, 2), (SCALE, x_at, 0, d_at, 0)]:
        try: vector.command(*args)
        except ValueError: continue
        raise AssertionError(f'accepted {args}')
    print(f'vector streams: 11 operations x {groups} groups match one-group commands; rejections pass')


def attention(sim, vector, rng, gpu):
    worst = 0
    gpu_rows, rtl_rows = [], []
    core = tc.NVFP4TensorCore(sim)
    # P*V runs in FP32 through MUL and SUM, and the output recurrence through AXPY and NORMALIZE,
    # so this check covers the vector unit without NVFP4 quantization of P.
    for causal in (False,True):
        codes = [0,1,2,3,4,8,9,10,11,12]
        q_codes = [[rng.choice(codes) for _ in range(32)] for _ in range(2)]
        k_codes = [[rng.choice(codes) for _ in range(32)] for _ in range(24)]
        core.set_dims(2,24,32)
        core.set_mode('nvfp4')
        core.load_a(q_codes,[[0x38]*2 for _ in range(2)])
        core.load_b(k_codes,[[0x38]*2 for _ in range(24)])
        core.run(overwrite=True)
        scores = np.array([[value(v) for v in row] for row in core.read_c()],dtype=np.float32)
        decode = lambda code: (-1 if code & 8 else 1) * E2M1_MAG2[code & 7] / 2
        q = np.array([[decode(v) for v in row] for row in q_codes])
        k = np.array([[decode(v) for v in row] for row in k_codes])
        assert np.array_equal(scores,q @ k.T), 'NVFP4 QK score matrix'
        v=np.array([[rng.uniform(-2,2) for _ in range(8)] for _ in range(24)],dtype=np.float32)
        for row in range(2):
            maximum,total=0xff80_0000,0
            output=[0]*8
            for block in range(3):
                lo=block*8
                mask=sum(1<<j for j in range(8) if not causal or lo+j<=row*8+3)
                result=vector.run(SOFTMAX,[bits(x) for x in scores[row,lo:lo+8]],
                                  scalar=bits(math.log2(math.e)/math.sqrt(32)),
                                  old_max=maximum,old_sum=total,mask=mask)
                maximum,total=result['max'],result['sum']
                pv=[]
                for column in range(8):
                    products=vector.run(MUL,result['result'],[bits(x) for x in v[lo:lo+8,column]])
                    pv.append(vector.run(SUM,products['result'])['reduction'])
                output=vector.run(AXPY,output,pv,scalar=result['alpha'])['result']
            output=vector.run(NORMALIZE,output,scalar=total)['result']
            mask = (1 << 24)-1 if not causal else (1 << (row*8+4))-1
            gpu_rows.append([mask,bits(math.log2(math.e)/math.sqrt(32)),
                             *[bits(x) for x in scores[row]], *[bits(x) for x in v.flatten()]])
            rtl_rows.append(output)
            logits=scores[row].astype(np.float64)/math.sqrt(32)
            if causal: logits[np.arange(24)>row*8+3]=-np.inf
            p=np.exp(logits-np.max(logits)); expected=(p/p.sum())@v.astype(np.float64)
            error=max(abs(np.array([value(x) for x in output])-expected))
            worst=max(worst,error)
            assert error<2e-6, f'tiled attention error {error}'
    print(f'tiled attention: dense and causal, 3 blocks, max absolute error {worst:.3g}')
    report = {'fp64_max_absolute_error': worst}
    if gpu:
        source = (ROOT/'host/gpu/attention_check.cu').read_text()
        command = ('cat > /tmp/nvfp4_ws_check/attention_check.cu && '
                   '/usr/local/cuda/bin/nvcc -O2 -arch=sm_120a --ftz=true --fmad=false '
                   '/tmp/nvfp4_ws_check/attention_check.cu -o /tmp/nvfp4_ws_check/attention_check')
        subprocess.run(['ssh',gpu,'mkdir -p /tmp/nvfp4_ws_check'],check=True)
        process = subprocess.run(['ssh',gpu,command],input=source,text=True,capture_output=True)
        if process.returncode: raise RuntimeError(process.stderr)
        frames = '\n'.join(' '.join(f'{x:x}' for x in row) for row in gpu_rows)+'\n'
        process = subprocess.run(['ssh',gpu,'/tmp/nvfp4_ws_check/attention_check'],
                                 input=frames,text=True,capture_output=True,check=True)
        gpu_output = [[int(x,16) for x in row.split()] for row in process.stdout.splitlines()]
        assert len(gpu_output) == len(rtl_rows)
        error = max(abs(value(a)-value(b)) for got,want in zip(rtl_rows,gpu_output) for a,b in zip(got,want))
        assert error < 2e-6, f'GPU attention error {error}'
        report['gpu_max_absolute_error'] = error
        print(f'RTX5090 tiled attention: {len(rtl_rows)} dense/causal rows, max absolute error {error:.3g}')
    return report


def compare_gpu(records,outputs,gpu):
    expected=oracle(records,gpu)
    worst=[0]*5
    vector_worst = {}
    bad=0
    for r,got,want in zip(records,outputs,expected):
        fields=5 if r['op']==PRIMITIVE else 12
        for i in range(fields):
            if same(got[i],want[i]): continue
            distance=ulp_distance(got[i],want[i])
            if r['op']==PRIMITIVE:
                worst[i]=max(worst[i],distance or 0)
                tolerance=2 if i in (2,3) else 0
                ok=distance is not None and distance<=tolerance
            else:
                vector_worst[r['op']] = max(vector_worst.get(r['op'], 0), distance or 0)
                # Compound exp2/rcp commands include FP32 input rounding and reduction rounding.
                tolerance=6 if r['op'] in (SOFTMAX,NORMALIZE,EXP2,RECIP) else 0
                ok=distance is not None and distance<=tolerance
            if not ok:
                bad+=1
                if bad<8: print(f'GPU mismatch op={r["op"]} field={i}: {got[i]:08x} vs {want[i]:08x}, {distance} ULP')
    assert bad==0,f'{bad} GPU comparisons exceeded bounds'
    print(f'RTX5090: {len(records)} records pass; primitive max ULP [add,mul,exp2,rcp,max]={worst}')
    return dict(records=len(records),primitive_max_ulp=worst,vector_max_ulp=vector_worst)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',help='SSH host with RTX5090 and nvcc')
    parser.add_argument('--samples',type=int,default=10000)
    parser.add_argument('--runs',type=int,default=12)
    args=parser.parse_args()
    rng=random.Random(17)
    report={}
    with ComputeSim() as sim:
        records,outputs,report['primitive_rne_ulp']=primitives(sim,rng,args.samples)
        report['matrix']=matrices(sim,rng,args.runs)
        vector=FlashVector(sim)
        vr,vo=vectors(vector,rng,220)
        streams(sim,vector,rng)
        report['attention']=attention(sim, vector, rng, args.gpu)
        if args.gpu: report['gpu']=compare_gpu(records+vr,outputs+vo,args.gpu)
        # Partial command writes are denied; an invalid operation is rejected.
        _,denied,_=sim.access(True,vector.base+0x20,0,3)
        assert denied
        sim.write_word(vector.base+0x3c,0xf1)
        assert sim.read_word(vector.base+8)==2
    path = ROOT / ('results/compute_validation_gpu.json' if args.gpu else 'results/compute_validation.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS')


if __name__=='__main__':
    main()
