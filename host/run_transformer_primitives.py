"""Validate runtime NVFP4 quantization, RMSNorm rsqrt and SwiGLU SiLU primitives."""
import argparse
import math
import random
import struct
import subprocess

import numpy as np

from compute_sim import ComputeSim
from flash_vector import FlashVector, RSQRT, SILU, ABS, QUANT
from nvfp4_tc import pack_codes
from quantization import Quantizer, quantize_block, quantize_rows
from harness import ROOT


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def value(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def quant_rows(sim, vector, rng):
    """QUANT over rows of 1-16 blocks writes the operand layout and leaves the rest untouched."""
    x_at, codes_at, scales_at, rows = 0x2001_E000, 0x2001_F000, 0x2001_F400, 3
    for blocks in (1, 2, 3, 4, 5, 16):
        values = np.array([[rng.uniform(-1, 1) * 2.**rng.randint(-8, 8) for _ in range(16 * blocks)]
                           for _ in range(rows)], dtype=np.float32)
        sim.write_bytes(x_at, values.tobytes())
        sim.write_bytes(codes_at, b'\xa5' * 128 * rows)
        sim.write_bytes(scales_at, b'\xa5' * 16 * rows)
        vector.command(QUANT, x_at, scales_at, codes_at, rows, blocks, bits(64.))
        codes, scales = quantize_rows(values, 64.)
        used = 4 * -(-blocks // 4)  # scale bytes in whole words
        for r in range(rows):
            got = sim.read_bytes(codes_at + 128 * r, 128)
            assert got == pack_codes(codes[r].tolist()) + b'\xa5' * (128 - 8 * blocks), (blocks, r)
            got = sim.read_bytes(scales_at + 16 * r, 16)
            assert got == bytes(scales[r]) + bytes(used - blocks) + b'\xa5' * (16 - used), (blocks, r)
    print(f'QUANT rows: {rows} rows of 1 to 16 blocks match quantize_rows; other bytes untouched')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu')
    args = parser.parse_args()
    rng = random.Random(61)
    records, outputs = [], []
    quant_records, quant_outputs = [], []
    with ComputeSim() as sim:
        quantizer = Quantizer(sim)
        blocks = [[0.] * 16, [-0.] * 16, [1.] * 16,
                  [6., -.25, .25, .75, 1.25, 1.75, 2.5, 3.5, 5.] + [0.] * 7,
                  [1e30, -1e30] * 8]
        blocks += [[rng.uniform(-1, 1) * 2.**rng.randint(-12, 9) for _ in range(16)] for _ in range(200)]
        for index, block in enumerate(blocks):
            inverse = (1., 64., 2688.)[index % 3]
            got = quantizer.run(block, inverse)
            want = quantize_block(block, inverse)
            assert got == want, (index, got, want)
            quant_records.append([bits(inverse), *[bits(v) for v in block]])
            quant_outputs.append(got)
        for bad in (np.nan, np.inf, -np.inf):
            try: quantizer.run([bad] + [0.] * 15)
            except ValueError: pass
            else: raise AssertionError('nonfinite quantizer input accepted')
        for inverse in (0., -1., np.nan, np.inf):
            try: quantizer.run([1.] * 16, inverse)
            except ValueError: pass
            else: raise AssertionError('invalid inverse scale accepted')
        print(f'Runtime quantization: {len(blocks)} blocks nearest-even PASS; invalid inputs rejected')
        vector = FlashVector(sim)
        quant_rows(sim, vector, rng)
        special = [0, 0x80000000, 0x7f800000, 0xff800000, 0xbf800000, 0x7fc00001, 1, 0x80000001]
        actual = vector.run(RSQRT, special)['result']
        expected = [0x7f800000, 0xff800000, 0, None, None, None, 0x7f800000, 0xff800000]
        for a, b in zip(actual, expected):
            assert (a & 0x7fffffff) > 0x7f800000 if b is None else a == b
        worst = {RSQRT: 0, SILU: 0, ABS: 0}
        for operation in (RSQRT, SILU, ABS):
            for _ in range(100):
                x = [np.float32(2.**rng.uniform(-125, 127) if operation == RSQRT else rng.uniform(-80, 80)) for _ in range(8)]
                words = [bits(v) for v in x]
                got = vector.run(operation, words)
                records.append(dict(op=operation, mask=255, scalar=bits(1.), old_max=bits(-np.inf), old_sum=0, x=words, y=[0]*8))
                outputs.append(got['result'])
                for a, word in zip(x, got['result']):
                    expected = np.float32(1 / math.sqrt(float(a)) if operation == RSQRT else
                                          float(a) / (1 + math.exp(-float(a))) if operation == SILU else abs(float(a)))
                    distance = abs(bits(expected) - word)
                    worst[operation] = max(worst[operation], distance)
                    assert distance <= (3 if operation == RSQRT else 100 if operation == SILU else 0), (operation, a, value(word), expected, distance)
        print(f'RSQRT/SiLU/ABS CPU max ULP: {worst}')
    if args.gpu:
        from run_compute import oracle, ulp_distance
        expected = oracle(records, args.gpu)
        worst = {RSQRT: 0, SILU: 0, ABS: 0}
        for record, got, want in zip(records, outputs, expected):
            for a, b in zip(got, want[:8]):
                distance = ulp_distance(a, b)
                assert distance is not None
                worst[record['op']] = max(worst[record['op']], distance)
                assert distance <= (4 if record['op'] == RSQRT else 6 if record['op'] == SILU else 0), (record['op'], a, b, distance)
        print(f'RTX5090 RSQRT/SiLU/ABS max ULP: {worst}')
        subprocess.run(['scp', str(ROOT / 'host/gpu/quant_check.cu'), f'{args.gpu}:/tmp/nvfp4_ws_check/quant_check.cu'], check=True)
        subprocess.run(['ssh', args.gpu, '/usr/local/cuda/bin/nvcc -O3 -arch=sm_120a --ftz=true --fmad=false '
                        '/tmp/nvfp4_ws_check/quant_check.cu -o /tmp/nvfp4_ws_check/quant_check'], check=True)
        encoded = '\n'.join(' '.join(f'{word:x}' for word in record) for record in quant_records) + '\n'
        process = subprocess.run(['ssh', args.gpu, '/tmp/nvfp4_ws_check/quant_check'], input=encoded,
                                 capture_output=True, text=True, check=True)
        expected = [tuple(int(word, 16) for word in line.split()) for line in process.stdout.splitlines()]
        assert len(expected) == len(quant_outputs)
        for index, (got, want) in enumerate(zip(quant_outputs, expected)):
            assert got == want, (index, got, want)
        print(f'RTX5090 native E4M3/E2M1 conversion: {len(expected)} blocks bit-exact PASS')


if __name__ == '__main__':
    main()
