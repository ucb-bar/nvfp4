"""End-to-end test of the NVFP4 tensor core through PyUARTSI, in RTL co-simulation or on the FPGA.

    pixi run sim -r 8                 # Verilator model of nvfp4_tsi_soc
    pixi run real -r 40 --gpu cell    # SophiaLake board, plus RTX 5090 check

Each run loads A and B (and C, or continues from the previous C), starts the core, and reads all
of C back. Every output is checked bit-exactly against a chain of 64-element dot products on the
independent integer model (host/nvfp4_model.py); with --gpu, also against the same chain of
m16n8k64 MMAs on an RTX 5090.
"""
import argparse
import random
import struct
import sys
import time

import nvfp4_tc as tc
from designs import rand_e2m1_vec, rand_fp32, rand_ue4m3, same
from flash_vector import FlashVector, NORMALIZE, ONE, SOFTMAX
from harness import ROOT, open_tsi
from nvfp4_model import chunk, tile
from run_gpu import run_cases


class Check:
    def __init__(self):
        self.fails = 0

    def __call__(self, ok, what):
        if not ok:
            self.fails += 1
            print(f"  FAIL: {what}")
        return ok


def read_full_c(core):
    words = struct.unpack(f"<{tc.TM * tc.TN}I", core.read_bytes(tc.C, tc.CROW * tc.TM))
    return [list(words[tc.TN * i:tc.TN * (i + 1)]) for i in range(tc.TM)]


def test_link(tsi, check, rng):
    check(tsi.read_word(tc.SYS_BASE) == 0x5453_4930, "sys ID")
    check(tsi.read_word(tc.SYS_BASE + 0x14) == 50_000_000, "CLK_HZ")
    for _ in range(8):
        v = rng.getrandbits(32)
        tsi.write_word(tc.SYS_BASE + 0x4, v)
        check(tsi.read_word(tc.SYS_BASE + 0x4) == v, f"scratch 0x{v:08x}")
    denied = tsi.read_word(tc.SYS_BASE + 0xC)
    check(tsi.read_word(0x3000_0000) == 0, "unmapped read returns 0")
    check(tsi.read_word(tc.TC_BASE + 0x100) == 0, "unmapped device offset reads 0")
    check(tsi.read_word(tc.SYS_BASE + 0xC) == denied + 2, "unmapped accesses are denied and counted")
    check(tsi.read_bytes(0x1_0000_0000, 8) == bytes(8), "addresses above 4 GiB read as 0")


def test_memories(core, check, rng):
    regions = [(tc.A, 8192), (tc.SFA, 1024), (tc.B, 8192), (tc.SFB, 1024), (tc.C, 16384)]
    data = {off: rng.randbytes(size) for off, size in regions}
    for off, size in regions:
        core.write_bytes(off, data[off])
    for off, size in regions:
        check(core.read_bytes(off, size) == data[off], f"read-back of region 0x{off:04x}")


def rand_scales(rng, k, mode):
    """K/16 scale codes per row: UE4M3 per 16 elements for NVFP4, E8M0 near 1.0 otherwise."""
    if mode == "nvfp4":
        return [rand_ue4m3(rng) for _ in range(k // 16)]
    return [rng.randrange(0x70, 0x8F) for _ in range(k // 16)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="use the Verilator co-simulation instead of the board")
    ap.add_argument("--port", help="board UART (default: the one SophiaLake by-id port)")
    ap.add_argument("-r", "--runs", type=int, default=12, help="random runs")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gpu", help="also check every output on this host's RTX 5090 (ssh)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    check = Check()
    tsi = open_tsi("tsi-sim" if args.sim else "fpga", args.port)

    runs, t0 = [], time.time()
    with tsi:
        print("link checks");      test_link(tsi, check, rng)
        core = tc.NVFP4TensorCore(tsi)
        print(f"{core.pm} x {core.pn} units, tile up to {tc.TM} x {tc.TN} x {tc.TK}")
        print("memory read-back"); test_memories(core, check, rng)

        ops0 = core.read(tc.OPS)
        core.write(tc.DIMS, 3 << 24 | 4 << 16 | 64)
        core.write(tc.CTRL, 1)
        check(core.read(tc.STATUS) == 0b10, "START with DIMS 3x4x64 is rejected")
        check(core.read(tc.OPS) == ops0, "a rejected START does not run")

        state = read_full_c(core)
        for x in range(args.runs):
            full = x < 2 or rng.random() < 0.2
            m = tc.TM if full else core.pm * rng.randint(1, tc.TM // core.pm)
            n = tc.TN if full else core.pn * rng.randint(1, tc.TN // core.pn)
            k = tc.TK if full else tc.KSTEP * rng.randint(1, tc.TK // tc.KSTEP)
            mode = "nvfp4" if x % 4 != 3 else rng.choice(["mxfp4", "fp4", "mxfp4_b16"])
            cont = bool(runs) and rng.random() < 0.3 and (m, n) == (runs[-1]["m"], runs[-1]["n"])
            overwrite = not cont and rng.random() < 0.4
            a = [rand_e2m1_vec(rng, k) for _ in range(m)]
            b = [rand_e2m1_vec(rng, k) for _ in range(n)]
            sfa = [rand_scales(rng, k, mode) for _ in range(m)]
            sfb = [rand_scales(rng, k, mode) for _ in range(n)]
            core.set_dims(m, n, k)
            core.set_mode(mode)
            core.load_a(a, sfa)
            core.load_b(b, sfb)
            if not cont and not overwrite:
                c = [[rand_fp32(rng) for _ in range(n)] for _ in range(m)]
                core.write_c(c)
                for i in range(m):
                    state[i][:n] = c[i]
            c_in = [row[:n] for row in state[:m]]
            cycles = core.run(overwrite=overwrite)
            after = read_full_c(core)
            outside = all(after[i][j] == state[i][j] for i in range(tc.TM) for j in range(tc.TN)
                          if i >= m or j >= n)
            check(outside, f"run {x}: C outside {m}x{n} changed")
            runs.append(dict(m=m, n=n, k=k, mode=mode, overwrite=overwrite, a=a, b=b, sfa=sfa, sfb=sfb,
                             c_in=c_in, c_out=[row[:n] for row in after[:m]], cycles=cycles))
            state = after
            print(f"  run {x:2d}: {m:2d} x {n:2d} x {k:3d} {mode:9s} "
                  f"{'overwrite' if overwrite else 'continue ' if cont else 'accumulate'} {cycles:5d} cycles")
        check(core.read(tc.OPS) - ops0 == len(runs), "OPS counts every run")
        vector = FlashVector(tsi)
        softmax = vector.run(SOFTMAX, [ONE] * 8, scalar=ONE)
        check(softmax['result'] == [ONE] * 8 and softmax['sum'] == 0x41000000
              and softmax['max'] == ONE and softmax['alpha'] == 0, "vector online softmax over TSI")
        normalized = vector.run(NORMALIZE, softmax['result'], scalar=softmax['sum'])
        check(normalized['result'] == [0x3e000000] * 8, "vector normalization over TSI")
        flags = tsi.read_word(tc.SYS_BASE + 0x8)
        check(flags & 0b11 == 0, f"link FLAGS 0x{flags:x} (dropped bytes or bad command)")
    elapsed = time.time() - t0

    rtl = [tile(r['a'], r['b'], r['sfa'], r['sfb'], r['c_in'], tc.MODES[r['mode']], r['overwrite'])
           for r in runs]
    total = sum(r["m"] * r["n"] for r in runs)
    bad = sum(r["c_out"][i][j] != e[i][j] for r, e in zip(runs, rtl) for i in range(r["m"]) for j in range(r["n"]))
    check(bad == 0, f"{bad} of {total} outputs differ from the integer dot-product model")
    macs = sum(r["m"] * r["n"] * r["k"] for r in runs)
    print(f"{len(runs)} runs, {total} outputs, {macs} MACs in {elapsed:.1f} s; "
          f"device == integer dot-product model on {total - bad}/{total}")

    if args.gpu:
        work = ROOT / "build/gpu/tsi"
        work.mkdir(parents=True, exist_ok=True)
        gpu = []
        for run in runs:
            acc = [[0 if run['overwrite'] else value for value in row] for row in run['c_in']]
            mode = tc.MODES[run['mode']]
            for lo in range(0, run['k'], 64):
                cases = []
                for i in range(run['m']):
                    for j in range(run['n']):
                        a, b, sa, sb = chunk(run['a'][i], run['b'][j], run['sfa'][i], run['sfb'][j], lo, mode)
                        cases.append(dict(a=a, b=b, sa=sa, sb=sb, c=acc[i][j], mode=mode))
                outputs = iter(run_cases(args.gpu, cases, work))
                acc = [[next(outputs) for j in range(run['n'])] for i in range(run['m'])]
            gpu.append(acc)
        dev_eq = sum(same(r["c_out"][i][j], g[i][j]) for r, g in zip(runs, gpu) for i in range(r["m"]) for j in range(r["n"]))
        rtl_eq = sum(same(e[i][j], g[i][j]) for r, e, g in zip(runs, rtl, gpu) for i in range(r["m"]) for j in range(r["n"]))
        print(f"RTX 5090 tensor core: device == GPU on {dev_eq}/{total} "
              f"(integer dot-product model == GPU on {rtl_eq}/{total})")

    print("PASS" if check.fails == 0 else f"FAIL ({check.fails} checks)")
    sys.exit(1 if check.fails else 0)


if __name__ == "__main__":
    main()
