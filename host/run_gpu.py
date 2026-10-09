"""Run the FPGA test vectors on a real NVFP4 tensor core (sm_120a mma.sync) and compare.

    pixi run python host/run_gpu.py allegro_fp4 -n 100000 --seed 7 --fpga-out results/allegro_fp4_n100000_s7.out

Stimulus is regenerated from (design, n, seed), so use the same values as the FPGA run, whose
outputs come from run_fpga.py --save-out.
"""
import argparse
import json
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

from designs import DESIGNS, same
from harness import ROOT, make_cases

REMOTE_DIR = "nvfp4_gpu"
NVCC = "/usr/local/cuda-13.2/bin/nvcc"
KIND_NVFP4, KIND_MX_B16, KIND_MX_B32 = 0, 1, 2


def record(c):
    """(kind, 76-byte record) computing the same dot product on one m16n8k64 MMA."""
    if len(c["a"]) == 64:
        a, b, sa, sb, cc, mode = c["a"], c["b"], c["sa"], c["sb"], c["c"], c["mode"]
        kind = {0: KIND_NVFP4, 1: KIND_MX_B32, 2: KIND_MX_B16, 3: KIND_MX_B16}[mode]
        if mode == 2:
            sa = sb = [0x7F] * 4  # plain FP4: E8M0 1.0
    else:
        # 16-element designs: pad with zero elements. Padding blocks reuse the real block's scale so
        # they cannot change the alignment exponent (MMA-Sim takes a block's exponent from its scales).
        a, b = c["a"] + [0] * 48, c["b"] + [0] * 48
        sa, sb = [c["sa"]] * 4, [c["sb"]] * 4
        cc, kind = c.get("c", 0), KIND_NVFP4
    nib = lambda v: sum((x & 0xF) << (4 * i) for i, x in enumerate(v)).to_bytes(32, "little")
    return kind, nib(a) + nib(b) + bytes(sa) + bytes(sb) + struct.pack("<I", cc)


def result_bits(out: bytes) -> int:
    return int.from_bytes(out[:4], "little")


def run_remote(host, kinds_recs, work):
    src = ROOT / "host/gpu/nvfp4_mma.cu"
    subprocess.run(["ssh", host, f"mkdir -p {REMOTE_DIR}"], check=True)
    files = [src]
    for kind, recs in kinds_recs.items():
        p = work / f"in_{kind}.bin"
        p.write_bytes(b"".join(recs))
        files.append(p)
    subprocess.run(["scp", "-q", *map(str, files), f"{host}:{REMOTE_DIR}/"], check=True)
    cmds = [f"cd {REMOTE_DIR}",
            f"[ nvfp4_mma -nt nvfp4_mma.cu ] || {NVCC} -O2 -gencode arch=compute_120a,code=sm_120a -o nvfp4_mma nvfp4_mma.cu"]
    cmds += [f"./nvfp4_mma {k} in_{k}.bin out_{k}.bin" for k in kinds_recs]
    p = subprocess.run(["ssh", host, " && ".join(cmds)], capture_output=True, text=True)
    print(p.stderr.strip())
    if p.returncode:
        sys.exit("remote run failed")
    outs = {}
    for kind in kinds_recs:
        subprocess.run(["scp", "-q", f"{host}:{REMOTE_DIR}/out_{kind}.bin", str(work)], check=True)
        raw = (work / f"out_{kind}.bin").read_bytes()
        outs[kind] = list(struct.unpack(f"<{len(raw) // 4}I", raw))
    return outs


def run_cases(host, cases, work):
    """Return the RTX 5090 output of each case, grouped into one remote run per MMA kind."""
    kinds, by_kind = [], {}
    for c in cases:
        kind, rec = record(c)
        kinds.append((kind, len(by_kind.setdefault(kind, []))))
        by_kind[kind].append(rec)
    outs = run_remote(host, by_kind, work)
    return [outs[kind][i] for kind, i in kinds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design", choices=DESIGNS)
    ap.add_argument("-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--modes", default="nvfp4", help="allegro_fp4 only")
    ap.add_argument("--reference", default="silicon", choices=["silicon", "mmasim"], help="allegro_fp4 only")
    ap.add_argument("--fpga-out", help="run_fpga.py --save-out file from the same (n, seed) run")
    ap.add_argument("--host", default="cell")
    ap.add_argument("--save", help="write per-case GPU/golden/FPGA results to this JSON file")
    args = ap.parse_args()

    design = DESIGNS[args.design](**({"modes": args.modes.split(","), "reference": args.reference}
                                     if args.design == "allegro_fp4" else {}))
    cases = make_cases(design, args.n, args.seed)
    work = ROOT / "build/gpu" / args.design
    work.mkdir(parents=True, exist_ok=True)

    gpu = run_cases(args.host, cases, work)

    golden_raw = design.expected_batch(cases) if hasattr(design, "expected_batch") else \
        [design.expected(c) for c in cases]
    golden = [result_bits(g) for g in golden_raw]
    fpga = [None] * len(cases)
    if args.fpga_out:
        raw = Path(args.fpga_out).read_bytes()
        step = design.out_bytes
        assert len(raw) == step * len(cases), "FPGA output file does not match (n, seed)"
        fpga = [result_bits(raw[i * step:(i + 1) * step]) for i in range(len(cases))]

    name = {"allegro_fp4": f"golden ({args.reference})", "dotprod_formal": "exact model",
            "vortex_fedp": "Vortex RTL", "fp4_fedp": "silicon model", "fp4_fedp_dsp": "silicon model"}[design.name]
    bad = sum(g == 0x7FBADBAD for g in gpu)
    print(f"{design.name}: {len(cases)} vectors on the tensor core ({bad} with inconsistent outputs)")
    tally = Counter()
    rows = []
    for i, (c, g, gold, fp) in enumerate(zip(cases, gpu, golden, fpga)):
        key = (same(g, gold), fp is not None and same(g, fp))
        tally[key] += 1
        if not key[0] or (args.fpga_out and not key[1]):
            rows.append(dict(index=i, case=c, gpu=f"{g:08x}", golden=f"{gold:08x}",
                             fpga=None if fp is None else f"{fp:08x}"))
    for (eg, ef), n in sorted(tally.items(), reverse=True):
        fp_txt = f" and GPU {'==' if ef else '!='} FPGA" if args.fpga_out else ""
        print(f"  GPU {'==' if eg else '!='} {name:24s}{fp_txt}: {n}")
    for r in rows[:12]:
        print(f"    case {r['index']}: gpu={r['gpu']} golden={r['golden']} fpga={r['fpga']}  {design.describe(r['case'])}")
    if hasattr(design, "spec"):
        exact = sum(same(g, design.spec(c)) for c, g in zip(cases, gpu))
        print(f"  GPU == exact fused dot product (single RNE): {exact}/{len(cases)}")
    if args.save:
        Path(args.save).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
