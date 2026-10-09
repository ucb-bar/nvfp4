"""Check a 64-element PE (fpga/pe/fp4_fedp.sv, or fp4_fedp_dsp.sv) on its Verilator model.

    pixi run python host/run_pe.py                  # stored FPGA/GPU vectors + 100k random vectors
    pixi run python host/run_pe.py -n 200000 --gpu cell
    pixi run python host/run_pe.py --design fp4_fedp_dsp

Each output is compared with the silicon-calibrated model (FP4FEDP.expected) and with real
silicon: the RTX 5090 outputs stored in tcgen05/vectors/*.vec (Jetson Thor gives the same
outputs), and with --gpu, the random vectors run on that host's RTX 5090. fp4_fedp_dsp returns
four outputs per case (designs.FP4FEDPDSP); silicon is compared with the first, the case itself.
"""
import argparse
import random
import sys
from collections import Counter

import vl_ref
from designs import DESIGNS, same
from harness import ROOT
from run_gpu import run_cases

MODE_OF_KIND = {0: 0, 1: 3, 2: 1}  # script kind (NVFP4, MX16, MX32) -> unit mode


def same_words(x, y):
    """Equal outputs of one or more FP32 words (designs.same per word)."""
    while x or y:
        if not same(x & 0xFFFF_FFFF, y & 0xFFFF_FFFF):
            return False
        x, y = x >> 32, y >> 32
    return True


def report(name, design, cases, rtl, silicon=None):
    t = design.t
    model = [int.from_bytes(design.expected(c), "little") for c in cases]
    first = [r & 0xFFFF_FFFF for r in rtl]
    line = f"{name:22s} {len(cases):>7} vectors: RTL == model {sum(same_words(r, m) for r, m in zip(rtl, model)):>7}"
    if silicon is not None:
        line += f"   RTL == silicon {sum(same(r, s) for r, s in zip(first, silicon)):>7}"
        line += f"   model == silicon {sum(same(m & 0xFFFF_FFFF, s) for m, s in zip(model, silicon)):>7}"
    print(line)
    bad = [i for i, (r, m) in enumerate(zip(rtl, model)) if not same_words(r, m)]
    for i in bad[:5]:
        print(f"    RTL != model #{i}: rtl={rtl[i]:08x} model={model[i]:08x} {design.describe(cases[i])}")
    if silicon is not None:
        kind_of = {v: k for k, v in MODE_OF_KIND.items()}
        cats = Counter(t.category(c, kind_of.get(c["mode"], 1)) for c, r, s in zip(cases, first, silicon) if not same(r, s))
        if cats:
            print(f"    RTL != silicon by situation: {dict(cats)}")
    return len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100000, help="random vectors, spread over the 4 modes")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--design", choices=("fp4_fedp", "fp4_fedp_dsp"), default="fp4_fedp")
    ap.add_argument("--gpu", help="also run the random vectors on this host's RTX 5090 (ssh)")
    args = ap.parse_args()

    design = DESIGNS[args.design](modes=("nvfp4", "mxfp4", "fp4", "mxfp4_b16"))
    t = design.t
    cases = list(design.directed_cases())
    bad = report("directed", design, cases, [int.from_bytes(o, "little") for o, _ in vl_ref.run(design, cases)])
    for path in sorted((ROOT / "tcgen05/vectors").glob("*.vec")):
        hdr, vecs, cols = t.read_vec_file(str(path))
        mode = MODE_OF_KIND[hdr["kind"]]
        cases = [dict(v, mode=mode) for v in vecs]
        rtl = [int.from_bytes(o, "little") for o, _ in vl_ref.run(design, cases)]
        bad += report(path.stem, design, cases, rtl, cols["rtx5090"])

    rng = random.Random(args.seed)
    cases = [design.random_case(rng) for _ in range(args.n)]
    rtl = [int.from_bytes(o, "little") for o, _ in vl_ref.run(design, cases)]
    silicon = None
    if args.gpu:
        work = ROOT / "build/gpu/pe"
        work.mkdir(parents=True, exist_ok=True)
        silicon = run_cases(args.gpu, cases, work)
    bad += report(f"random_seed{args.seed}", design, cases, rtl, silicon)
    print("PASS" if bad == 0 else f"FAIL: {bad} RTL outputs differ from the model")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
