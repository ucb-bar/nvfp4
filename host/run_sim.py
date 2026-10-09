"""Simulate sophialake_top + one design in xsim and check it against the golden model.

    pixi run python host/run_sim.py dotprod_formal -n 20
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from designs import DESIGNS
from vl_ref import read_filelist
from harness import (ROOT, VIVADO, ID_BYTES, check_ident, compare, load_cases, make_cases, parse_ident,
                     parse_results, report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design", choices=DESIGNS)
    ap.add_argument("-n", type=int, default=20, help="random cases on top of the directed ones")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cases", help="replay the cases in a --dump JSON file instead of generating new ones")
    ap.add_argument("--modes", default="nvfp4", help="allegro_fp4 only")
    args = ap.parse_args()

    design = DESIGNS[args.design](**({"modes": args.modes.split(",")} if args.design == "allegro_fp4" else {}))
    dsgn = ROOT / "fpga/designs" / args.design
    work = ROOT / "build/sim" / args.design
    work.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.cases) if args.cases else make_cases(design, args.n, args.seed)
    with open(work / "vectors.txt", "w") as f:
        f.write(f"{len(cases)}\n")
        for c in cases:
            f.write("\n".join(f"{b:02x}" for b in design.pack(c)) + "\n")

    files, incs, defs = read_filelist(dsgn / "sources.f")
    common = ROOT / "fpga/common"
    srcs = [dsgn / "dut_cfg_pkg.sv", *files, dsgn / "dut_wrap.sv",
            *(common / f for f in ("uart_rx.sv", "uart_tx.sv", "uart_harness.sv", "board_clk_rst.sv", "sophialake_top.sv")),
            ROOT / "fpga/sim/tb_top.sv"]
    bin_ = VIVADO / "bin"
    steps = [
        [bin_ / "xvlog", "-sv", *sum((["-i", str(i)] for i in incs), []),
         *sum((["-d", d] for d in defs), []), *map(str, srcs)],
        [bin_ / "xvlog", str(VIVADO / "data/verilog/src/glbl.v")],
        [bin_ / "xelab", "-L", "unisims_ver", "-timescale", "1ns/1ps", "-s", "tb", "tb_top", "glbl"],
        [bin_ / "xsim", "tb", "-R", "--testplusarg", f"VEC={work / 'vectors.txt'}",
         "--testplusarg", f"OUT={work / 'rx.txt'}"],
    ]
    for cmd in steps:
        p = subprocess.run(list(map(str, cmd)), cwd=work, capture_output=True, text=True)
        if p.returncode or "ERROR:" in p.stdout:
            print(p.stdout[-4000:], p.stderr[-2000:])
            sys.exit(f"step failed: {cmd[0].name}")
    print([l for l in p.stdout.splitlines() if l.startswith("TB")][-1])

    rx = bytes(int(l, 16) for l in (work / "rx.txt").read_text().split())
    ident = parse_ident(rx[:ID_BYTES])
    print(f"ident: {ident}")
    check_ident(design, ident)
    results = parse_results(rx[ID_BYTES:], design.out_bytes)
    mismatches, latencies = compare(design, cases, results)
    ok = report(design, cases, results, mismatches, latencies)
    if args.cases:
        fpga = [row["got"] for row in json.loads(Path(args.cases).read_text())]
        same = sum(got[::-1].hex() == f for (got, _), f in zip(results, fpga))
        print(f"RTL simulation matches the FPGA output on {same}/{len(fpga)} replayed cases")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
