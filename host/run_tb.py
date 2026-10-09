"""Run the tensor-core bus test (fpga/sim/tb_tc.sv) in xsim.

    pixi run python host/run_tb.py
"""
import subprocess
import sys

from harness import ROOT, VIVADO
from vl_ref import read_filelist


def main():
    files, incs, defs = read_filelist(ROOT / "fpga/designs/nvfp4_tsi/sources.f")
    files = [f for f in files if f.stem not in ("board_clk_rst", "sophialake_tsi_top")]
    work = ROOT / "build/sim/tb_tc"
    work.mkdir(parents=True, exist_ok=True)
    bin_ = VIVADO / "bin"
    steps = [
        [bin_ / "xvlog", "-sv", *(v for p in incs for v in ("-i", str(p))), *map(str, files), str(ROOT / "fpga/sim/tb_tc.sv")],
        [bin_ / "xelab", "-timescale", "1ns/1ps", "-s", "tb", "tb_tc"],
        [bin_ / "xsim", "tb", "-R"],
    ]
    for cmd in steps:
        p = subprocess.run(list(map(str, cmd)), cwd=work, capture_output=True, text=True)
        if p.returncode or "ERROR:" in p.stdout:
            print(p.stdout[-4000:], p.stderr[-2000:])
            sys.exit(f"{cmd[0].name} failed")
    lines = [l for l in p.stdout.splitlines() if l.startswith(("ok", "FAIL", "PASS"))]
    print("\n".join(lines))
    sys.exit(0 if lines and lines[-1] == "PASS" else 1)


if __name__ == "__main__":
    main()
