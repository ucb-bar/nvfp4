"""Stream test vectors through a programmed SophiaLake FPGA and check every result.

    pixi run python host/run_fpga.py allegro_fp4 -n 100000
"""
import argparse
import sys
import time
from pathlib import Path

import serial

from designs import DESIGNS
from harness import (IDENT, ID_BYTES, RUN, board_port, check_ident, compare, dump_mismatches, make_cases,
                     parse_ident, parse_results, report)

BAUD = 1_000_000  # BAUD of fpga/common/sophialake_top.sv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design", choices=DESIGNS)
    ap.add_argument("-n", type=int, default=10000, help="random cases on top of the directed ones")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", help="board UART (default: the one SophiaLake by-id port)")
    ap.add_argument("--batch", type=int, default=256, help="frames in flight per write")
    ap.add_argument("--modes", default="nvfp4", help="allegro_fp4 only: comma list of nvfp4,mxfp4,fp4,mxfp4_b16")
    ap.add_argument("--dump", help="write mismatching cases (with FPGA and golden outputs) to this JSON file")
    ap.add_argument("--save-out", help="write every FPGA result frame (OUT_BYTES each, in case order) to this file")
    ap.add_argument("--reference", default="silicon", choices=["silicon", "mmasim"], help="allegro_fp4 only")
    args = ap.parse_args()

    design = DESIGNS[args.design](**({"modes": args.modes.split(","), "reference": args.reference}
                                     if args.design == "allegro_fp4" else {}))
    port = board_port(args.port)
    link = serial.Serial(port, BAUD, timeout=2)
    link.reset_input_buffer()

    link.write(bytes([IDENT]))
    ident = parse_ident(link.read(ID_BYTES))
    print(f"{port}: {ident}")
    check_ident(design, ident)

    t0 = time.time()
    cases = make_cases(design, args.n, args.seed)
    t_gen = time.time() - t0

    step = design.out_bytes + 2
    rx = bytearray()
    t0 = time.time()
    for i in range(0, len(cases), args.batch):
        chunk = cases[i:i + args.batch]
        link.write(b"".join(bytes([RUN]) + design.pack(c) for c in chunk))
        want = len(chunk) * step
        got = link.read(want)
        rx += got
        if len(got) != want:
            print(f"short read at case {i}: {len(got)}/{want} bytes")
            break
    t_run = time.time() - t0

    results = parse_results(bytes(rx), design.out_bytes)
    t0 = time.time()
    mismatches, latencies = compare(design, cases, results)
    t_chk = time.time() - t0

    link.write(bytes([IDENT]))
    flags = parse_ident(link.read(ID_BYTES))["flags"]
    print(f"{len(results)} vectors in {t_run:.1f} s ({len(results) / t_run:.0f}/s over UART); "
          f"golden gen {t_gen:.1f} s, check {t_chk:.1f} s; harness flags=0x{flags:02x}")
    if args.dump:
        dump_mismatches(args.dump, mismatches)
    if args.save_out:
        Path(args.save_out).write_bytes(b"".join(out for out, _ in results))
    ok = report(design, cases, results, mismatches, latencies) and flags == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
