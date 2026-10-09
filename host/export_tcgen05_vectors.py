"""Export the FPGA test vectors, with FPGA, RTX 5090 and model outputs, for tcgen05/nvfp4_tcgen05_check.py.

    pixi run python host/export_tcgen05_vectors.py      # writes tcgen05/vectors/*.vec

File format: one JSON header line, then count 76-byte records (a[32] b[32] sfa[4] sfb[4] c[4]),
then one little-endian uint32 column per entry of header["columns"].
"""
import json
import struct

from designs import DESIGNS
from harness import ROOT, make_cases
from run_gpu import record, run_remote

OUT = ROOT / "tcgen05/vectors"

# (file name, design, n, seed, modes, FPGA output file, Allegro modes kept, description)
SETS = [
    ("allegro_nvfp4_s7", "allegro_fp4", 100000, 7, "nvfp4", "allegro_fp4_n100000_s7.out", {0},
     "roundandr/Allegro NVFP4 FPGA run (seed 7)"),
    ("allegro_mx16_s8", "allegro_fp4", 20000, 8, "mxfp4,fp4,mxfp4_b16", "allegro_fp4_mx_n20000_s8.out", {2, 3},
     "roundandr/Allegro MXFP4 block16 + plain FP4 FPGA run (seed 8)"),
    ("allegro_mx32_s8", "allegro_fp4", 20000, 8, "mxfp4,fp4,mxfp4_b16", "allegro_fp4_mx_n20000_s8.out", {1},
     "roundandr/Allegro MXFP4 block32 FPGA run (seed 8)"),
    ("dotprod_formal_s11", "dotprod_formal", 100000, 11, None, "dotprod_formal_n100000_s11.out", None,
     "sashakatne/nvfp4-dotprod-formal-dv FPGA run (seed 11), 16 elements zero-padded to 64"),
    ("vortex_fedp_s13", "vortex_fedp", 100000, 13, None, "vortex_fedp_n100000_s13.out", None,
     "vortexgpgpu/vortex VX_tcu_fedp_tfr FPGA run (seed 13), 16 elements zero-padded to 64"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, dname, n, seed, modes, fpga_file, keep_modes, desc in SETS:
        kw = {"modes": modes.split(","), "reference": "mmasim"} if dname == "allegro_fp4" else {}
        design = DESIGNS[dname](**kw)
        cases = make_cases(design, n, seed)
        raw = (ROOT / "results" / fpga_file).read_bytes()
        step = design.out_bytes
        fpga_all = [int.from_bytes(raw[i * step:i * step + 4], "little") for i in range(len(cases))]
        idx = [i for i, c in enumerate(cases) if keep_modes is None or c["mode"] in keep_modes]
        recs = [record(cases[i]) for i in idx]
        kinds = {k for k, _ in recs}
        assert len(kinds) == 1, kinds
        kind = kinds.pop()
        work = ROOT / "build/gpu/export" / name
        work.mkdir(parents=True, exist_ok=True)
        rtx = run_remote("cell", {kind: [r for _, r in recs]}, work)[kind]
        cols = {"fpga": [fpga_all[i] for i in idx], "rtx5090": rtx}
        if dname == "allegro_fp4":
            cols["mmasim"] = [int.from_bytes(design.expected(cases[i]), "little") for i in idx]
        hdr = dict(format="nvfp4-vectors-v1", name=name, kind=kind, count=len(idx), columns=list(cols),
                   description=desc, source_indices="case index i of make_cases(design, n, seed) kept in order")
        body = b"".join(r for _, r in recs) + b"".join(struct.pack(f"<{len(v)}I", *v) for v in cols.values())
        (OUT / f"{name}.vec").write_bytes(json.dumps(hdr).encode() + b"\n" + body)
        print(f"{name}: {len(idx)} vectors, kind {kind}, columns {list(cols)}")


if __name__ == "__main__":
    main()
