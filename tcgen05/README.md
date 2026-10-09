# NVFP4 check on tcgen05 GPUs (B200, GB200, B300, Jetson AGX Thor)

`nvfp4_tcgen05_check.py` runs NVFP4 and MXFP4 test vectors as real `tcgen05.mma` instructions and compares the results with reference models. It is a single file that needs only the Python standard library, a suitable `nvcc`, and one of these GPUs:

| GPU | Compute capability | Target | CUDA |
|---|---|---|---|
| B200, GB200 | 10.0 | `sm_100a` | ≥ 12.8 |
| B300, GB300 | 10.3 | `sm_103a` | ≥ 12.9 |
| Jetson AGX Thor | 11.0 | `sm_110a` | ≥ 13.0 |
| Jetson AGX Thor, pre-CUDA-13 toolkit | 10.1 | `sm_101a` | 12.8 – 12.9 (CUDA 13 rejects `sm_101`) |

The script detects the GPU through the CUDA driver API (`libcuda`), so it does not need `nvidia-smi`, and picks the newest installed `nvcc` that supports the target. Consumer Blackwell has no tcgen05: RTX 50xx (sm_120) and DGX Spark (sm_121) use warp-level `mma.sync` instead (`host/gpu/nvfp4_mma.cu` in the NVFP4 workspace, not in this bundle). Thor is the reverse: the warp-level block-scaled `mma` is rejected for `sm_110`, and only the tcgen05 form exists.

## Quick start

```bash
tar xzf nvfp4_tcgen05_bundle.tar.gz && cd tcgen05
nvcc --version                     # B200: CUDA >= 12.8, B300: >= 12.9, Thor: >= 13.0
python3 nvfp4_tcgen05_check.py --vectors vectors/*.vec    # ~5 min on B200, longer on Thor
python3 nvfp4_tcgen05_check.py -n 200000                  # extra random NVFP4 vectors
```

Then send back `tcgen05_results/summary.json`. If calibration fails, rerun with `-v` and send the full terminal output instead.

## Run

```bash
python3 nvfp4_tcgen05_check.py                                  # probes + 100k random NVFP4 vectors (~1 min)
python3 nvfp4_tcgen05_check.py --formats nvfp4,mx16,mx32 -n 200000
python3 nvfp4_tcgen05_check.py --vectors vectors/*.vec          # the exact vectors run on the FPGA and RTX 5090
```

Copy the whole `tcgen05/` directory (about 27 MB with `vectors/`) to run the last command. Results go to `tcgen05_results/`. Please send back `summary.json` (and the `*.<arch>.out` files if convenient).

On Thor (JetPack 7), the toolkit is in `/usr/local/cuda-13.0` and is often not on `PATH`. The script finds it there, or you can pass `--nvcc /usr/local/cuda-13.0/bin/nvcc`. CUDA 13.0 and 13.2 give the same tcgen05 code for `sm_110a`. Thor has 20 SMs, so the full `--vectors` run takes a few minutes, mostly in the Python reference models on the Arm CPU.

Useful flags:
- `--nvcc /path/to/nvcc` and `--arch sm_110a` override auto-detection.
- `-v` prints failing probes.
- `--probe-only` stops after calibration.

## What it does

1. The script compiles a small raw-PTX kernel. Each test vector is one `tcgen05.mma.cta_group::1` with M=128, N=32, K=64, with the accumulator preloaded with C. Every row of A and every column of B holds the same vector, so all 4,096 outputs must be equal, and the kernel flags the vector if they are not.
2. **Calibration.** The script runs 4,237 known-answer probes for each of four operand-layout variants. Each variant runs in its own process, which protects against hangs and faults. The probes cover k-pairing, scale-block mapping, element decoding and C. The script continues only if exactly one variant passes every probe. None of the probes depend on the behavior under test.
3. It compares the GPU output with these references:

| Reference | Meaning |
|---|---|
| `mmasim` | MMA-Sim's algorithm (35 fractional bits, RZ), ported to pure Python. It matches MMA-Sim bit-for-bit on 120,018 vectors |
| `rtx5090` | `mmasim` plus the rule measured on an RTX 5090: if all 64 products are zero, d = c + (+0) |
| `file:fpga` | Allegro / dotprod_formal / Vortex RTL output from the FPGA runs (`--vectors` only) |
| `file:rtx5090` | Actual RTX 5090 `mma.sync` output for the same vector (`--vectors` only) |
| `file:mmasim` | MMA-Sim via torch (`--vectors`, Allegro sets only) |

The script groups disagreements by situation: all products zero, one 16-block all zero, |C| < 2⁻²⁴|dot|, C is inf/NaN, or other.

## Reading the result

- **GPU == rtx5090 everywhere:** this tcgen05 part shares the RTX 5090 semantics. Allegro's 651 NVFP4 disagreements with silicon then apply to its B200 target too. Thor gives this result, apart from the model gap in vector 15 (see Status).
- **GPU == mmasim everywhere:** the zero-product pass-through is specific to sm_120. MMA-Sim is right for `tcgen05`, and Allegro's RTL is wrong only in the 113 vectors where it also disagrees with MMA-Sim.
- **B200 and Thor differ from each other:** also new information. Thor is a separate compute capability (11.0), even though it uses the same tcgen05 instruction family.
- **Anything else** is new information. The summary lists the first 50 disagreements per reference, with the raw vectors.

Known residual: in `allegro_mx32_s8`, one vector with a subnormal output (MXFP4 block32, index 5806) differs between the RTX 5090 and both models, so a `rtx5090`-model mismatch there is expected.

## Files

- `nvfp4_tcgen05_check.py`: the script.
- `vectors/*.vec`: written by `host/export_tcgen05_vectors.py` in the NVFP4 workspace. Each file has one JSON header line, then 76-byte records (`a[32] b[32] sfa[4] sfb[4] c[4]`, where element k is nibble k), then one uint32 column per reference.

| File | Vectors | Format |
|---|---|---|
| `allegro_nvfp4_s7.vec` | 100,009 | NVFP4 |
| `allegro_mx16_s8.vec` | 13,359 | MXFP4 block16 + plain FP4 |
| `allegro_mx32_s8.vec` | 6,641 | MXFP4 block32 |
| `dotprod_formal_s11.vec` | 100,008 | NVFP4, 16 elements zero-padded to 64 |
| `vortex_fedp_s13.vec` | 100,008 | NVFP4, 16 elements zero-padded to 64 |

## Status

The kernel ran on a Jetson AGX Thor (Jetson Linux R38.4, CUDA 13.0) on 2026-10-07:

- Calibration picked variant 0: 4,237 of 4,237 probes correct. Variants 1 and 3 fault, and variant 2 gets 4,109.
- Random NVFP4 run (seed 1, 100,018 vectors): GPU == `rtx5090` on 100,017. All 464 disagreements with `mmasim` are "all products zero": Thor passes C through, the same as the RTX 5090.
- The one disagreement with `rtx5090` (vector 15, "some 16-block all zero": an all-zero block with scale 448 and C ≈ −3e-10) is a gap in the model. The RTX 5090 gives the same output as Thor (`0x423fffff`), so Thor and the RTX 5090 agree on all 100,018 vectors.
- Full run (`--vectors vectors/*.vec`, then `--formats nvfp4,mx16,mx32 -n 200000`): Thor and the RTX 5090 agree bit for bit on all 920,079 vectors: the 320,025 FPGA and RTX 5090 vectors and 600,054 random vectors in NVFP4, MXFP4 block 16 and MXFP4 block 32. The random sets were also run on the RTX 5090 for this comparison.
- `rtx5090` still differs from both GPUs on 49 random vectors: 15 with an all-zero 16-block (the model lets the zero block set the alignment exponent), 28 subnormal outputs (MMA-Sim rounds, the GPUs truncate), and 6 normal outputs off by 1 ULP for a reason not yet known.

The kernel has not run on B200 or B300. Builds for `sm_100a`, `sm_103a` and `sm_110a` contain `UTCOMMA`, `UTCBAR` and `UTCATOMSWS` with CUDA 13.0 and 13.2. GPU detection, toolchain selection, calibration, file parsing and the comparisons were also tested with a mocked GPU.

If calibration fails on your machine, the `-v` output is what's needed to fix it. Descriptor and TMEM layouts follow CUTLASS 4.6.1 (`mma_sm100_desc.hpp`, `mma_traits_sm100_frag.hpp`).
