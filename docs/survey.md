# NVFP4 RTL survey (2026-09-30)

The survey built three open-source NVFP4 datapaths for the Artix-7 test board and tested them on the board over UART against golden models. The same test vectors also ran on a real NVFP4 tensor core (RTX 5090, `mma.sync` sm_120a), which is the referee for both the golden models and the RTL. The results of this survey set the arithmetic contract of the tensor core in [compute.md](compute.md).

## Single-design harness

Each surveyed design runs in a UART test harness (the board top and `uart_harness.sv` in `fpga/common/`) at 1 Mbaud 8N1:

- `0xA5 <IN_BYTES>` runs one vector. The FPGA answers `0x5A <OUT_BYTES> <latency>`.
- `0x49` returns the design ID, the frame sizes, the clock frequency, and the error flags.

The host streams batches of 256 frames. `fpga/designs/<design>/` holds `dut_wrap.sv`, `dut_cfg_pkg.sv` (frame sizes, ID) and `sources.f` (file list, include directories, defines) for each design.

`pixi run build <design> <mhz>` and `pixi run program <design> <mhz>` build and load a design. The design is one of `dotprod_formal`, `allegro_fp4`, `vortex_fedp`. Build `dotprod_formal` at 25 MHz: its one-cycle datapath does not meet timing at 50 MHz. The harness tests have no task. Run them with `pixi run python`:

```bash
pixi run python host/run_sim.py allegro_fp4 -n 20         # xsim of the harness top vs the golden model
pixi run python host/run_fpga.py allegro_fp4 -n 100000 --seed 7 --save-out results/allegro_fp4_n100000_s7.out
pixi run python host/run_gpu.py allegro_fp4 -n 100000 --seed 7 --fpga-out results/allegro_fp4_n100000_s7.out
pixi run python host/run_sim.py allegro_fp4 --cases results/allegro_nvfp4_mismatch.json   # replay in RTL simulation
```

## Results (Vivado 2024.2, RTX 5090 driver 595.91, CUDA 13.2)

| Design | What it computes | Clock | WNS | DUT LUT / FF / DSP | Latency | FPGA vectors | FPGA vs golden | FPGA vs RTX 5090 |
|---|---|---|---|---|---|---|---|---|
| `dotprod_formal`: [sashakatne/nvfp4-dotprod-formal-dv](https://github.com/sashakatne/nvfp4-dotprod-formal-dv) @125db59 | One 16-element block, UE4M3 scales, exact FP32 | 25 MHz | +12.1 ns | 648 / 179 / 1 | 2 | 1,100,016 | 0 mismatches | **100,008 / 100,008 agree** (1M more: golden = GPU) |
| `allegro_fp4`: [roundandr/Allegro](https://github.com/roundandr/Allegro) @ac5b2a4 | 64 elements (4 NVFP4 blocks) + FP32 C, RZ | 50 MHz | +4.3 ns | 3,710 / 1,277 / 4 | 6 | 120,018 | see below | **651 / 100,009 NVFP4** and **278 / 20,009 MX** disagree |
| `vortex_fedp`: [vortexgpgpu/vortex](https://github.com/vortexgpgpu/vortex) @b8c5a24, `VX_tcu_fedp_tfr` | 16 elements, 1 scale/operand + FP32 C | 50 MHz | +1.2 ns | 4,100 / 813 / 21 | 6 | 100,008 | FPGA = RTL (Verilator) on 100,008 | **80,855 / 100,008 disagree** |

Stimulus:
- Elements: uniform nibbles, runs of ±6 (cancellation), zeros/±0.5, and constant vectors.
- Scales: every finite UE4M3 code, with extra zero, subnormal and 448 values, plus about 1% NaN.
- C: 0, −0, uniform(−32, 32), wide exponents (2⁻³⁰…2³⁰), random normals, and specials.
- Directed edge cases per design, such as maximum magnitude, exact cancellation, and NaN scale.

The GPU and FPGA see identical vectors: `run_gpu.py` regenerates them from the same `(design, n, seed)` and compares against the FPGA's saved outputs. NaNs compare equal regardless of payload (the GPU returns `7FFFFFFF`).

### Golden models

| Design | Golden | Verified against silicon |
|---|---|---|
| `dotprod_formal` | `DotprodFormal.expected`: a Python port of the repo's own SV golden. The result is exact, so this equals exact math | 1,100,016 / 1,100,016 |
| `allegro_fp4` | `AllegroFP4(reference="silicon")`: Allegro's MMA-Sim wrapper with two fixes: (1) only 0x7F is a UE4M3 NaN, and (2) **when all 64 products are zero, d = c + (+0)** | 100,009 / 100,009 NVFP4; 20,008 / 20,009 MX |
| `vortex_fedp` | Bit-exact: a Verilator build of the same RTL (`vl_ref.py`), cross-checked with xsim. `spec()`: exact math rounded once to RNE | Vortex has no NVFP4 golden. The RTX 5090 is the reference here; silicon matches single-RNE exact math on only 84,780 / 100,008, because NVIDIA truncates |

`reference="mmasim"` gives the golden without fix (2); that is what Allegro's own tests use.

### GPU harness

`host/gpu/nvfp4_mma.cu` puts each vector in every row of A and every column of B of one `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64…ue4m3`. MXFP4 modes use `ue8m0` with `4X` or `2X`. All 128 outputs are the same dot product, and the kernel flags the vector if any of them disagree (this never happened).

The fragment K layout comes from CUTLASS 4.6.1 `MMA_Traits<SM120_16x8x64_TN_VS>`. It was checked on silicon with 4,228 probes before any real vectors ran: all 64 × 64 k-pairings, scale-block mapping, C pass-through, and 2X scale mapping, all correct.

### Findings

1. **dotprod_formal is correct.** It agrees bit-exactly with both its exact golden and the RTX 5090. That includes subnormal UE4M3 scales and the codes 0x79–0x7E (finite; 11,921 vectors used them). Timing: the core is a single combinational cycle with a post-route critical path of 23.7 ns over 30 logic levels, so it fails at 50 MHz (−3.8 ns) and 40 MHz (−0.08 ns).
2. **MMA-Sim's NVFP4/MXFP4 model is wrong when every product is zero.**
   - MMA-Sim gives each 16-element block an exponent taken from its scales, even when all of that block's products are zero. It then truncates C to that alignment window.
   - The RTX 5090 returns C unchanged, and −0 becomes +0. This happened in 538 / 100,009 NVFP4 vectors and 278 / 20,009 MX vectors.
   - When only some blocks are zero, silicon follows MMA-Sim, including the zero block's exponent (2 vectors show this). So this is a whole-vector bypass, not a per-block rule.
   - MMA-Sim also rounds one subnormal MXFP4 output instead of truncating it (silicon `800b7fff`, MMA-Sim `800b8000`).
   - With fix (2), the golden matches silicon on every vector except that subnormal case.
   - A Jetson AGX Thor (`tcgen05.mma`, sm_110a) and the RTX 5090 (`mma.sync`, sm_120a) agree bit for bit on all 920,079 vectors from `tcgen05/nvfp4_tcgen05_check.py`, in NVFP4 and both MXFP4 block sizes. Thus the all-zero pass-through and every Allegro difference from silicon apply to tcgen05 too.
   - The model with fix (2) still differs from both GPUs on 49 of 600,054 random vectors: an all-zero 16-block with a large scale (15), subnormal outputs that MMA-Sim rounds instead of truncating (28), and normal outputs off by 1 ULP (6). The alignment rule for partial-zero blocks is not fully known.
3. **Allegro's RTL disagrees with silicon on 651 NVFP4 vectors (0.65%) and 278 MX vectors.**
   - 538 NVFP4 and 278 MX vectors: all products are zero, and the RTL truncates C instead of passing it through. It matches MMA-Sim's truncation in 512 of the NVFP4 cases and differs from both in 26. Allegro's own golden cannot catch this.
   - 113 NVFP4 vectors: silicon agrees with MMA-Sim and the RTL is wrong. 103 of these have nonzero products and a tiny C, so the C alignment window in the RTL is too narrow. 9 have all products zero, and 1 has a partly zero block.
   - The one MX vector that looked like an RTL bug against MMA-Sim is in fact correct: the RTL matches silicon there.

   Allegro targets `tcgen05` (B200, sm_100). Only the RTX 5090's `mma.sync` path was tested here. MMA-Sim uses the same function for both, but B200 behavior is unverified.
4. **Golden-model fix (1).** Allegro's wrapper (`mma_sim_nvfp4_ref.is_ue4m3_nan`) treats UE4M3 0x79–0x7E as NaN. Silicon treats them as finite (39,848 Allegro vectors used them), as OCP E4M3FN and Allegro's own spec say.
5. **Vortex's NVFP4 path does not work at `master`.** The FPGA matches the RTL exactly, but the RTX 5090 disagrees on 81% of vectors.
   - In the `VX_tcu_core` configuration (`LANE_MASK=0`, `vld_mask='1`), FP16 works, but in 55% of NVFP4 cases with a nonzero dot product, `d == c`: the products are dropped entirely. The rest are also wrong.
   - The same behavior occurs at `a07e6135`, before the July 2026 FEDP pipeline rework.
   - With `LANE_MASK=1`, as in Vortex's FEDP unit test, the output depends on sampling time. For 16 × (1·1) the answer should be 16, but the output is 0 at the nominal latency and 32 later.
   - Reading the source shows that the UE4M3 hidden bit is hard-wired to 1, so zero and subnormal scales decode wrongly (silicon decodes them per OCP), and NaN scales are never flagged.
   - Vortex ships NVFP4 disabled by default (`VX_CFG_TCU_NVFP4_ENABLE = false`), and its NVFP4 regression checks against a relative tolerance.

### Not built

- `mit-han-lab/fouroversix` `nvfp4_mac`: it depends on Cadence ChipWare `CW_fp_add`/`CW_fp_mult`, which are not open. Replacing them would change the numerics being tested.
- `Danieliao/MiX` NVFP4 systolic array: its PE writes FP32 with a custom exponent offset (bias 14, not 127) and a truncating accumulator, and it is checked with a 1e-6 relative tolerance. It is a sound area/power model, but not an IEEE NVFP4 unit.

`third_party/` keeps the `fouroversix` and `MiX` clones from the survey. `third_party.repos` does not pin them, and no task uses them.
