# Changelog

## 1.0.0 (2026-10-08)

First release. The SoC (`fpga/designs/nvfp4_tsi`) contains:

- A 2 × 64 × 4 NVFP4 tensor core with NVIDIA tensor-core arithmetic, products in DSP slices.
- An eight-lane SIMD FP32 vector unit that streams operands from scratch memory and quantizes rows to NVFP4 (QUANT).
- A kernel engine that runs uploaded programs as a second bus master.
- A TL-UL device fabric with 32-byte beats, and 128 KiB of dual-port scratch memory.
- `host/transformer.py`, a compiler for one RMSNorm + FlashAttention + SwiGLU transformer layer.

Release cleanup:

- Removed `fpga/compute/nvfp4_dot_prod.sv`, an Allegro-derived dot product that no design built.
- Removed dead RTL: the `busy` output of `nvfp4_tsi_soc` and `compute_fabric`, an unused read path in `nvfp4_tc`, `fp32_pkg::fp_sub`, and `tl_pkg::BEAT_WORDS`. QUANT encodes elements with `nvfp4_quant_pkg::encode_element` instead of a local copy.
- The host scripts select the board UART and open TSI links through `harness.open_tsi`. They refuse to run when 0 or 2 or more board UARTs are connected. `run_tsi.py` and `run_fpga.py` no longer take `--baud`.
- All Verilator models build through `vl_ref.verilate`, which also rebuilds when an included `.svh` file changes.
- Fixed `run_pe.py --design fp4_fedp_dsp --gpu`, which packed the 64-element cases as 16-element vectors and failed. The DSP-slice grid now has a direct silicon check: 420,022 / 420,025 case outputs match the RTX 5090, with the same 3 model gaps as `fp4_fedp`.
- New test of the kernel instructions ADD, SHL, SHR, AND, SHL_REG and SLTU.
- The transformer program no longer allocates an unused buffer: 108,896 scratch bytes.
- Added `docs/architecture.html`, an illustrated description of the design with background for readers outside hardware.
- Moved the 2026-09-30 RTL survey from `README.md` to `docs/survey.md`.
- `pixi run test`, `sim`, `lint` and `synth-compute` fetch the third-party RTL first.

### Development builds

Each build ran the transformer fixtures on the board (seed 3 causal, seed 11 dense) with final error 0, and matched the RTX 5090 on every tensor-core output.

| Build | Causal / dense cycles | LUTs | DSPs | WNS | Evidence |
|---|---|---:|---:|---:|---|
| 1.0.0 (2026-10-08) | 139,244 / 140,924 | 55,724 | 196 | +1.390 ns | `results/transformer_fpga_1.0.0.log`, `results/ws_tsi_1.0.0_fpga_gpu.log` |
| QUANT on the vector unit (2026-10-08) | 139,244 / 140,924 | 55,989 | 196 | +1.016 ns | `results/transformer_fpga_quant.log`, `results/ws_tsi_quant_fpga_gpu.log` |
| 32-byte bus, SIMD vector unit, quantizer device at `0x1004_0000` (2026-10-08) | 218,097 / 220,427 | 54,463 | 200 | +1.149 ns | `results/transformer_fpga_wide.log`, `results/ws_tsi_wide_fpga_gpu.log` |
| 32-bit bus, one-lane vector unit, products in DSP slices (2026-10-08) | 764,128 / 771,980 | 32,605 | 168 | +0.789 ns | `results/transformer_fpga_dsp.log`, `results/ws_tsi_dsp_fpga_gpu.log` |
| Products in LUTs, `fp4_fedp` grid (2026-10-07) | 764,128 / 771,978 | 43,847 | 40 | +0.301 ns | `results/transformer_fpga_fedp.log`, `results/ws_tsi_fedp_fpga_gpu.log` |
| 2 × 32 × 8 Allegro-derived grid (2026-10-03) | 762,880 / 770,730 | 47,549 | — | +1.534 ns | `results/transformer_fpga_*_allegro_2x8x32.json`, `results/transformer_fpga_matrix_gpu.log`, `results/transformer_implementation.json`, `results/transformer_build.log`, `results/transformer_program.log` |

Equivalence checks between builds:

- The SIMD vector unit equals the one-lane unit on 6,000 random commands over all operations, with NaN, infinity, subnormal and empty-mask inputs, on all 12 output fields. The one-lane unit used 3,769 LUTs and 4 DSPs.
- QUANT equals the quantizer device on 9,000 random blocks, including non-finite inputs, subnormal and saturated scales, and invalid tensor scales (3,683 rejected by both). The quantizer device used 1,888 LUTs and 4 DSPs and took about 230 cycles per block with its copies. QUANT takes about 38.
- The transformer layer on the 32-byte bus writes the same 52 scratch buffers as on the 32-bit bus (direct RTL, seed 3 causal and dense).
- The `fp4_fedp` grid with products in LUTs (`DSP = 0`) used 31,496 LUTs in `nvfp4_tc`, 9,064 FFs in total, and had its worst path from the A RAM output through the 16 products and block sum to the block-term DSP.
