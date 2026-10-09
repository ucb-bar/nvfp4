# Transformer kernels over TSI

`host/transformer.py` compiles a complete pre-norm transformer layer to the FPGA's
programmable MMIO engine. The host uploads FP32 inputs, static NVFP4 weights, scratch
initialization, and one program. After start, the FPGA performs all intermediate
activation quantization, matrix/vector operations, online attention state updates,
and memory transfers. TSI remains available for status polling and result reads.

The kernel supports 16 tokens, one 32-wide head, and a 64-wide SwiGLU FFN:

1. RMSNorm with gamma and epsilon `1e-5`.
2. NVFP4 Q, K and V projections.
3. FlashAttention with two eight-key tiles, optional causal masking, FP32 `(m,l,O)`
   state, NVFP4 QK and PV products, and final normalization. Only a score tile is
   stored; the full attention score matrix is never materialized.
4. NVFP4 output projection and residual addition.
5. RMSNorm, NVFP4 gate/up projections, SiLU and elementwise product, NVFP4 down
   projection, and residual addition.

The program has 4,071 instructions and uses 108,896 scratch bytes. Each of its 371 vector
operations, activation quantization included, is one command beat in scratch memory: the
engine copies it to the vector unit, which streams the whole operand (for example 1,024 SiLU
elements, or 16 rows of NVFP4 blocks) from scratch memory. RMSNorm
rows and attention query pairs are unrolled, so causal masks and addresses are constants. The
matrix array is `M×K×N = 2×64×4` with multiple accumulator entries. Its execution order is
column block, reduction chunk, then activation row block for weight reuse.

## Run

```bash
pixi run test                      # includes the kernel, primitive and layer tests
pixi run check-transformer-gpu
pixi run build
pixi run program
pixi run transformer
```

Alternative fixtures and transports:

```bash
pixi run python host/run_transformer.py --transport direct --seed 11 --noncausal
pixi run python host/run_transformer.py --transport tsi-sim
pixi run python host/run_transformer.py --transport fpga --seed 3
```

`--port` selects a UART explicitly. Otherwise the FPGA transport opens the board's by-id USB
serial port (`harness.PORT`), and refuses to run if it finds none or more than one.
The board runs at 921,600 baud. Programming is volatile JTAG configuration.

To supply a layer, construct `TransformerKernel(x, weights, gamma1, gamma2, causal)`.
The weights dictionary contains `q,k,v,o` with shape `[32,32]`, `gate,up` with shape
`[32,64]`, and `down` with shape `[64,32]`. Inputs have shape `[16,32]`. Both gamma
vectors have shape `[32]`. Upload `kernel.image` to `SCRATCH_BASE`, run
`KernelEngine(tsi).run(kernel.program)`, and read the 16 × 32 FP32 output at scratch address
`kernel.output`. `kernel.addresses` gives the address of every intermediate buffer.
`host/run_transformer.py` performs this procedure and compares every major intermediate with an
independent quantized reference.

## Numeric contract

FP32 vector arithmetic uses nearest-even rounding and flushes subnormal inputs and
outputs to signed zero. Multiplication and addition are separate operations. Matrix
accumulation has the NVIDIA tensor-core contract per 64-element chunk; a K = 32 product runs as
one chunk padded with zero elements and zero scales. See [compute.md](compute.md).

NVFP4 uses E2M1 elements, one unsigned E4M3 scale per 16 elements, and a tensor-level
FP32 scale, as described by [NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html).
This inference kernel chooses a calibrated tensor scale of `1/64` for weights and
ordinary activations, and `1/2688` for softmax probabilities. Block scales are
`RNE_E4M3(RNE(RNE(amax * inverse_tensor_scale) * RNE(1/6)))`; elements use nearest-even
E2M1 conversion with finite saturation. Matrix outputs are rescaled by the product
of operand tensor scales. The vector unit's QUANT command computes them on its eight lanes. It
accepts finite blocks and a positive finite inverse tensor scale, and rejects nonfinite or
overflowing scaled maxima.

Tensor scales are explicit calibration constants, rather than a runtime global amax
pass. Ordinary activation/weight values beyond ±42 saturate at this scale. This is
a small inference prototype, without RoPE, biases, KV caching, dropout, backward,
multiple heads, or arbitrary model dimensions. These can be added to the compiler
without changing the instruction/data upload protocol.

Reciprocal square root normalizes the mantissa to `[1,4)`, performs four FP32 Newton
iterations, and applies the exponent separately. SiLU evaluates
`x * rcp(1 + exp2(-x * log2(e)))`. Against the RTX 5090, the tests observe maximum
errors of 2 ULP for reciprocal square root and 3 ULP for SiLU, and bit-identical
native E4M3/E2M1 conversions for 205 blocks. This compares selected arithmetic
operations, not the output of a packaged NVIDIA FlashAttention kernel.

## Physical FPGA validation

Both complete-layer runs executed on the FPGA board over UART TSI with the release bitstream,
including program and input upload and intermediate and result readback. The reference uses
explicit NVFP4 quantization, 64-element reduction chunks (K = 32 products padded to 64), and FP32
arithmetic order.

| Fixture | Attention | Cycles | Compute time at 50 MHz | Final max absolute error |
|---|---|---:|---:|---:|
| seed 3 | causal | 139,244 | 2.785 ms | 0 |
| seed 11 | dense | 140,924 | 2.818 ms | 0 |

Times exclude TSI program/input upload and result transfer. Cycles include occasional
TSI status-poll contention. RMSNorm intermediates differ from the reference by at
most `4.77e-7`, attention and SwiGLU by at most `5.96e-8`.
Final quantization masks those small approximation differences on these two fixtures;
bit-identical final outputs are observations, not a general guarantee for all inputs.

The routed bitstream meets setup timing at 50 MHz with WNS `+1.390 ns`: 55,724 LUTs, 20,325
registers, 81.5 block RAM tiles, and 196 DSPs. On the same programmed image, 30 random
tensor-core runs (60,416 outputs) match native RTX 5090 `m16n8k64` chains on every output
(`results/ws_tsi_1.0.0_fpga_gpu.log`).

Where the cycles go (UART RTL simulation of seed 3, 139,368 cycles, counted per instruction):

| Cause | Cycles | Share |
|---|---:|---:|
| Waiting for the vector unit, QUANT included (15,951) | 63,162 | 45% |
| Copies to and from the tensor core | 31,882 | 23% |
| Instruction fetch and execute (2 cycles each) | 11,992 | 9% |
| Other bus transfers and tensor-core waits | 32,332 | 23% |

The tensor core computes for 2,592 cycles. QUANT takes about 38 cycles per 16-element block.

Results and build evidence:

- [Causal FPGA run](../results/transformer_fpga_3_causal.json)
- [Dense FPGA run](../results/transformer_fpga_11_dense.json)
- [Board log of both runs](../results/transformer_fpga_1.0.0.log)
- [FPGA/native GPU tensor-core comparison](../results/ws_tsi_1.0.0_fpga_gpu.log)
- [GPU primitive and conversion comparisons](../results/transformer_primitives_gpu.log)

## Memory map

| Base | Device | Size |
|---|---|---|
| `0x00000000` | link/system registers | 4 KB |
| `0x10000000` | NVFP4 tensor core | 64 KB |
| `0x10010000` | vector unit | 64 KB |
| `0x10020000` | kernel engine | 128 KB |
| `0x20000000` | scratch BRAM | 128 KB |

Two masters share the same device fabric: UART TSI and the kernel engine. The fabric moves
32-byte beats. A TSI word occupies its lane of a beat. Scratch memory and the tensor core's A
and B element memories take a whole beat per request. The register devices (system, vector and
kernel registers) take the words of a beat one at a time. The vector unit reads and
writes scratch memory through a second port, outside the fabric. Round-robin
arbitration alternates waiting masters. Ownership is held from request presentation
through response acceptance, including backpressure. There is one outstanding bus
transaction. During execution, callers must leave program, input and working buffers
unchanged; reads of the kernel's status/counters remain available.

### Kernel registers

| Offset | Meaning |
|---|---|
| `0x00` | ID `0x4b45524e` (`KERN`) |
| `0x04` | instruction capacity: 8192 |
| `0x08` | status: busy bit 0, fault bit 1, halted bit 2 |
| `0x0c` | start bit 0 |
| `0x10` | entry instruction index |
| `0x14` | program length in instructions |
| `0x18` | execution cycles, including bus contention |
| `0x1c` | PC; normally next instruction after the faulting instruction |
| `0x20` | fault code: 1 illegal opcode, 2 invalid PC/length, 3 bus denial, 4 WAIT error |
| `0x100` | sixteen 32-bit integer registers |
| `0x1000` | 8192 little-endian 64-bit instruction slots, write only |

Register/program writes are denied while running. Memory and registers persist
between launches; reset clears control and integer registers but not BRAM. A fault
terminates execution and allows a new program upload. Uploaded programs are trusted:
there is no watchdog or memory isolation; a nonterminating program requires reset.

Instructions contain opcode `[63:56]`, destination register `[55:52]`, source 1
`[51:48]`, source 2 `[47:44]`, reserved zeros `[43:32]`, and immediate `[31:0]`.

| Opcode | Operation |
|---|---|
| 0 | HALT |
| 1 | MOVI: `rd = imm` |
| 2 | ADDI: `rd = rs1 + imm` |
| 3 | ADD: `rd = rs1 + rs2` |
| 4 | LOAD: `rd = memory[rs1 + imm]` |
| 5 | STORE: `memory[rs1 + imm] = rs2` |
| 6 | JMP: `pc = imm` |
| 7 | JNZ: branch to `imm` if `rs1 != 0` |
| 8 | COPY: transfer `imm` words from `rs1` to `rs2`, up to 8 per read/write pair |
| 9 | FILL: write `imm` words of value `rs2` at address `rs1`, up to 8 per write |
| 10 | WAIT: poll address `rs1`; busy mask in `imm[15:0]`, error mask in `imm[31:16]` |
| 11/12 | SHL/SHR: shift `rs1` by `imm[4:0]` into `rd` |
| 13 | AND: `rd = rs1 & imm` |
| 14 | SHL_REG: shift `rs1` by `rs2[4:0]` into `rd` |
| 15 | SLTU: `rd = unsigned(rs1) < unsigned(rs2)` |

COPY and FILL preserve source/destination registers; zero count is a no-op. Transfers
require aligned full words. A COPY step ends at the next 32-byte boundary of the source or the
destination, so buffers with the same offset in a beat move 8 words per step; other offsets
are rotated into place. Overlapping COPY regions are unsupported. `Program.write`
reserves registers 12/13, `Program.copy` reserves 10/11, and `Program.wait` reserves 12.
The compiler keeps loop variables in the other registers.

### Vector opcodes 11–14

Vector opcodes 11, 12 and 13 select reciprocal square root, SiLU, and absolute value. Opcode 14,
QUANT, converts rows of 16-element FP32 blocks to the tensor core's operand layout: E2M1 codes,
128 B per row, and UE4M3 scales, 16 B per row. The vector register map is in
[compute.md](compute.md#vector-mmio-and-host-use).
