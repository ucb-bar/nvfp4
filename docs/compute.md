# Tensor core and vector unit

This page describes the NVFP4 tensor core and the FP32 vector unit of the SoC. See
[transformer.md](transformer.md) for the kernel engine and the transformer layer that uses them.

## Tensor core

The default physical shape is **M × K × N = 2 × 64 × 4**. Eight `fp4_fedp` dot products each consume
64 E2M1 elements and four NVFP4 scale blocks, and update eight FP32 accumulator lanes: 512
element MACs per issue, the NVFP4 rate of one RTX 5090 tensor core. Each computes one
m16n8k64 dot product with NVIDIA's arithmetic, so a K chain matches a chain of native MMAs bit
for bit. `nvfp4_tc` supports power-of-two PM/PN in [2,32]. FPGA capacity limits which are
practical. With the products in DSP slices (the default) the grid needs 20 PM PN slices, so 2 × 4
is the largest shape on the xc7a100t (240 slices).

### Loop order and storage

```text
for column_block in N / PN:
  for k_chunk in ceil(K / 64):            # accumulation loop
    read B[column_block, k_chunk]         # stationary weights
    for row_block in M / PM:              # innermost weight reuse
      read A[row_block, k_chunk]
      read C[row_block, column_block]
      issue C += dot(A, held_B)
    drain all outstanding writes
```

The registered B RAM outputs hold the weights throughout the row loop. B element and scale
RAM read enables assert once per `(column_block,k_chunk)` group, not once per issue.
The banked C RAM is the multi-entry accumulator: eight independent read/write lanes,
512 entries per lane for the default shape. Every result carries its C address through a
writeback FIFO. The drain prevents reading stale C even for M=2, when there is only one
accumulator entry in the reuse loop. There is no dependence on a minimum row count to hide
pipeline latency.

The compute time is `(N/PN) * ceil(K/64) * (M/PM + 8)` cycles, including one weight load and
seven drain cycles per group. A 64 × 64 × 256 tile takes **2,560 cycles**. The small-row case
pays proportionally more drain overhead. The tile memory holds one 64 × 64 × 256 tile. Each
scale RAM word holds the four scale bytes of one 64-element chunk. MXFP4 block32 uses bytes 0
and 1 for the two 32-element halves.

K may be any multiple of 32. When K = 64c + 32, the last chunk zeroes the elements of A past K
and the scales of A and B past K (mode 1: byte 1). The result equals a native MMA on operands
zero-padded to 64 with zero scales. Zero scales matter because a block's exponent comes from its
scales even when its products are zero. A zero scale puts the padding at 2^-252, below every
other term.

### Dot product

`fpga/pe/fp4_fedp.sv` defines the unit. Its four 16-element block sums are the exact datapath of
sashakatne/nvfp4-dotprod-formal-dv (`mul_lane_nvfp4`, `align_nvfp4`, `exact_acc_tree`). The fused
accumulate (`fp4_fedp_acc.sv`) follows rule `rtx5090` of `tcgen05/nvfp4_tcgen05_check.py`: every
term, C included, is cut toward zero at 2^(emax - 35), then the sum is cut toward zero to FP32
once. If all 64 products are zero, d = C with -0 → +0. NaN scales or C give `0x7FFFFFFF`.
Latency is 5 cycles. `host/nvfp4_model.py` is an independent integer model of the same contract.

By default (`nvfp4_tc` parameter `DSP = 1`) each pair of rows is one `fp4_fedp_dsp`, which
computes the products in DSP48E1 slices instead of LUTs. Elements are integers in units of 1/2
(|x| ≤ 12). For one element and one column pair, a slice computes
(b1·2^20 + b0) × (a1·2^10 + a0). The product holds a0·b0, a1·b0, a0·b1 and a1·b1 in 10-bit
fields. The pre-adder forms the first factor. Two slices add two elements and a bias of 512 per
field, so a field never borrows from its neighbour. Eight fields add to a block sum plus 4096,
and flipping bit 12 removes the bias. The 2 × 4 grid uses 128 slices for its 512 products per
cycle, plus 32 for the block-term products in `fp4_fedp_acc`. `DSP = 0` builds the same grid
from `fp4_fedp` units with the products in LUTs.

### Tensor core MMIO

The MMIO core at `0x1000_0000` has read-only register `PK` (`0x24`) = 64. `ARRAY` returns
`0x0204`. DIMS requires M divisible by PM, N by PN, and K by 32. The driver pads the final
two-byte scale row of a K = 64c + 32 tile to a full MMIO word. A bus request carries any words
of one 32-byte beat. A beat of A or B elements (one 64-element chunk of a row) is written or read
in one cycle. Scale and C words take one cycle each.

## Vector unit

`flash_vector_unit.sv` is an eight-lane SIMD unit. It accepts one command through ready/valid,
captures its operands, and holds every output until it is accepted. Each lane
(`vector_lane.sv`) runs the per-element steps on its own `fp32_alu.sv`. One more ALU runs the
steps that combine lanes, in lane order: the MAX and SUM reductions, and the SOFTMAX maximum,
alpha, probability sum and final sum. It also computes the NORMALIZE reciprocal. Because the
combining steps run in lane order, every result equals a serial evaluation in lane order.

`fp32_alu.sv` registers alignment, normalization, rounding and every SFU product. One 32 × 32
integer multiplier serves FP32 multiplication, exp2 and reciprocal. An ALU takes one operation at
a time. A lane gives one result per operation latency, not one per cycle.

| Opcode | Operation | Behavior |
|---|---|---|
| 0 / 1 / 2 | ADD / SUB / MUL | Elementwise x+y, x-y, x*y |
| 3 | SCALE | x × scalar; applies tensor scales or score scaling |
| 4 | AXPY | round(round(x × scalar) + y), for output rescaling and accumulation |
| 5 / 6 | MAX / SUM | Masked row reductions; inactive lanes do not enter the reduction |
| 7 / 8 | EXP2 / RECIP | Base-2 exponential / reciprocal |
| 9 | SOFTMAX | Online row update below |
| 10 | NORMALIZE | x × reciprocal(scalar); zero denominator produces a zero scale |
| 11 / 12 / 13 | RSQRT / SILU / ABS | See [transformer.md](transformer.md#numeric-contract) |
| 14 | QUANT | NVFP4 block quantization; streaming only, below |

Masked result lanes are zero. With an empty mask, MAX reduction is -infinity and SUM is zero.
SOFTMAX takes raw scores, `scalar = softmax_scale * log2(e)`, previous raw maximum `m`,
previous sum `l`, and a mask. Scores and the positive scale must be finite. A mask
implements padding or causal exclusion. Initialize `(m,l)=(-infinity,0)`.

```text
m_new = max(m_old, max(active scores))
alpha = exp2((m_old - m_new) * scalar)      # 0 when l_old = 0
P[j]  = exp2((score[j] - m_new) * scalar)  # 0 when masked
l_new = alpha * l_old + sum(P)
O_new = alpha * O_old + P @ V
O     = O_new / l_new                     # after the last key block
```

An empty initial row returns `m=-infinity, l=0, alpha=0, P=0`. An empty later block retains
its running state and returns `alpha=1`. The caller owns per-row m/l/O storage and submits
one SOFTMAX command per score block. The vector unit can express FP32 P@V using MUL+SUM,
then AXPY and NORMALIZE. Backward attention, dropout, and logsumexp output are not part of
this forward unit.

All vector arithmetic uses FP32 RNE and sign-preserving flush-to-zero for scalar ALU
inputs and results, with canonical NaN `0x7fffffff`. AXPY deliberately uses separate MUL
and ADD rounding. SUM accumulates active lanes in ascending order. NVIDIA production
kernels may use FMA, different reduction trees, and FP16/BF16/FP8/FP4 P conversion before
P@V, so this command contract does not imply bitwise equivalence to every FlashAttention
kernel.

EXP2 uses a 64-entry Q1.31 table with a cubic polynomial over the residual interval [0,1/64).
RECIP uses a 128-entry Q2.30 seed table and two Newton iterations. These are original
approximations, not a reconstruction of NVIDIA's proprietary SFU. `scripts/gen_sfu_tables.py`
regenerates the checked-in constants. `fp32_pkg.sv` contains combinational reference
functions, and the tests check the registered ALU against them and against independent FP32
references.

## Vector MMIO and host use

The vector device is at **0x1001_0000**. A command streams GROUPS groups of 8 FP32 elements from
scratch memory through the unit. Group g reads X + 32g, and Y + 32g for ADD, SUB, MUL and AXPY, and
writes its result lanes to D + 32g. The device has its own scratch port (one 32-byte beat per
cycle), so its operands do not cross the bus, and it runs while the kernel engine and the tensor
core work.

- MASK selects lanes in every group. Other result lanes are 0. MAX and SUM pass x through.
- For MAX and SUM, REDUCTION combines the group reductions left to right: ((r0 ⊕ r1) ⊕ r2) ⊕ ….
- SOFTMAX takes one group.
- QUANT converts GROUPS rows of MASK 16-element blocks, read from X in order, with inverse tensor
  scale SCALAR. The codes of block b of row r go to D + 128r + 8b, low nibble first. The scale
  bytes of row r go to Y + 16r, as whole words, zero past the last block. These are the tensor
  core's A and B layouts. Per block, two lane passes compute x × SCALAR, a scan finds the largest
  |x| (subnormals count as 0), and one more multiply by RNE(1/6) gives the scale before UE4M3
  rounding. The |product| of the largest input equals RNE(max |x| × SCALAR), so the scale needs
  no separate multiply. A block takes about 38 cycles.
- The device rejects a command if X, Y or D is not a 32-byte-aligned scratch address, GROUPS is 0,
  or SOFTMAX has more than one group. It also rejects QUANT if MASK is not 1–16 or SCALAR is not a
  positive finite normal number. QUANT stops with rejected set at a non-finite input or scale.

Writes wait while a command runs. Register reads are served. Whole-word writes only.

| Offset | Register | Contents |
|---|---|---|
| 0x00 / 0x04 | ID / LANES | `0x46564543` (FVEC) / 8 |
| 0x08 | STATUS | busy[0], last command rejected[1] |
| 0x10 / 0x14 / 0x18 / 0x1c | REDUCTION / MAX / SUM / ALPHA | outputs; MAX, SUM and ALPHA of the last group |
| 0x20 / 0x24 / 0x28 | X / Y / D | scratch addresses |
| 0x2c | GROUPS | groups[15:0], MASK[23:16] (QUANT: rows, blocks per row) |
| 0x30 | SCALAR | FP32 scalar |
| 0x34 / 0x38 | OLD_MAX / OLD_SUM | input row state |
| 0x3c | CTRL | start[0], opcode[7:4]; the write starts the command |

Offsets 0x20–0x3c are one 32-byte beat. The kernel engine starts a command with one COPY of a
command buffer from scratch memory. CTRL is the last word written.

```python
from flash_vector import FlashVector, SOFTMAX, NORMALIZE, SCALE

vector = FlashVector(tsi)  # run() stages one group in the last 256 B of scratch memory
row = vector.run(SOFTMAX, score_bits, scalar=scale_log2_bits,
                 old_max=previous_max_bits, old_sum=previous_sum_bits, mask=0xff)
# Use row['alpha'] to rescale O; accumulate row['result'] @ V into O.
normalized = vector.run(NORMALIZE, output_bits, scalar=row['sum'])
vector.command(SCALE, x_address, 0, x_address, 64, scalar=scale_bits)  # 512 elements in place
```

## NVIDIA literature and implementation findings

The [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691) describes maintaining an
unnormalized output numerator and normalizing once at the end. The row recurrence above
implements that formulation. [Dao-AILab's current softmax implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/softmax.py)
uses FP32 state, fast exp2, approximate reciprocal for finalization, and converts P to the
matrix operand type. Its Blackwell path can mix SFU exp2 with software emulation. Its FMA
score scaling and parallel reductions can produce different rounding than this unit.

[NVIDIA's Blackwell Ultra softmax article](https://developer.nvidia.com/blog/making-softmax-more-efficient-with-nvidia-blackwell-ultra/)
identifies MUFU.EX2 on the SFUs as the hardware exponential path. Blackwell Ultra's SFU
throughput changes concern GB300 versus GB200. They do not establish RTX 5090 throughput.
The [PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/)
specifies ex2.approx.f32 accuracy within 2 ULP and rcp.approx.f32 within 1 ULP of correctly
rounded values, along with .ftz behavior. It does not publish their internal polynomial
or table constants. Our CUDA oracle explicitly executes the FTZ PTX variants, and uses
RNE ADD/MUL with FMA contraction disabled.

[NVIDIA's discussion of 4-bit attention](https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/)
explains why softmax P needs a separate row scaling and quantization treatment for NVFP4
P@V. Keeping this unit's P and softmax state in FP32 avoids inventing an unvalidated P
quantization policy for the vector recurrence. The transformer kernel quantizes P with a fixed
tensor scale (see [transformer.md](transformer.md#numeric-contract)).

## Measured results (release 1.0.0)

| Check | Result |
|---|---|
| Full test task, lint, xsim bus, vector handshake, UART co-simulation | PASS |
| Independent matrix regression | 18 runs, 6,016 exact outputs, all four modes, K = 32, 96, 160, 224 tails |
| UART co-simulation matrix comparison | 18,824 / 18,824 outputs match integer model |
| UART co-simulation versus RTX 5090 `m16n8k64` chains (`pixi run sim -r 6 --seed 5 --gpu cell`) | 14,880 / 14,880 outputs match |
| One `fp4_fedp` unit versus RTX 5090 silicon (`host/run_pe.py`) | NVFP4 300,025 / 300,025; MXFP4 b16 13,359 / 13,359; b32 6,640 / 6,641; random 99,998 / 100,000. The 3 differences are known model gaps |
| `fp4_fedp_dsp`, 2 × 2 (`host/run_pe.py --design fp4_fedp_dsp --gpu cell`) | 1,680,136 / 1,680,136 outputs equal the model (stored, directed and 100,000 random cases, four outputs each). The output of each case matches silicon on 420,022 / 420,025, the same 3 model gaps as `fp4_fedp` (`results/fp4_fedp_dsp_gpu.log`) |
| Integer model versus rule `rtx5090` | 420,025 / 420,025 |
| Registered ALU / vector GPU oracle | 20,416 records; ADD/MUL/MAX bitwise match; EXP2 max 2 ULP, RECIP max 1 ULP |
| QUANT rows, 1 to 16 blocks per row | Layout equals `quantize_rows`; bytes outside the rows unchanged |
| Vector commands versus GPU oracle | EXP2 max 2 ULP, RECIP max 1 ULP, SOFTMAX max 2 ULP (bound 6) |
| NVFP4 QK plus vector forward attention | Dense/causal, three key blocks; max absolute error 2.38e-7 versus ordered CUDA oracle |
| Routed board build, Artix-7, 50 MHz | 55,724 LUTs (`nvfp4_tc` 21,909, `flash_vector` 25,803), 20,325 FFs, 81.5 BRAM tiles, 196 DSPs; WNS +1.390 ns |
| Board, 30 random runs (`pixi run real -r 30 --seed 7 --gpu cell`) | 60,416 / 60,416 outputs match the model and RTX 5090 `m16n8k64` chains |
| Board, transformer layer (seeds 3 causal, 11 dense) | PASS, final max absolute error 0; 139,244 and 140,924 cycles |

The critical path, 18.5 ns over 11 logic levels, is in the tensor core: B RAM output → operand
decode → two cascaded DSP slices → field register. QUANT's lane result → UE4M3 scale encoding →
scale register path has +1.460 ns of slack. Vivado keeps the 40-bit field registers of
`fp4_fedp_dsp` in fabric (2,560 FFs): with the constant bias in the sum it does not absorb them
into the second slice. The vector unit, with nine FP32 ALUs and QUANT, takes 25,803 LUTs and 36
DSPs.

The GPU forward oracle uses the same explicit FP32 ordering and native SFU instructions.
It is a CUDA arithmetic reference, not a comparison against a packaged FlashAttention kernel.

## Validation and reproduction

```bash
pixi run lint
pixi run test
pixi run check-gpu                       # RTX 5090 on ssh cell
pixi run sim -r 2 --gpu cell             # NVFP4 MMA chains through UART TSI RTL simulation
pixi run synth-compute                   # post-synthesis estimate for Artix-7
pixi run build                           # full board implementation and bitstream
```

`results/compute_validation.json` records the local checks.
`results/compute_validation_gpu.json` records the GPU checks. The direct
MMIO test checks bank mapping, K = 64c + 32 tails, overwrite/accumulate, all four
matrix modes, pipeline hazards on one C block, unchanged C outside the active tile,
cycle counts, response backpressure, rejected dimensions/opcodes and partial writes.
`tb_vector.sv` checks operand capture, output backpressure, masks and reset cancellation.
The stream test checks that commands of 16 groups equal one-group commands, that MAX and SUM
combine group reductions left to right, and that the device rejects bad addresses and group
counts. The attention test computes QK on the NVFP4 grid, then runs the complete vector
forward recurrence for dense and causal attention. The kernel test checks COPY between buffers
at different word offsets in a beat.

The RTX 5090 oracle compares primitive and compound vector operations. It permits at most
2 ULP difference for exp2/reciprocal primitives and 6 ULP for compounded approximate
operations. Add/multiply/max and other exact vector operations require bitwise agreement,
with NaN payloads treated as equivalent. These are regression bounds, not an exhaustive
proof for every FP32 input. NVFP4 GPU validation runs each 64-element chunk as one native
m16n8k64 MMA (a 32-element tail padded with zero elements and zero scales), and chains
four instructions for K256.

`pixi run synth-compute` writes post-synthesis resource and timing reports to
`build/compute_synth/`. They estimate placement and routing, and do not establish routed
timing or hardware behavior.
