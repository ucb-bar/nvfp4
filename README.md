# NVFP4 tensor core on FPGA

This workspace contains an NVFP4 compute SoC for an AMD Artix-7 FPGA (`xc7a100tfgg484-1`) on a test board. A host controls the SoC over UART TSI with [PyUARTSI](https://github.com/ucb-bar/pyuartsi). The SoC has four devices:

- A 2 × 64 × 4 (M × K × N) NVFP4 tensor core, 512 multiply-accumulates per cycle.
- An eight-lane SIMD FP32 vector unit for softmax, normalization, activations and NVFP4 quantization.
- A kernel engine that runs uploaded programs as a second bus master.
- 128 KiB of dual-port scratch memory.

The tensor core's arithmetic equals NVIDIA tensor cores bit for bit: a K chain on the FPGA gives the same FP32 result as the same chain of `m16n8k64` MMAs on an RTX 5090. The SoC runs a complete transformer layer (RMSNorm, FlashAttention, SwiGLU) from one uploaded program.

The workspace also holds an earlier survey of open-source NVFP4 RTL, tested on the same board against the RTX 5090. See [the survey](docs/survey.md).

An illustrated description of the design, written for readers outside hardware as well, is at <https://ucb-bar.github.io/nvfp4/>.

## Results

Release 1.0.0, board measurements at 50 MHz (Vivado 2024.2, RTX 5090 with CUDA 13.2):

| Check | Result |
|---|---|
| Tensor core versus RTX 5090 `m16n8k64` chains, 30 random runs | 60,416 / 60,416 outputs identical |
| Transformer layer, 16 tokens, seed 3 causal / seed 11 dense | 139,244 / 140,924 cycles (2.8 ms), final error 0 against the quantized reference |
| Resources | 55,724 LUTs (88%), 20,325 FFs, 81.5 BRAM tiles, 196 DSPs |
| Timing | WNS +1.390 ns |

Logs: `results/ws_tsi_1.0.0_fpga_gpu.log` and `results/transformer_fpga_1.0.0.log`.

## Requirements

- Linux x86-64 with [pixi](https://pixi.sh). Pixi installs Python, Verilator, a C++ compiler and the Python packages from `pixi.toml`.
- Vivado 2024.2 for the bitstream build, programming and the xsim bus test. The tasks look for `~/Documents/Xilinx/Vivado/2024.2/settings64.sh`, or the path in `VIVADO_SETTINGS`.
- The test board on its USB-C "FPGA" port, for the board runs.
- Optional: an ssh host with an RTX 5090 and CUDA 13.2 at `/usr/local/cuda-13.2`, for the `--gpu` checks. The examples use the host name `cell`.

## Usage

1. Check the workspace: `pixi run doctor`.
2. Run the simulation tests: `pixi run test`. The first run fetches the third-party RTL and builds the Verilator models.
3. Build the bitstream: `pixi run build`. The output goes to `build/nvfp4_tsi_50mhz/`.
4. Load the bitstream into the board: `pixi run program`.
5. Run the tensor core on the board and compare it with the RTX 5090: `pixi run real -r 30 --gpu cell`.
6. Run the transformer layer on the board: `pixi run transformer`.

`pixi task list` shows all tasks. Other checks:

```bash
pixi run lint                      # Verilator -Wall on the SoC RTL
pixi run sim -r 2 --gpu cell       # UART co-simulation, K chains against RTX 5090 MMA chains
pixi run check-gpu                 # FP32 ALU and vector operations against RTX 5090 PTX
pixi run check-transformer-gpu     # transformer primitives and NVFP4 conversion against RTX 5090
pixi run synth-compute             # post-synthesis area and timing estimate
pixi run python host/run_pe.py -n 100000 --gpu cell    # one fp4_fedp unit against the model and silicon
pixi run python host/run_pe.py --design fp4_fedp_dsp   # the DSP-slice 2 x 2 grid against the model
```

## Architecture

```text
UART TSI ─────┐
              ├→ arbiter → crossbar, 32-byte beats → 0x0000_0000  system registers
kernel engine ┘                                    → 0x1000_0000  NVFP4 tensor core and tile memory
                                                   → 0x1001_0000  FP32 vector unit ──┐
                                                   → 0x1002_0000  kernel engine      │ second port
                                                   → 0x2000_0000  scratch memory ◄───┘
```

- [docs/architecture.html](docs/architecture.html): an illustrated, paper-style description of the whole design, with background for readers outside hardware. It is published at <https://ucb-bar.github.io/nvfp4/>.
- [docs/compute.md](docs/compute.md): the tensor core schedule and arithmetic, the vector unit and its register map, validation and resources.
- [docs/transformer.md](docs/transformer.md): the kernel engine, its instruction set, the transformer compiler, the numeric contract and the board runs.
- [CHANGELOG.md](CHANGELOG.md): the release contents and the earlier builds.

## Board and link

- The clock is a 50 MHz oscillator on W19. W19 is not a clock-capable pin, so `CLOCK_DEDICATED_ROUTE FALSE` is set. An MMCM makes the design clock.
- The host link is the board's USB-C "FPGA" port. Its UART is the board's by-id USB serial port (`harness.PORT` in `host/harness.py`), on T21 (RX) and U21 (TX), the Chipyard UART-TSI pins. The SoC runs it at 921,600 baud 8N1.
- The LEDs show a heartbeat (`led[0]`), TSI activity (`led[1]`), a link error or denied access (`led[2]`) and MMCM lock (`led[3]`).
- Programming is over JTAG and is volatile. A power cycle restores the flash image.
- Other boards can share the JTAG server. `pixi run program` opens only the target with the board's Xilinx serial, `Q4FSQ41IV6A` by default, and only if it holds one `xc7a100t`. `fpga/scripts/program.tcl` shows how to select another board. The host scripts open only the board's by-id serial port, and refuse to run if they find none or more than one.

## Layout

| Path | Contents |
|---|---|
| `pixi.toml` | Environment and tasks |
| `third_party.repos` | Third-party RTL at pinned commits, fetched by `pixi run setup` into `third_party/` |
| `fpga/tsi/` | UART TSI target, TL-UL subset with 32-byte beats, crossbar, arbiter, word/beat adapters |
| `fpga/mmio/` | Bus devices: system registers, tensor core, vector device, kernel engine, scratch memory |
| `fpga/pe/` | The 64-element NVFP4 dot product: `fp4_fedp_dsp.sv` (products in DSP slices, the default), `fp4_fedp.sv` (products in LUTs), `fp4_fedp_acc.sv` (the shared scale, align and FP32 stages) |
| `fpga/compute/` | FP32 ALU, SFU tables, SIMD vector unit and lanes, NVFP4 quantization functions |
| `fpga/soc/` | The SoC (`nvfp4_tsi_soc.sv`), its device fabric and its board top |
| `fpga/common/` | Board clock and reset, UART, FIFO, block RAMs, pin constraints, survey harness top |
| `fpga/designs/<design>/` | File list (`sources.f`) per design; survey designs also have a wrapper and frame sizes |
| `fpga/scripts/` | Vivado build and JTAG programming, Verilator lint |
| `fpga/sim/` | Testbenches and Verilator drivers for the simulation tests |
| `host/` | Drivers (`nvfp4_tc.py`, `flash_vector.py`, `kernel.py`), the transformer compiler (`transformer.py`), reference models, simulation transports and the test scripts `run_*.py` |
| `host/gpu/` | CUDA programs that run the same vectors on the RTX 5090 |
| `tcgen05/` | Stand-alone NVFP4 check for tcgen05 GPUs (B200, B300, Jetson AGX Thor), its test vectors with the RTX 5090 outputs, and the Jetson AGX Thor results |
| `docs/` | Architecture, kernel and survey documents |
| `results/` | Board, simulation and GPU logs |
| `dist/` | Generated: release archive and bitstream. Not in git |
| `build/` | Generated: Vivado projects, bitstreams, simulation models. `pixi run clean` deletes it |
