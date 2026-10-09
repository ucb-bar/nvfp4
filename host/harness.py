"""Shared paths, board link selection, and the frame protocol of fpga/common/uart_harness.sv."""
import glob
import json
import os
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The Vivado 2024.2 install: XILINX_VIVADO, else the directory of VIVADO_SETTINGS (see fpga/scripts/vivado.sh).
VIVADO = Path(os.environ.get("XILINX_VIVADO") or Path(os.environ.get(
    "VIVADO_SETTINGS", Path.home() / "Documents/Xilinx/Vivado/2024.2/settings64.sh")).parent)
PORT = "/dev/serial/by-id/usb-Xilinx_SophiaLake_Bringup_Platform_*-if01-port0"
TSI_BAUD = 921_600

RUN, IDENT, RESULT = 0xA5, 0x49, 0x5A
ID_BYTES = 11


def board_port(port=None):
    """Return port, or the UART of the one connected SophiaLake board."""
    if port:
        return port
    ports = glob.glob(PORT)
    if len(ports) != 1:
        raise SystemExit(f"found {len(ports)} SophiaLake UARTs; select one with --port")
    return ports[0]


def open_tsi(transport, port=None):
    """Return a TSI link: 'direct' (MMIO Verilator model), 'tsi-sim' (UART Verilator model) or 'fpga'."""
    if transport == "direct":
        from compute_sim import ComputeSim
        return ComputeSim()
    from pyuartsi import UARTTSI
    if transport == "tsi-sim":
        from tsi_sim import SimTransport
        return UARTTSI("sim", TSI_BAUD, transport=SimTransport())
    return UARTTSI(board_port(port), TSI_BAUD)


def make_cases(design, n_random, seed):
    rng = random.Random(seed)
    return list(design.directed_cases()) + [design.random_case(rng) for _ in range(n_random)]


def parse_ident(buf: bytes):
    assert len(buf) == ID_BYTES and buf[0] == IDENT, f"bad ident frame {buf.hex()}"
    return dict(id=buf[1:5][::-1], in_bytes=int.from_bytes(buf[5:7], "little"),
                out_bytes=int.from_bytes(buf[7:9], "little"), clk_mhz=buf[9], flags=buf[10])


def check_ident(design, ident):
    want = (design.design_id, design.in_bytes, design.out_bytes)
    got = (ident["id"], ident["in_bytes"], ident["out_bytes"])
    if got != want:
        raise SystemExit(f"bitstream mismatch: FPGA reports {got}, host expects {want}")


def parse_results(buf: bytes, out_bytes: int):
    step = out_bytes + 2
    results = []
    for i in range(0, len(buf) - step + 1, step):
        frame = buf[i:i + step]
        assert frame[0] == RESULT, f"lost frame sync at byte {i}: {frame.hex()}"
        results.append((frame[1:1 + out_bytes], frame[-1]))
    return results


def compare(design, cases, results):
    mismatches = []
    latencies = Counter()
    batch = design.expected_batch(cases) if hasattr(design, "expected_batch") else None
    for i, (case, (got, lat)) in enumerate(zip(cases, results)):
        latencies[lat] += 1
        want = batch[i] if batch else design.expected(case)
        if got != want:
            mismatches.append((i, case, got, want))
    return mismatches, latencies


def report(design, cases, results, mismatches, latencies):
    print(f"{design.name}: {len(results)}/{len(cases)} results, {len(mismatches)} mismatches, "
          f"latency cycles {dict(sorted(latencies.items()))}")
    for i, case, got, want in mismatches[:10]:
        print(f"  case {i}: got {got[::-1].hex()} want {want[::-1].hex()}  {design.describe(case)}")
    if hasattr(design, "spec_report"):
        design.spec_report(cases, results)
    return len(results) == len(cases) and not mismatches


def dump_mismatches(path, mismatches):
    rows = [dict(index=i, case=case, got=got[::-1].hex(), want=want[::-1].hex()) for i, case, got, want in mismatches]
    Path(path).write_text(json.dumps(rows, indent=1))


def load_cases(path):
    return [row["case"] for row in json.loads(Path(path).read_text())]
