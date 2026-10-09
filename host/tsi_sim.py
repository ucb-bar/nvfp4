"""A PyUARTSI transport backed by a Verilator simulation of nvfp4_tsi_soc (fpga/sim/tsi_cosim.cpp).

    tsi = UARTTSI("sim", 921_600, transport=SimTransport())

The model runs the UART at 8 clock cycles per bit (6.25 Mbaud) instead of the board's 921,600 baud,
so tile transfers finish quickly. The RTL is the same apart from the divisor.
"""
import subprocess

from pyuartsi.exceptions import TransportTimeoutError

from harness import ROOT
from vl_ref import read_filelist, verilate

CLK_HZ = 50_000_000
BAUD = 6_250_000
CLKS_PER_BIT = (CLK_HZ + BAUD // 2) // BAUD  # the divisor nvfp4_tsi_soc computes


def build():
    filelist = ROOT / "fpga/designs/nvfp4_tsi/sources.f"
    files = [f for f in read_filelist(filelist)[0] if f.stem not in ("board_clk_rst", "sophialake_tsi_top")]
    return verilate(ROOT / "build/vl_tsi", "nvfp4_tsi_soc", filelist, files, ROOT / "fpga/sim/tsi_cosim.cpp",
                    [f"-GCLK_HZ={CLK_HZ}", f"-GBAUD={BAUD}", "-CFLAGS", f"-O2 -DCLKS_PER_BIT={CLKS_PER_BIT}"])


class SimTransport:
    """Implements pyuartsi.transport.Transport on top of the co-simulation process."""

    def __init__(self):
        self._p = subprocess.Popen([str(build())], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def write_all(self, data: bytes) -> None:
        self._p.stdin.write(f"W {data.hex()}\n")

    def read_exact(self, size: int) -> bytes:
        self._p.stdin.write(f"R {size}\n")
        self._p.stdin.flush()
        reply = self._p.stdout.readline().split()
        if not reply or reply[0] != "D":
            raise TransportTimeoutError(f"simulation produced no reply for {size} bytes")
        return bytes.fromhex(reply[1])

    def close(self) -> None:
        if self._p.poll() is None:
            self._p.stdin.write("Q\n")
            self._p.stdin.flush()
            self._p.wait()
