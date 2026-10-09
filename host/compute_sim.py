"""Direct MMIO Verilator transport for fast compute tests; no UART serialization."""
import subprocess

from harness import ROOT
from vl_ref import read_filelist, verilate


def build():
    filelist = ROOT / 'fpga/designs/nvfp4_tsi/sources.f'
    board = {'board_clk_rst', 'sophialake_tsi_top', 'nvfp4_tsi_soc', 'uart_tsi', 'uart_rx', 'uart_tx'}
    files = [p for p in read_filelist(filelist)[0] if p.stem not in board] + [ROOT / 'fpga/sim/compute_test_top.sv']
    return verilate(ROOT / 'build/vl_compute', 'compute_test_top', filelist, files, ROOT / 'fpga/sim/compute_sim.cpp')


class ComputeSim:
    def __init__(self):
        self.process = subprocess.Popen([str(build())], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def command(self, text):
        self.process.stdin.write(text + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('compute simulation terminated')
        return [int(v, 16) for v in line.split()]

    def access(self, write, address, value=0, mask=15):
        return self.command(f'B {int(write):x} {address:x} {value:x} {mask:x}')

    def read_word(self, address):
        value, denied, _ = self.access(False, address)
        if denied:
            raise ValueError(f'denied read at {address:08x}')
        return value

    def write_word(self, address, value):
        _, denied, _ = self.access(True, address, value)
        if denied:
            raise ValueError(f'denied write at {address:08x}')

    def write_bytes(self, address, values):
        if address % 4 or len(values) % 4:
            raise ValueError('MMIO transfers must contain aligned full words')
        for i in range(0, len(values), 4):
            self.write_word(address + i, int.from_bytes(values[i:i + 4], 'little'))

    def read_bytes(self, address, size):
        return b''.join(self.read_word(address + i).to_bytes(4, 'little') for i in range(0, size, 4))

    def close(self):
        self.process.stdin.write('Q\n')
        self.process.stdin.flush()
        self.process.wait(timeout=10)
        if self.process.returncode:
            raise RuntimeError('compute simulation failed')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
