"""Host driver for the NVFP4 tensor core (fpga/mmio/nvfp4_tc.sv) over UART TSI.

    with UARTTSI(port, 921_600) as tsi:
        tc = NVFP4TensorCore(tsi)
        tc.set_dims(64, 64, 256)
        tc.load_a(a, sfa)      # a: M rows x K E2M1 codes, sfa: M rows x K/16 scale codes
        tc.load_b(b, sfb)      # b: N columns x K codes,  sfb: N columns x K/16 codes
        tc.run(overwrite=True) # C = A x B   (overwrite=False: C = A x B + C)
        c = tc.read_c()        # M x N FP32 bit patterns
"""
import struct
import time

SYS_BASE = 0x0000_0000
TC_BASE = 0x1000_0000
TM, TN, TK = 64, 64, 256
KSTEP = 32  # K granularity; the core pads a final 32-element chunk to its 64-element width

ID, TILE, ARRAY, CTRL, STATUS, MODE, DIMS, OPS, CYCLES = range(0, 0x24, 4)
PK = 0x0024  # physical reduction width
A, SFA, B, SFB, C = 0x2000, 0x4000, 0x6000, 0x8000, 0xC000
ROW, SROW, CROW = TK // 2, TK // 16, TN * 4  # bytes per row of A/B, scales, C
CHUNK = 1024  # bytes per TSI command
MODES = {"nvfp4": 0, "mxfp4": 1, "fp4": 2, "mxfp4_b16": 3}


def pack_codes(codes) -> bytes:
    """4-bit codes -> bytes, element k in nibble k (low nibble first)."""
    return sum((x & 0xF) << (4 * k) for k, x in enumerate(codes)).to_bytes(len(codes) // 2, "little")


class NVFP4TensorCore:
    def __init__(self, tsi):
        self.tsi, self.base = tsi, TC_BASE
        if self.read(ID) != 0x4E56_5443:
            raise RuntimeError(f"no NVFP4 tensor core at 0x{TC_BASE:08x}")
        array = self.read(ARRAY)
        self.pm, self.pn = array >> 8 & 0xFF, array & 0xFF
        self.pk = self.read(PK)
        self.m, self.n, self.k = TM, TN, TK

    def read(self, off):
        return self.tsi.read_word(self.base + off)

    def write(self, off, value):
        self.tsi.write_word(self.base + off, value)

    def write_bytes(self, off, data):
        for i in range(0, len(data), CHUNK):
            self.tsi.write_bytes(self.base + off + i, data[i:i + CHUNK])

    def read_bytes(self, off, size):
        return b"".join(self.tsi.read_bytes(self.base + off + i, min(CHUNK, size - i)) for i in range(0, size, CHUNK))

    def _write_rows(self, off, stride, rows):
        rows = [r + bytes((-len(r)) % 4) for r in rows]  # K32 scale rows need a final full word
        if all(len(r) == stride for r in rows):
            self.write_bytes(off, b"".join(rows))
        else:
            for i, r in enumerate(rows):
                self.write_bytes(off + stride * i, r)

    def set_dims(self, m, n, k):
        if not (0 < m <= TM and m % self.pm == 0 and 0 < n <= TN and n % self.pn == 0
                and 0 < k <= TK and k % KSTEP == 0):
            raise ValueError(f"dimensions must fit the tile and be multiples of {self.pm}x{self.pn}x{KSTEP}")
        self.m, self.n, self.k = m, n, k
        self.write(DIMS, m << 24 | n << 16 | k)

    def set_mode(self, mode="nvfp4"):
        self.write(MODE, MODES[mode])

    def load_a(self, a, sfa):
        self._write_rows(A, ROW, [pack_codes(r) for r in a])
        self._write_rows(SFA, SROW, [bytes(s) for s in sfa])

    def load_b(self, b, sfb):
        self._write_rows(B, ROW, [pack_codes(c) for c in b])
        self._write_rows(SFB, SROW, [bytes(s) for s in sfb])

    def write_c(self, c):
        self._write_rows(C, CROW, [struct.pack(f"<{len(r)}I", *r) for r in c])

    def read_c(self):
        if self.n == TN:
            raw = self.read_bytes(C, CROW * self.m)
            words = struct.unpack(f"<{TN * self.m}I", raw)
            return [list(words[TN * i:TN * (i + 1)]) for i in range(self.m)]
        return [list(struct.unpack(f"<{self.n}I", self.read_bytes(C + CROW * i, 4 * self.n))) for i in range(self.m)]

    def run(self, overwrite=False, timeout=30.0):
        """C = A x B + C, or C = A x B with overwrite. Returns the cycles the core took."""
        self.write(CTRL, 1 | (2 if overwrite else 0))
        deadline = time.monotonic() + timeout
        while (status := self.read(STATUS)) & 1:
            if time.monotonic() > deadline:
                raise TimeoutError("tensor core did not finish")
        if status & 2:
            raise ValueError(f"DIMS {self.m}x{self.n}x{self.k} rejected")
        return self.read(CYCLES)
