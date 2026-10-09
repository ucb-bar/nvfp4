"""Streaming vector device driver. All values passed here are FP32 bit patterns."""
import struct
import time

BASE = 0x1001_0000
STAGE = 0x2001_FF00  # run() operands: the last 256 B of scratch memory
ADD, SUB, MUL, SCALE, AXPY, MAX, SUM, EXP2, RECIP, SOFTMAX, NORMALIZE, RSQRT, SILU, ABS, QUANT = range(15)
ONE, NEG_INF = 0x3f80_0000, 0xff80_0000


def command_beat(operation, x, y, d, groups, mask=0xff, scalar=ONE, old_max=NEG_INF, old_sum=0):
    """The 8 command words at offset 0x20; writing the last one starts the command."""
    return [x, y, d, mask << 16 | groups, scalar, old_max, old_sum, 1 | operation << 4]


class FlashVector:
    def __init__(self, tsi):
        self.tsi, self.base, self.stage = tsi, BASE, STAGE
        if tsi.read_word(BASE) != 0x4656_4543:
            raise RuntimeError(f'no vector unit at 0x{BASE:08x}')
        self.lanes = tsi.read_word(BASE + 4)

    def command(self, operation, x, y, d, groups, mask=0xff, scalar=ONE, old_max=NEG_INF, old_sum=0, timeout=30):
        """Run one command (see command_beat) and wait for it. Raise ValueError if the unit rejects it."""
        beat = command_beat(operation, x, y, d, groups, mask, scalar, old_max, old_sum)
        self.tsi.write_bytes(self.base + 0x20, struct.pack('<8I', *beat))
        deadline = time.monotonic() + timeout
        while (status := self.tsi.read_word(self.base + 8)) & 1:
            if time.monotonic() >= deadline:
                raise TimeoutError('vector command did not finish')
        if status & 2:
            raise ValueError('vector command rejected')

    def run(self, operation, x, y=None, scalar=ONE, old_max=NEG_INF, old_sum=0, mask=None):
        """Run one group of 8 lanes through the staging area. Return the result lanes and the registers."""
        if len(x) != self.lanes or (y is not None and len(y) != self.lanes):
            raise ValueError(f'operands must have {self.lanes} lanes')
        if not ADD <= operation <= ABS:
            raise ValueError('unknown vector operation')
        if mask is None:
            mask = (1 << self.lanes) - 1
        self.tsi.write_bytes(self.stage, struct.pack('<16I', *x, *(y or [0] * self.lanes)))
        self.command(operation, self.stage, self.stage + 32, self.stage + 64, 1, mask, scalar, old_max, old_sum)
        result = struct.unpack('<8I', self.tsi.read_bytes(self.stage + 64, 32))
        reduction, maximum, total, alpha = struct.unpack('<4I', self.tsi.read_bytes(self.base + 0x10, 16))
        return dict(result=list(result), reduction=reduction, max=maximum, sum=total, alpha=alpha)
