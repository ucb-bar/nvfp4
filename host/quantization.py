"""NVFP4 finite nearest-even reference and the vector unit's block quantization."""
import struct

import numpy as np

from flash_vector import FlashVector, QUANT

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
E4M3 = np.array([(c & 7) * 2.**-9 if c < 8 else (8 + (c & 7)) * 2.**((c >> 3) - 10)
                 for c in range(127)], dtype=np.float64)


def nearest_even(value, candidates):
    if value <= candidates[0]: return 0
    if value >= candidates[-1]: return len(candidates) - 1
    distances = np.abs(candidates - value)
    ties = np.flatnonzero(distances == distances.min())
    even = ties[ties % 2 == 0]
    return int(even[0] if len(even) else ties[0])


def quantize_block(values, inverse=1.):
    values = np.asarray(values, dtype=np.float32).copy()
    if values.shape != (16,) or not np.isfinite(values).all() or not 0 < inverse < np.inf:
        raise ValueError('quantization requires 16 finite values and a positive finite inverse scale')
    values[np.abs(values) < np.finfo(np.float32).tiny] = np.copysign(0., values[np.abs(values) < np.finfo(np.float32).tiny])
    with np.errstate(over='ignore', under='ignore'):
        scaled = np.float32(values * np.float32(inverse))
        maximum = np.float32(np.max(np.abs(values)) * np.float32(inverse))
        scale_value = np.float32(maximum * np.float32(1 / 6))
    if not np.isfinite(maximum):
        raise ValueError('inverse tensor scale overflows the block')
    scale = nearest_even(float(scale_value), E4M3)
    codes = []
    for value in scaled:
        code = nearest_even(abs(float(value)), E2M1 * E4M3[scale]) if scale else 0
        codes.append(code | (8 if np.signbit(value) else 0))
    packed = sum(code << (4 * i) for i, code in enumerate(codes))
    return packed, scale


def quantize_rows(values, inverse=64.):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % 16:
        raise ValueError('rows must have a reduction width divisible by 16')
    codes = np.zeros(values.shape, dtype=np.uint8)
    scales = np.zeros((len(values), values.shape[1] // 16), dtype=np.uint8)
    for row in range(len(values)):
        for block in range(values.shape[1] // 16):
            packed, scale = quantize_block(values[row, block * 16:(block + 1) * 16], inverse)
            for lane in range(16): codes[row, block * 16 + lane] = packed >> (4 * lane) & 15
            scales[row, block] = scale
    return codes, scales


class Quantizer:
    """One 16-element block through the vector unit's QUANT command, in its staging area."""
    def __init__(self, tsi):
        self.tsi, self.vector = tsi, FlashVector(tsi)

    def run(self, values, inverse=1.):
        if len(values) != 16:
            raise ValueError('a quantizer block has 16 elements')
        stage = self.vector.stage
        self.tsi.write_bytes(stage, struct.pack('<16f', *values))
        inverse_bits = struct.unpack('<I', struct.pack('<f', inverse))[0]
        self.vector.command(QUANT, stage, stage + 128, stage + 64, 1, 1, inverse_bits)
        return int.from_bytes(self.tsi.read_bytes(stage + 64, 8), 'little'), self.tsi.read_word(stage + 128)
