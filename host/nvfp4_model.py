"""Independent integer model of the tensor core's 64-element dot product (fpga/pe/fp4_fedp.sv).

The arithmetic is that of NVIDIA tensor cores (RTX 5090, Jetson Thor), as calibrated by rule
"rtx5090" of tcgen05/nvfp4_tcgen05_check.py: exact 16-element block sums, every term (C included)
cut toward zero at 2^(emax - 35), one final cut toward zero to FP32.
"""
from designs import E2M1_MAG2


def pack_rz(value, exponent):
    if value == 0:
        return 0
    sign = 0x8000_0000 if value < 0 else 0
    value = abs(value)
    leading = value.bit_length() - 1
    unbiased = leading + exponent
    if unbiased > 127:
        return sign | 0x7f80_0000
    if unbiased < -126:
        shift = -149 - exponent
        mantissa = value >> shift if shift >= 0 else value << -shift
        return sign | mantissa if mantissa else 0
    shift = leading - 23
    mantissa = value >> shift if shift >= 0 else value << -shift
    return sign | (unbiased + 127) << 23 | (mantissa & 0x7f_ffff)


def scale(raw, mode):
    """Scale code -> (S, e) with value S / 8 x 2^e and 1 <= S / 8 < 2 where possible; None for NaN."""
    if mode == 2:
        return 8, 0
    if mode in (1, 3):
        if raw == 255:
            return None
        return (4, -126) if raw == 0 else (8, raw - 127)
    raw &= 127
    if raw == 127:
        return None
    if raw >> 3:
        return 8 + (raw & 7), (raw >> 3) - 7
    return (raw & 7, -6) if raw & 7 else (0, -126)


def dot(a, b, sa, sb, c, mode=0):
    """d = C + a . b over 64 elements and 4 scale codes per operand (mode 1: codes 0 and 1)."""
    terms, nonzero = [], False
    for g in range(4):
        sigma = 0  # sum of 16 products, in units of 1/4
        for x, y in zip(a[16*g:16*g+16], b[16*g:16*g+16]):
            product = E2M1_MAG2[x & 7] * E2M1_MAG2[y & 7]
            nonzero |= product != 0
            sigma += -product if (x ^ y) & 8 else product
        pa, pb = scale(sa[g // 2 if mode == 1 else g], mode), scale(sb[g // 2 if mode == 1 else g], mode)
        if pa is None or pb is None:
            return 0x7fff_ffff
        terms.append((sigma * pa[0] * pb[0], pa[1] + pb[1]))  # T x 2^(X - 8)
    c_exp, c_frac = (c >> 23) & 255, c & 0x7f_ffff
    if c_exp == 255:
        return 0x7fff_ffff if c_frac else c
    if not nonzero:
        return 0 if c == 0x8000_0000 else c
    c_sig = c_frac | (0x80_0000 if c_exp else 0)  # Sc x 2^(Xc - 23)
    c_x = c_exp - 127 if c_exp else -126
    emax = max([c_x] + [x for _, x in terms])
    total = 0
    for t, x in terms:
        aligned = (abs(t) << 27) >> (emax - x)  # units of 2^(emax - 35)
        total += -aligned if t < 0 else aligned
    aligned = (c_sig << 12) >> (emax - c_x)
    total += -aligned if c >> 31 else aligned
    return pack_rz(total, emax - 35)


def chunk(a, b, sa, sb, lo, mode):
    """Operands of the 64-element step at K offset lo as the hardware sees them. A step with only 32
    elements left is padded with zero elements and zero scales (mode 1 keeps scale 0 only)."""
    n = min(64, len(a) - lo)
    keep = 4 if n == 64 else 1 if mode == 1 else 2
    pad = [0] * (64 - n)
    codes = lambda s: [x if g < keep else 0 for g, x in enumerate((list(s[lo//16:lo//16 + 4]) + [0] * 4)[:4])]
    return list(a[lo:lo+64]) + pad, list(b[lo:lo+64]) + pad, codes(sa), codes(sb)


def tile(a, b, sa, sb, c, mode=0, overwrite=False):
    """C (M x N FP32 bits) after the tensor core runs A (M x K codes) x B (N x K codes), K a
    multiple of 32, as a chain of 64-element steps."""
    out = [[0 if overwrite else value for value in row] for row in c]
    for lo in range(0, len(a[0]), 64):
        for i in range(len(a)):
            for j in range(len(b)):
                out[i][j] = dot(*chunk(a[i], b[j], sa[i], sb[j], lo, mode), out[i][j], mode)
    return out
