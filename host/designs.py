"""Per-design stimulus, packing and golden models for the NVFP4 FPGA tests.

Each design turns a test case into the IN_BYTES frame its dut_wrap expects and
predicts the OUT_BYTES response bit for bit.
"""
import math
import random
import struct
import sys
from collections import Counter
from fractions import Fraction

from harness import ROOT

E2M1_MAG2 = [0, 1, 2, 3, 4, 6, 8, 12]  # |E2M1| in units of 0.5


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def pack_nibbles(vals) -> int:
    w = 0
    for i, v in enumerate(vals):
        w |= (v & 0xF) << (4 * i)
    return w


def rand_ue4m3(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.05:
        return 0x00                      # zero scale
    if r < 0.12:
        return rng.randrange(1, 8)       # subnormal
    if r < 0.20:
        return rng.choice([0x38, 0x78, 0x7E])  # 1.0, 256, 448 (max finite)
    return rng.randrange(0x7F)                  # every finite code; 0x7F is the only NaN


def rand_e2m1_vec(rng: random.Random, n: int):
    style = rng.random()
    if style < 0.6:
        return [rng.randrange(16) for _ in range(n)]
    if style < 0.75:
        return [rng.choice([0x7, 0xF]) for _ in range(n)]         # +-6, heavy cancellation
    if style < 0.85:
        return [rng.choice([0x0, 0x8, 0x1, 0x9]) for _ in range(n)]  # zeros and +-0.5
    k = rng.randrange(16)
    return [k] * n


def rand_fp32(rng: random.Random) -> int:
    """Return a random FP32 C operand: zeros, moderate and wide-exponent values, random normals or specials."""
    r = rng.random()
    if r < 0.25:
        return 0
    if r < 0.30:
        return 0x80000000                                  # -0
    if r < 0.70:
        return f32_bits(rng.uniform(-32.0, 32.0))
    if r < 0.85:
        return f32_bits(math.ldexp(rng.uniform(-1, 1), rng.randrange(-30, 30)))
    if r < 0.95:
        return (rng.getrandbits(1) << 31) | (rng.randrange(1, 255) << 23) | rng.getrandbits(23)
    return rng.choice([0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001, 0x007FFFFF])


def is_nan(x: int) -> bool:
    return (x >> 23) & 0xFF == 0xFF and x & 0x7FFFFF != 0


def same(x: int, y: int) -> bool:
    """Return whether two FP32 words are equal, with any NaN equal to any NaN."""
    return x == y or (is_nan(x) and is_nan(y))


class DotprodFormal:
    """sashakatne/nvfp4-dotprod-formal-dv, NVFP4 mode: one 16-element block, exact FP32 result."""

    name = "dotprod_formal"
    design_id = b"DPF4"
    in_bytes = 20
    out_bytes = 5
    QNAN = 0x7FC00000

    def random_case(self, rng):
        sa = rand_ue4m3(rng) if rng.random() > 0.01 else 0x7F
        sb = rand_ue4m3(rng) if rng.random() > 0.01 else 0x7F
        return dict(a=rand_e2m1_vec(rng, 16), b=rand_e2m1_vec(rng, 16), sa=sa, sb=sb)

    def directed_cases(self):
        yield dict(a=[0] * 16, b=[0] * 16, sa=0x38, sb=0x38)
        yield dict(a=[0x7] * 16, b=[0x7] * 16, sa=0x7E, sb=0x7E)   # max magnitude
        yield dict(a=[0x7] * 16, b=[0xF] * 16, sa=0x7E, sb=0x7E)   # max negative
        yield dict(a=[0x1] * 16, b=[0x1] * 16, sa=0x01, sb=0x01)   # smallest nonzero
        yield dict(a=[0x7, 0xF] * 8, b=[0x7] * 16, sa=0x38, sb=0x38)  # exact cancellation
        yield dict(a=[0x8] * 16, b=[0x7] * 16, sa=0x38, sb=0x38)   # -0 elements
        yield dict(a=[0x2] * 16, b=[0x2] * 16, sa=0x7F, sb=0x38)   # NaN scale
        yield dict(a=list(range(16)), b=list(range(15, -1, -1)), sa=0x40, sb=0x30)

    def pack(self, c) -> bytes:
        def words(elems, scale):
            w = [pack_nibbles(elems[4 * i:4 * i + 4]) for i in range(4)] + [scale]
            return struct.pack("<5H", *w)
        return words(c["a"], c["sa"]) + words(c["b"], c["sb"])

    @staticmethod
    def _ue4m3(x):
        exp, mant = (x >> 3) & 0xF, x & 7
        if exp == 0:
            return (mant, -9) if mant else (0, 0)
        return (8 + mant, exp - 10)

    def expected(self, c) -> bytes:
        if c["sa"] == 0x7F or c["sb"] == 0x7F:
            return struct.pack("<IB", self.QNAN, 0b0110)  # {sat, invalid, is_nan, is_inf}
        inner = 0
        for x, y in zip(c["a"], c["b"]):
            p = E2M1_MAG2[x & 7] * E2M1_MAG2[y & 7]
            inner += -p if (x ^ y) & 8 else p
        (ga, ka), (gb, kb) = self._ue4m3(c["sa"]), self._ue4m3(c["sb"])
        m = inner * ga * gb
        bits = 0 if m == 0 else f32_bits(math.ldexp(m, ka + kb - 2))  # exact: |m| < 2^24
        return struct.pack("<IB", bits, 0)

    def describe(self, c) -> str:
        return (f"a={pack_nibbles(c['a']):016x} b={pack_nibbles(c['b']):016x} "
                f"sa=0x{c['sa']:02x} sb=0x{c['sb']:02x}")


class AllegroFP4:
    """roundandr/Allegro fp4_dot_prod: 64 elements (4 NVFP4 blocks) + FP32 C.

    reference="mmasim": Allegro's MMA-Sim wrapper (the golden its own tests use).
    reference="silicon" (default): the same, plus the rule measured on an RTX 5090 tensor core
    (host/run_gpu.py): when all 64 products are zero, d = c + (+0) instead of c truncated to the
    scale-derived alignment window. This matched mma.sync on 100,009/100,009 NVFP4 vectors.
    """

    name = "allegro_fp4"
    design_id = b"ALG4"
    in_bytes = 77
    out_bytes = 4
    MODES = {"nvfp4": 0, "mxfp4": 1, "fp4": 2, "mxfp4_b16": 3}

    def __init__(self, modes=("nvfp4",), reference="silicon"):
        self.reference = reference
        sys.path.insert(0, str(ROOT / "third_party/Allegro/src/test/cocotb"))
        import mma_sim_nvfp4_ref as ref
        # The wrapper flags every exp=15, mant!=0 UE4M3 code (0x79-0x7E) as NaN. OCP E4M3FN,
        # torch.float8_e4m3fn and Allegro's own spec make 0x7F the only NaN (0x7E = 448).
        ref.is_ue4m3_nan = lambda raw: (raw & 0x7F) == 0x7F
        self._golden = ref.NVFP4DotMmaSimGolden()
        self.modes = [self.MODES[m] for m in modes]

    def random_case(self, rng):
        mode = rng.choice(self.modes)
        if mode == 0:
            sa = [rand_ue4m3(rng) for _ in range(4)]
            sb = [rand_ue4m3(rng) for _ in range(4)]
        elif mode == 2:
            sa = [rng.randrange(256) for _ in range(4)]
            sb = [rng.randrange(256) for _ in range(4)]
        else:
            sa = [rng.randrange(0xFF) for _ in range(4)]
            sb = [rng.randrange(0xFF) for _ in range(4)]
        return dict(a=rand_e2m1_vec(rng, 64), b=rand_e2m1_vec(rng, 64),
                    sa=sa, sb=sb, c=rand_fp32(rng), mode=mode)

    def directed_cases(self):
        one = [0x38] * 4
        yield dict(a=[0] * 64, b=[0] * 64, sa=one, sb=one, c=0, mode=0)
        yield dict(a=[0x7] * 64, b=[0x7] * 64, sa=[0x7E] * 4, sb=[0x7E] * 4, c=0, mode=0)
        yield dict(a=[0x7] * 64, b=[0xF] * 64, sa=[0x7E] * 4, sb=[0x7E] * 4, c=0, mode=0)
        yield dict(a=[0x1] * 64, b=[0x1] * 64, sa=[0x01] * 4, sb=[0x01] * 4, c=0, mode=0)
        yield dict(a=[0x7, 0xF] * 32, b=[0x7] * 64, sa=one, sb=one, c=f32_bits(1.0), mode=0)
        yield dict(a=[0x2] * 64, b=[0x2] * 64, sa=one, sb=one, c=f32_bits(-64.0), mode=0)
        yield dict(a=[0x2] * 64, b=[0x2] * 64, sa=[0x38, 0x7F, 0x38, 0x38], sb=one, c=0, mode=0)
        yield dict(a=[0x2] * 64, b=[0x2] * 64, sa=one, sb=one, c=0x7F800000, mode=0)
        yield dict(a=[0x2] * 64, b=[0x2] * 64, sa=one, sb=one, c=f32_bits(2.0 ** 100), mode=0)

    def pack(self, c) -> bytes:
        return (pack_nibbles(c["a"]).to_bytes(32, "little") + pack_nibbles(c["b"]).to_bytes(32, "little")
                + bytes(c["sa"]) + bytes(c["sb"]) + struct.pack("<I", c["c"]) + bytes([c["mode"]]))

    def expected(self, c) -> bytes:
        cb = c["c"]
        if self.reference == "silicon" and all(E2M1_MAG2[x & 7] * E2M1_MAG2[y & 7] == 0 for x, y in zip(c["a"], c["b"])):
            nan_sf = any((x & 0x7F) == 0x7F for x in c["sa"] + c["sb"]) if c["mode"] == 0 else \
                any(x == 0xFF for x in (c["sa"][:2] + c["sb"][:2] if c["mode"] == 1 else c["sa"] + c["sb"])) \
                if c["mode"] != 2 else False
            if not nan_sf and not ((cb >> 23) & 0xFF == 0xFF and cb & 0x7FFFFF):
                return struct.pack("<I", 0 if cb == 0x80000000 else cb)
        d = self._golden(pack_nibbles(c["a"]), pack_nibbles(c["b"]),
                         int.from_bytes(bytes(c["sa"]), "little"), int.from_bytes(bytes(c["sb"]), "little"),
                         c["c"], c["mode"])
        return struct.pack("<I", d)

    def describe(self, c) -> str:
        return (f"mode={c['mode']} a={pack_nibbles(c['a']):064x} b={pack_nibbles(c['b']):064x} "
                f"sa={bytes(c['sa']).hex()} sb={bytes(c['sb']).hex()} c=0x{c['c']:08x}")


def ue4m3_value(x):
    """OCP UE4M3 (E4M3FN magnitude): 0x7F is NaN, exp=0 is subnormal. Returns None for NaN."""
    x &= 0x7F
    if x == 0x7F:
        return None
    exp, mant = x >> 3, x & 7
    return Fraction(mant, 8) * Fraction(2) ** -6 if exp == 0 else Fraction(8 + mant, 8) * Fraction(2) ** (exp - 7)


def e2m1_value(x):
    v = Fraction(E2M1_MAG2[x & 7], 2)
    return -v if x & 8 else v


def f32_rne(q: Fraction) -> int:
    """Correctly rounded (RNE) FP32 encoding of an exact rational."""
    if q == 0:
        return 0
    sign, q = (1 << 31) if q < 0 else 0, abs(q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    e = max(e, -126)
    m = q / Fraction(2) ** (e - 23)
    n = m.numerator // m.denominator
    r = m - n
    n += 1 if r > Fraction(1, 2) or (r == Fraction(1, 2) and n & 1) else 0
    if n == 1 << 24:
        n, e = n >> 1, e + 1
    if e > 127:
        return sign | 0x7F800000
    if n < 1 << 23:
        return sign | n  # subnormal
    return sign | ((e + 127) << 23) | (n - (1 << 23))


def ulp_distance(a: int, b: int):
    """Distance in FP32 units in the last place; 0 for two NaNs, None for one NaN."""
    if is_nan(a) or is_nan(b):
        return 0 if is_nan(a) and is_nan(b) else None
    order = lambda x: -(x & 0x7FFFFFFF) if x >> 31 else x
    return abs(order(a) - order(b))


class VortexFEDP:
    """vortexgpgpu/vortex VX_tcu_fedp_tfr, NVFP4: 16 elements, one UE4M3 scale per operand, + FP32 C.

    Bit-exact expectations come from a Verilator build of the same RTL (no NVFP4 golden model ships
    with Vortex); spec_report() measures the RTL against an exact fused dot product rounded once (RNE).
    """

    name = "vortex_fedp"
    design_id = b"VXF4"
    in_bytes = 23
    out_bytes = 4
    NVFP4_ID = 11

    def random_case(self, rng):
        sa = rand_ue4m3(rng) if rng.random() > 0.01 else 0x7F
        sb = rand_ue4m3(rng) if rng.random() > 0.01 else 0x7F
        return dict(a=rand_e2m1_vec(rng, 16), b=rand_e2m1_vec(rng, 16), sa=sa, sb=sb,
                    c=rand_fp32(rng), fmt=self.NVFP4_ID)

    def directed_cases(self):
        f = self.NVFP4_ID
        yield dict(a=[0] * 16, b=[0] * 16, sa=0x38, sb=0x38, c=0, fmt=f)
        yield dict(a=[0x7] * 16, b=[0x7] * 16, sa=0x7E, sb=0x7E, c=0, fmt=f)
        yield dict(a=[0x7] * 16, b=[0xF] * 16, sa=0x7E, sb=0x7E, c=0, fmt=f)
        yield dict(a=[0x2] * 16, b=[0x2] * 16, sa=0x38, sb=0x38, c=f32_bits(1.0), fmt=f)
        yield dict(a=[0x7, 0xF] * 8, b=[0x7] * 16, sa=0x38, sb=0x38, c=f32_bits(1.0), fmt=f)
        yield dict(a=[0x2] * 16, b=[0x2] * 16, sa=0x00, sb=0x38, c=f32_bits(1.0), fmt=f)  # zero scale
        yield dict(a=[0x2] * 16, b=[0x2] * 16, sa=0x01, sb=0x38, c=0, fmt=f)              # subnormal scale
        yield dict(a=[0x2] * 16, b=[0x2] * 16, sa=0x7F, sb=0x38, c=0, fmt=f)              # NaN scale

    def pack(self, c) -> bytes:
        return (pack_nibbles(c["a"]).to_bytes(8, "little") + pack_nibbles(c["b"]).to_bytes(8, "little")
                + bytes([c["sa"], c["sb"]]) + struct.pack("<I", c["c"]) + bytes([c["fmt"]]))

    def expected_batch(self, cases):
        import vl_ref
        return [out for out, _ in vl_ref.run(self, cases)]

    @staticmethod
    def spec(c) -> int:
        cb = c["c"]
        sa, sb = ue4m3_value(c["sa"]), ue4m3_value(c["sb"])
        if sa is None or sb is None or ((cb >> 23) & 0xFF == 0xFF and cb & 0x7FFFFF):
            return 0x7FC00000
        if (cb >> 23) & 0xFF == 0xFF:
            return cb
        exact = sum(e2m1_value(x) * e2m1_value(y) for x, y in zip(c["a"], c["b"])) * sa * sb
        cval = Fraction(bits_f32(cb)) if cb & 0x7FFFFFFF else Fraction(0)
        total = exact + cval
        if total == 0:
            return cb if exact == 0 else 0  # x + (-0) = x; c=-0 with zero products stays -0
        return f32_rne(total)

    @staticmethod
    def scale_class(c):
        def cls(x):
            return "nan" if x == 0x7F else "zero" if x == 0 else "subnormal" if x < 8 else "normal"
        order = ["nan", "zero", "subnormal", "normal"]
        return min(cls(c["sa"]), cls(c["sb"]), key=order.index)

    def spec_report(self, cases, results):
        stats = {}
        for c, (got, _) in zip(cases, results):
            d = ulp_distance(int.from_bytes(got, "little"), self.spec(c))
            k = "nan-mismatch" if d is None else "exact" if d == 0 else "1 ulp" if d == 1 else \
                "<=4 ulp" if d <= 4 else "<=2^10 ulp" if d <= 1024 else ">2^10 ulp"
            stats.setdefault(self.scale_class(c), Counter())[k] += 1
        print("  RTL vs exact fused dot product (single RNE), by scale class:")
        for cls in ("normal", "subnormal", "zero", "nan"):
            if cls in stats:
                print(f"    {cls:9s} {dict(stats[cls])}")

    def describe(self, c) -> str:
        return (f"a={pack_nibbles(c['a']):016x} b={pack_nibbles(c['b']):016x} sa=0x{c['sa']:02x} "
                f"sb=0x{c['sb']:02x} c=0x{c['c']:08x}")


class FP4FEDP(AllegroFP4):
    """fpga/pe/fp4_fedp.sv: 4 dotprod_formal block datapaths + NVIDIA-matching fused accumulate.

    Same frame and modes as allegro_fp4. The golden model is the silicon-calibrated model of
    tcgen05/nvfp4_tcgen05_check.py (rule "rtx5090"), which matches RTX 5090 and Jetson Thor outputs.
    """

    name = "fp4_fedp"
    design_id = b"FED4"

    def __init__(self, modes=("nvfp4",)):
        sys.path.insert(0, str(ROOT / "tcgen05"))
        import nvfp4_tcgen05_check as t
        self.t = t
        self.modes = [self.MODES[m] for m in modes]

    def expected(self, c) -> bytes:
        t, mode = self.t, c["mode"]
        kind = {0: t.KIND_NVFP4, 1: t.KIND_MX32, 3: t.KIND_MX16}.get(mode, t.KIND_MX16)
        v = dict(c) if mode != 2 else dict(c, sa=[127] * 4, sb=[127] * 4)  # FP4: E8M0 1.0 scales
        return struct.pack("<I", t.model(v, kind, "rtx5090"))


class FP4FEDPDSP(FP4FEDP):
    """fpga/pe/fp4_fedp_dsp.sv with N = 2: the fp4_fedp arithmetic with the products in DSP slices.

    dut_wrap runs rows (a, b) against columns (b, a) of one case, so the 16-byte output is
    d[0][0] = a.b (the case), d[0][1] = a.a, d[1][0] = b.b and d[1][1] = b.a.
    """

    name = "fp4_fedp_dsp"
    design_id = b"FEDD"
    out_bytes = 16

    def expected(self, c) -> bytes:
        rows = [(c["a"], c["sa"]), (c["b"], c["sb"])]
        return b"".join(FP4FEDP.expected(self, dict(c, a=x, sa=sx, b=y, sb=sy))
                        for x, sx in rows for y, sy in rows[::-1])


DESIGNS = {"dotprod_formal": DotprodFormal, "allegro_fp4": AllegroFP4, "vortex_fedp": VortexFEDP,
           "fp4_fedp": FP4FEDP, "fp4_fedp_dsp": FP4FEDPDSP}
