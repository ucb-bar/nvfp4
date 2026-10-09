#!/usr/bin/env python3
"""Check NVFP4 / MXFP4 tcgen05 tensor-core arithmetic on datacenter-family Blackwell:
B200/GB200 (sm_100a), B300/GB300 (sm_103a) and Jetson AGX Thor (sm_110a).

Every test vector is one 64-element block-scaled dot product plus an FP32 accumulator C, which is
the arithmetic of one K=64 step of `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale` (NVFP4:
E2M1 elements, one UE4M3 scale per 16 elements). Each vector runs as a real tcgen05.mma, and the
outputs are compared with two built-in reference models:

  mmasim   MMA-Sim's algorithm (nv_fused_dot_add_with_block_scale, 35 fractional bits, RZ),
           with 0x7F as the only UE4M3 NaN.
  rtx5090  mmasim plus the rule measured on an RTX 5090 (sm_120a mma.sync): when all 64 products
           are zero, d = c + (+0) (C passes through instead of being truncated). This matched
           mma.sync on 100,009/100,009 NVFP4 vectors.

Requirements: Python 3.8+ (standard library only), an nvcc that supports the GPU's tcgen05 target
(sm_100a: CUDA >= 12.8, sm_103a: CUDA >= 12.9, Thor sm_110a: CUDA >= 13.0, pre-CUDA-13 Thor sm_101a:
CUDA 12.8-12.9), and the GPU. Consumer Blackwell (RTX 50xx sm_120, DGX Spark sm_121) has no tcgen05.

    python3 nvfp4_tcgen05_check.py                       # probes + 100k random NVFP4 vectors
    python3 nvfp4_tcgen05_check.py -n 1000000 --seed 3
    python3 nvfp4_tcgen05_check.py --vectors vectors/*.vec # exported FPGA/RTX 5090 vector sets
    python3 nvfp4_tcgen05_check.py --probe-only

Before any real vector runs, known-answer probes check the kernel's operand layouts. A few layout
variants are tried (each in its own process) and the script refuses to continue unless exactly one
variant passes every probe. Results go to ./tcgen05_results/ (summary.json plus raw outputs).
"""
import argparse
import ctypes
import glob
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

# --------------------------------------------------------------------------------------------
# Number formats
# --------------------------------------------------------------------------------------------

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
KIND_NVFP4, KIND_MX16, KIND_MX32 = 0, 1, 2  # (ue4m3, 16) (ue8m0, 16) (ue8m0, 32)
KIND_NAMES = {KIND_NVFP4: "NVFP4 (e2m1, ue4m3/16)", KIND_MX16: "MXFP4 (e2m1, ue8m0/16)",
              KIND_MX32: "MXFP4 (e2m1, ue8m0/32)"}
QNAN = 0x7FFFFFFF


def f2b(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def b2f(b):
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def to_f32_bits(x):
    """Round a double to FP32 (as torch does on conversion), with overflow to inf."""
    try:
        return f2b(x)
    except OverflowError:
        return 0xFF800000 if x < 0 else 0x7F800000


def is_nan_bits(b):
    return (b >> 23) & 0xFF == 0xFF and (b & 0x7FFFFF) != 0


def is_inf_bits(b):
    return (b >> 23) & 0xFF == 0xFF and (b & 0x7FFFFF) == 0


def e2m1(n):
    v = E2M1[n & 7]
    return -v if n & 8 else v


def ue4m3(x):
    """OCP E4M3FN magnitude; 0x7F (low 7 bits) is the only NaN (returns None)."""
    x &= 0x7F
    if x == 0x7F:
        return None
    e, m = x >> 3, x & 7
    return (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)


def ue8m0(x):
    return None if x == 0xFF else 2.0 ** (x - 127)


# --------------------------------------------------------------------------------------------
# Reference models (pure-Python port of MMA-Sim's nv_fused_dot_add_with_block_scale)
# --------------------------------------------------------------------------------------------

def _sig_exp(x, min_exp):
    s, e = math.frexp(x)
    s, e = s * 2, e - 1
    if e < min_exp:
        s *= 2.0 ** (e - min_exp)
        e = min_exp
    if s == 0.0:
        e = -126
    return s, e


def _fused_sum(sigs, exps, nfrac):
    emax = max(exps)
    total = 0.0
    for s, e in zip(sigs, exps):
        total += math.trunc(s * 2.0 ** (nfrac + e - emax)) * 2.0 ** -nfrac
    return total, emax


def model(v, kind, rule="mmasim", nfrac=35):
    """v: dict(a=[64 nibbles], b=[64 nibbles], sa=[4], sb=[4], c=fp32 bits). Returns FP32 bits."""
    a, b, cbits = v["a"], v["b"], v["c"]
    nsf = 2 if kind == KIND_MX32 else 4
    if kind == KIND_NVFP4:
        sa, sb, smin = [ue4m3(x) for x in v["sa"]], [ue4m3(x) for x in v["sb"]], -6
    else:
        sa, sb, smin = [ue8m0(x) for x in v["sa"][:nsf]], [ue8m0(x) for x in v["sb"][:nsf]], -126
    if None in sa or None in sb or is_nan_bits(cbits):
        return QNAN
    if is_inf_bits(cbits):
        return cbits
    if rule == "rtx5090" and all(E2M1[x & 7] * E2M1[y & 7] == 0 for x, y in zip(a, b)):
        return 0 if cbits == 0x80000000 else cbits
    block = 64 // nsf
    sc, ec = _sig_exp(b2f(cbits), -126)
    sigs, exps = [sc], [ec]
    for k in range(0, 64, 16):
        s = sum(e2m1(x) * e2m1(y) for x, y in zip(a[k:k + 16], b[k:k + 16]))
        s0, e0 = _sig_exp(sa[k // block], smin)
        s1, e1 = _sig_exp(sb[k // block], smin)
        sigs.append(s * s0 * s1)
        exps.append(e0 + e1)
    s, e = _fused_sum(sigs, exps, nfrac)
    s, e = _sig_exp(s * 2.0 ** e, -126)
    s = math.trunc(s * 2.0 ** 23) * 2.0 ** -23
    return to_f32_bits(s * 2.0 ** e)


def exact_value(v, kind):
    """Exact dot product (without C) as a Fraction, or None when a scale is NaN."""
    nsf = 2 if kind == KIND_MX32 else 4
    dec = ue4m3 if kind == KIND_NVFP4 else ue8m0
    sa, sb = [dec(x) for x in v["sa"][:nsf]], [dec(x) for x in v["sb"][:nsf]]
    if None in sa or None in sb:
        return None
    block = 64 // nsf
    return sum(Fraction(e2m1(x)) * Fraction(e2m1(y)) * Fraction(sa[k // block]) * Fraction(sb[k // block])
               for k, (x, y) in enumerate(zip(v["a"], v["b"])))


def category(v, kind):
    prods = [E2M1[x & 7] * E2M1[y & 7] for x, y in zip(v["a"], v["b"])]
    zero_blocks = [all(p == 0 for p in prods[k:k + 16]) for k in range(0, 64, 16)]
    if all(zero_blocks):
        return "all products zero"
    if any(zero_blocks):
        return "some 16-block all zero"
    if is_nan_bits(v["c"]) or is_inf_bits(v["c"]):
        return "C is inf/NaN"
    dot = exact_value(v, kind)
    if dot is None:
        return "NaN scale"
    c = Fraction(b2f(v["c"])) if not is_inf_bits(v["c"]) else None
    if c is not None and c != 0 and abs(c) < abs(dot) * Fraction(1, 2 ** 24):
        return "|C| < 2^-24 |dot|"
    return "other"


# --------------------------------------------------------------------------------------------
# Stimulus
# --------------------------------------------------------------------------------------------

def rand_elems(rng, n=64):
    r = rng.random()
    if r < 0.6:
        return [rng.randrange(16) for _ in range(n)]
    if r < 0.75:
        return [rng.choice([0x7, 0xF]) for _ in range(n)]
    if r < 0.85:
        return [rng.choice([0x0, 0x8, 0x1, 0x9]) for _ in range(n)]
    return [rng.randrange(16)] * n


def rand_scale(rng, kind):
    if kind != KIND_NVFP4:
        return rng.randrange(0xFF) if rng.random() > 0.01 else 0xFF
    r = rng.random()
    if r < 0.01:
        return 0x7F
    if r < 0.06:
        return 0x00
    if r < 0.13:
        return rng.randrange(1, 8)
    if r < 0.21:
        return rng.choice([0x38, 0x78, 0x7E])
    return rng.randrange(0x7F)


def rand_c(rng):
    r = rng.random()
    if r < 0.25:
        return 0
    if r < 0.30:
        return 0x80000000
    if r < 0.70:
        return f2b(rng.uniform(-32.0, 32.0))
    if r < 0.85:
        return f2b(math.ldexp(rng.uniform(-1, 1), rng.randrange(-30, 30)))
    if r < 0.95:
        return (rng.getrandbits(1) << 31) | (rng.randrange(1, 255) << 23) | rng.getrandbits(23)
    return rng.choice([0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001, 0x007FFFFF])


def directed(kind):
    one = 0x38 if kind == KIND_NVFP4 else 127
    cases = []
    for c in (0, 0x80000000, 0x3a82fe33, 0xa26f0096, 0x00000001, f2b(1e-30)):  # zero products: C kept?
        cases.append(dict(a=[0] * 64, b=[0x7] * 64, sa=[0x55] * 4, sb=[0x61] * 4, c=c))
        cases.append(dict(a=[0x8] * 64, b=[0x7] * 64, sa=[one] * 4, sb=[one] * 4, c=c))
    for c in (f2b(-4.5e-12), f2b(1e-20), f2b(-1e-9)):  # nonzero products, tiny C
        cases.append(dict(a=[0x2] * 64, b=[0x3] * 64, sa=[one] * 4, sb=[one] * 4, c=c))
    a = [0x2] * 64
    for i in range(16, 32):
        a[i] = 0
    cases.append(dict(a=a, b=[0x2] * 64, sa=[one, 0x7E if kind == KIND_NVFP4 else 200, one, one],
                      sb=[one] * 4, c=f2b(-3e-10)))  # partly zero, large zero-block scale
    cases.append(dict(a=[0x7] * 64, b=[0x7] * 64, sa=[0x7E if kind == KIND_NVFP4 else 140] * 4,
                      sb=[0x7E if kind == KIND_NVFP4 else 140] * 4, c=0))
    cases.append(dict(a=[0x7, 0xF] * 32, b=[0x7] * 64, sa=[one] * 4, sb=[one] * 4, c=f2b(1.0)))
    return cases


def random_vectors(n, seed, kind):
    rng = random.Random(seed)
    out = directed(kind)
    for _ in range(n):
        out.append(dict(a=rand_elems(rng), b=rand_elems(rng), sa=[rand_scale(rng, kind) for _ in range(4)],
                        sb=[rand_scale(rng, kind) for _ in range(4)], c=rand_c(rng)))
    return out


def pack(v):
    nib = lambda x: sum((e & 0xF) << (4 * i) for i, e in enumerate(x)).to_bytes(32, "little")
    return nib(v["a"]) + nib(v["b"]) + bytes(v["sa"]) + bytes(v["sb"]) + struct.pack("<I", v["c"])


def unpack(rec):
    nib = lambda bs: [(bs[i // 2] >> (4 * (i % 2))) & 0xF for i in range(64)]
    return dict(a=nib(rec[0:32]), b=nib(rec[32:64]), sa=list(rec[64:68]), sb=list(rec[68:72]),
                c=struct.unpack("<I", rec[72:76])[0])


def read_vec_file(path):
    """Exported set: one JSON header line, count*76-byte records, then one uint32 column per reference."""
    raw = Path(path).read_bytes()
    nl = raw.index(b"\n")
    hdr = json.loads(raw[:nl])
    n, body = hdr["count"], raw[nl + 1:]
    vecs = [unpack(body[76 * i:76 * (i + 1)]) for i in range(n)]
    cols, off = {}, 76 * n
    for name in hdr["columns"]:
        cols[name] = list(struct.unpack(f"<{n}I", body[off:off + 4 * n]))
        off += 4 * n
    return hdr, vecs, cols


# --------------------------------------------------------------------------------------------
# tcgen05 kernel
# --------------------------------------------------------------------------------------------

CUDA_SRC = r'''
// One CTA (4 warps) per resident slot; each iteration runs one test vector as a single
// tcgen05.mma.cta_group::1 (M=128, N=32, K=64) with the accumulator preloaded with C.
// Every row of A and every column of B carries the same 64 elements and scales, so all 128x32
// outputs equal the dot product under test; the kernel flags the vector if any output differs.
//
// SMEM A/B: K-major, SWIZZLE_NONE canonical layout ((8,m),2):((16B,SBO),LBO). Rows are 16 B
// (32 E2M1, element k at nibble k), the two K halves live in separate regions.
// TMEM: D at columns [0,32), SFA at 64..67, SFB at 96..99 (4 cells per lane, all 128 lanes).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define M 128
#define N 32
#define TMEM_COLS 128
#define SFA_COL 64
#define SFB_COL 96

#if !defined(KIND)
#define KIND 0
#endif
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
#define Q16 "block16"
#define Q32 "block32"
#else
#define Q16 "scale_vec::4X"
#define Q32 "scale_vec::2X"
#endif
#if KIND == 0
#define MMA_OP "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale." Q16
#define SCALE_FMT 0u
#elif KIND == 1
#define MMA_OP "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale." Q16
#define SCALE_FMT 1u
#else
#define MMA_OP "tcgen05.mma.cta_group::1.kind::mxf4.block_scale." Q32
#define SCALE_FMT 1u
#endif

struct Rec { uint8_t a[32], b[32], sfa[4], sfb[4]; uint32_t c; };
static_assert(sizeof(Rec) == 76, "record layout");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  uint64_t d = 0;
  d |= uint64_t((addr >> 4) & 0x3FFF);
  d |= uint64_t((lbo >> 4) & 0x3FFF) << 16;
  d |= uint64_t((sbo >> 4) & 0x3FFF) << 32;
  d |= uint64_t(1) << 46;  // version 1 (Blackwell tcgen05); base_offset 0, lbo_mode 0, layout SWIZZLE_NONE
  return d;
}
__device__ __forceinline__ void tm_st(uint32_t taddr, uint32_t v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" :: "r"(taddr), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t tm_ld(uint32_t taddr) {
  uint32_t v;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(v) : "r"(taddr) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }

// variant bit 0: swap LBO/SBO.  bit 1: scale cells hold one scale per column (byte-replicated)
// instead of four scales per cell.
__global__ void __launch_bounds__(128, 1) run(const Rec* recs, uint32_t* out, int n, int variant) {
  __shared__ alignas(128) uint8_t sA[M * 32];
  __shared__ alignas(128) uint8_t sB[N * 32];
  __shared__ alignas(8) uint64_t mbar;
  __shared__ uint32_t tmem_base;
  __shared__ uint32_t ref_val;
  const int tid = threadIdx.x, warp = tid / 32;

  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_u32(&tmem_base)), "r"(TMEM_COLS) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(&mbar)), "r"(1) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tbase = tmem_base;
  const uint32_t lane_base = uint32_t(warp * 32) << 16;

  const uint32_t a_k = M * 16, b_k = N * 16, mn = 8 * 16;  // K-half region stride, 8-row group stride
  const uint32_t a_lbo = (variant & 1) ? mn : a_k, a_sbo = (variant & 1) ? a_k : mn;
  const uint32_t b_lbo = (variant & 1) ? mn : b_k, b_sbo = (variant & 1) ? b_k : mn;
  const uint32_t idesc = (1u << 7) | (1u << 10) | (uint32_t(N >> 3) << 17) | (SCALE_FMT << 23) |
                         (uint32_t(M >> 4) << 24);

  uint32_t phase = 0;
  for (int v = blockIdx.x; v < n; v += gridDim.x) {
    const Rec& r = recs[v];
    for (int i = tid; i < M * 32 / 4; i += 128) {          // A: 2 regions of M rows x 16 B
      int off = 4 * i, half = off / (M * 16), in_row = off % 16;
      reinterpret_cast<uint32_t*>(sA)[i] = ld32(r.a + 16 * half + in_row);
    }
    for (int i = tid; i < N * 32 / 4; i += 128) {
      int off = 4 * i, half = off / (N * 16), in_row = off % 16;
      reinterpret_cast<uint32_t*>(sB)[i] = ld32(r.b + 16 * half + in_row);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    for (int j = 0; j < N; ++j) tm_st(tbase + lane_base + j, r.c);
    for (int q = 0; q < 4; ++q) {
      uint32_t wa, wb;
      if (variant & 2) {
        int s = (KIND == 2) ? (q & 1) : q;
        wa = 0x01010101u * r.sfa[s];
        wb = 0x01010101u * r.sfb[s];
      } else if (KIND == 2) {
        wa = r.sfa[0] | (r.sfa[1] << 8) | (r.sfa[0] << 16) | (r.sfa[1] << 24);
        wb = r.sfb[0] | (r.sfb[1] << 8) | (r.sfb[0] << 16) | (r.sfb[1] << 24);
      } else {
        wa = ld32(r.sfa);
        wb = ld32(r.sfb);
      }
      tm_st(tbase + lane_base + SFA_COL + q, wa);
      tm_st(tbase + lane_base + SFB_COL + q, wb);
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    if (tid == 0) {
      uint64_t da = smem_desc(smem_u32(sA), a_lbo, a_sbo);
      uint64_t db = smem_desc(smem_u32(sB), b_lbo, b_sbo);
      asm volatile(
          "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
          MMA_OP " [%0], %1, %2, %3, [%5], [%6], p;\n\t}\n"
          :: "r"(tbase), "l"(da), "l"(db), "r"(idesc), "r"(1u), "r"(tbase + SFA_COL), "r"(tbase + SFB_COL)
          : "memory");
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                   :: "r"(smem_u32(&mbar)) : "memory");
    }
    uint32_t done = 0;
    while (!done) {
      asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
                   "selp.u32 %0, 1, 0, p;\n\t}\n" : "=r"(done) : "r"(smem_u32(&mbar)), "r"(phase) : "memory");
    }
    phase ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    uint32_t mine[N];
    for (int j = 0; j < N; ++j) mine[j] = tm_ld(tbase + lane_base + j);
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    if (tid == 0) ref_val = mine[0];
    __syncthreads();
    bool same = true;
    for (int j = 0; j < N; ++j) same &= (mine[j] == ref_val);
    same = __syncthreads_and(same);
    if (tid == 0) out[v] = same ? ref_val : 0x7FBADBADu;
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (warp == 0)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tbase), "r"(TMEM_COLS) : "memory");
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

int main(int argc, char** argv) {
  if (argc != 4) { fprintf(stderr, "usage: %s <variant> <in.bin> <out.bin>\n", argv[0]); return 2; }
  int variant = atoi(argv[1]);
  FILE* f = fopen(argv[2], "rb");
  if (!f) { perror(argv[2]); return 1; }
  std::vector<Rec> recs; Rec r;
  while (fread(&r, sizeof r, 1, f) == 1) recs.push_back(r);
  fclose(f);
  int n = (int)recs.size();
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  Rec* d_recs; uint32_t* d_out;
  CK(cudaMalloc(&d_recs, sizeof(Rec) * (n ? n : 1)));
  CK(cudaMalloc(&d_out, 4 * (n ? n : 1)));
  CK(cudaMemcpy(d_recs, recs.data(), sizeof(Rec) * n, cudaMemcpyHostToDevice));
  int grid = p.multiProcessorCount * 2;
  if (grid > n) grid = n > 0 ? n : 1;
  run<<<grid, 128>>>(d_recs, d_out, n, variant);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<uint32_t> out(n);
  CK(cudaMemcpy(out.data(), d_out, 4 * n, cudaMemcpyDeviceToHost));
  f = fopen(argv[3], "wb"); fwrite(out.data(), 4, n, f); fclose(f);
  fprintf(stderr, "%s sm_%d%d: kind %d variant %d, %d vectors\n", p.name, p.major, p.minor, KIND, variant, n);
  return 0;
}
'''


# compute capability -> (tcgen05 target, minimum CUDA, first CUDA that rejects it)
TARGETS = {
    (10, 0): ("sm_100a", (12, 8), None),   # B200, GB200
    (10, 3): ("sm_103a", (12, 9), None),   # B300, GB300
    (11, 0): ("sm_110a", (13, 0), None),   # Jetson AGX Thor
    (10, 1): ("sm_101a", (12, 8), (13, 0)),  # Thor before the CUDA 13 rename
}


def detect_gpu():
    """(compute capability, name) of device 0 via the CUDA driver API, falling back to nvidia-smi."""
    for lib in ("libcuda.so.1", "libcuda.so"):
        try:
            cu = ctypes.CDLL(lib)
        except OSError:
            continue
        dev, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        if (cu.cuInit(0) == 0 and cu.cuDeviceGet(ctypes.byref(dev), 0) == 0
                and cu.cuDeviceGetAttribute(ctypes.byref(major), 75, dev) == 0
                and cu.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev) == 0):
            cu.cuDeviceGetName(name, 256, dev)
            return (major.value, minor.value), name.value.decode(errors="replace")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip().splitlines()
    except FileNotFoundError:
        out = []
    if out:
        cap, name = [x.strip() for x in out[0].split(",", 1)]
        return tuple(int(x) for x in cap.split(".")), name
    return None, None


def nvcc_candidates(explicit=None):
    cands = [explicit] if explicit else []
    cands += [shutil.which("nvcc"), os.path.join(os.environ.get("CUDA_HOME", ""), "bin/nvcc")]
    cands += sorted(glob.glob("/usr/local/cuda*/bin/nvcc"), reverse=True)
    seen, found = set(), []
    for c in cands:
        if not c or not os.path.exists(c) or os.path.realpath(c) in seen:
            continue
        seen.add(os.path.realpath(c))
        out = subprocess.run([c, "--version"], capture_output=True, text=True).stdout
        m = re.search(r"release (\d+)\.(\d+)", out)
        if m:
            found.append((c, (int(m[1]), int(m[2]))))
    return sorted(found, key=lambda x: x[1], reverse=True)


def select_toolchain(nvcc=None, arch=None):
    """Pick (nvcc path, CUDA version, target, GPU name) for device 0, or exit with the reason."""
    cap, name = detect_gpu()
    if arch is None:
        if cap is None:
            sys.exit("no CUDA GPU found (libcuda and nvidia-smi both failed); use --arch to force a target")
        if cap not in TARGETS:
            sys.exit(f"GPU 0 is {name} (compute {cap[0]}.{cap[1]}), which has no tcgen05 tensor cores. "
                     "Supported: B200/GB200 (10.0), B300/GB300 (10.3), Jetson AGX Thor (11.0). "
                     "Use --arch to override.")
        arch, lo, hi = TARGETS[cap]
    else:
        lo, hi = next(((l, h) for a, l, h in TARGETS.values() if a == arch), ((12, 8), None))
    cands = nvcc_candidates(nvcc)
    ok = [(c, v) for c, v in cands if v >= lo and (hi is None or v < hi)]
    if not ok:
        have = ", ".join(f"{c} ({v[0]}.{v[1]})" for c, v in cands) or "none"
        need = f"CUDA >= {lo[0]}.{lo[1]}" + (f" and < {hi[0]}.{hi[1]}" if hi else "")
        sys.exit(f"target {arch} needs {need}; nvcc found: {have}. Pass --nvcc /path/to/nvcc.")
    path, ver = ok[0]
    return path, ver, arch, name or "unknown GPU"


def build(workdir, nvcc, arch, kinds):
    src = workdir / "nvfp4_tcgen05.cu"
    src.write_text(CUDA_SRC)
    exes = {}
    for k in kinds:
        exe = workdir / f"nvfp4_tcgen05_k{k}"
        cmd = [nvcc, "-O2", "-std=c++17", f"-DKIND={k}", "-gencode",
               f"arch=compute_{arch[3:]},code={arch}", "-o", str(exe), str(src)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode:
            sys.exit(f"nvcc failed for kind {k}:\n{' '.join(cmd)}\n{p.stderr[-4000:]}")
        exes[k] = exe
    return exes


def run_gpu(exe, variant, vecs, workdir, timeout):
    inp, outp = workdir / f"in_{exe.name}_{variant}.bin", workdir / f"out_{exe.name}_{variant}.bin"
    inp.write_bytes(b"".join(pack(v) for v in vecs))
    if outp.exists():
        outp.unlink()
    try:
        p = subprocess.run([str(exe), str(variant), str(inp), str(outp)], capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout (kernel hang)"
    if p.returncode or not outp.exists():
        return None, (p.stderr or p.stdout).strip()[-500:]
    raw = outp.read_bytes()
    return list(struct.unpack(f"<{len(raw) // 4}I", raw)), p.stderr.strip()


# --------------------------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------------------------

def probes(kind):
    """Known-answer vectors whose results do not depend on the semantics under test."""
    one = 0x38 if kind == KIND_NVFP4 else 127
    blk = [0x38, 0x40, 0x48, 0x50] if kind == KIND_NVFP4 else [127, 128, 129, 130]  # 1, 2, 4, 8
    nsf = 2 if kind == KIND_MX32 else 4
    vecs, want = [], []

    def e(k):
        x = [0] * 64
        x[k] = 0x2  # 1.0
        return x
    for k in range(64):
        for k2 in range(64):
            vecs.append(dict(a=e(k), b=e(k2), sa=[one] * 4, sb=[one] * 4, c=0))
            want.append(1.0 if k == k2 else 0.0)
    for k in range(64):
        vecs.append(dict(a=e(k), b=e(k), sa=blk, sb=blk, c=0))
        want.append(4.0 ** (k // (64 // nsf)))
    for k in range(64):  # different A and B scales per block
        vecs.append(dict(a=e(k), b=e(k), sa=blk, sb=[one] * 4, c=0))
        want.append(2.0 ** (k // (64 // nsf)))
    for c in (1.0, -3.5, 2.0):
        vecs.append(dict(a=[0x2] + [0] * 63, b=[0x2] + [0] * 63, sa=[one] * 4, sb=[one] * 4, c=f2b(c)))
        want.append(c + 1.0)
    vals = [0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x9, 0xA, 0xF]
    for x in vals:  # element decoding
        vecs.append(dict(a=[x] + [0] * 63, b=[0x2] + [0] * 63, sa=[one] * 4, sb=[one] * 4, c=0))
        want.append(e2m1(x))
    return vecs, [f2b(w) for w in want]


def calibrate(exe, kind, workdir, timeout, verbose):
    vecs, want = probes(kind)
    passing = []
    for variant in range(4):
        got, msg = run_gpu(exe, variant, vecs, workdir, timeout)
        if got is None:
            print(f"    variant {variant}: failed to run: {msg}")
            continue
        bad = [i for i, (g, w) in enumerate(zip(got, want)) if g != w]
        print(f"    variant {variant}: {len(want) - len(bad)}/{len(want)} probes correct")
        if verbose and bad:
            for i in bad[:5]:
                print(f"      probe {i}: got {got[i]:08x} ({b2f(got[i])}) want {want[i]:08x} ({b2f(want[i])})")
        if not bad:
            passing.append(variant)
    return passing


# --------------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------------

def same(x, y):
    return x == y or (is_nan_bits(x) and is_nan_bits(y))


def compare(name, vecs, kind, gpu, refs, show):
    print(f"\n{name}: {len(vecs)} vectors, {KIND_NAMES[kind]}")
    bad = sum(g == 0x7FBADBAD for g in gpu)
    if bad:
        print(f"  WARNING: {bad} vectors gave inconsistent outputs across the 128x32 tile")
    summary = {"vectors": len(vecs), "kind": kind, "inconsistent": bad, "references": {}}
    for ref_name, ref in refs.items():
        diff = [i for i, (g, r) in enumerate(zip(gpu, ref)) if not same(g, r)]
        cats = Counter(category(vecs[i], kind) for i in diff)
        print(f"  GPU == {ref_name:12s}: {len(vecs) - len(diff):>8}/{len(vecs)}"
              + (f"   disagreements by situation: {dict(cats)}" if diff else ""))
        summary["references"][ref_name] = {
            "agree": len(vecs) - len(diff), "disagree_by_category": dict(cats),
            "examples": [dict(index=i, gpu=f"{gpu[i]:08x}", ref=f"{ref[i]:08x}", vector=pack(vecs[i]).hex())
                         for i in diff[:50]]}
        for i in diff[:show]:
            v = vecs[i]
            print(f"      #{i}: gpu={gpu[i]:08x} {ref_name}={ref[i]:08x}  c={v['c']:08x} "
                  f"sa={bytes(v['sa']).hex()} sb={bytes(v['sb']).hex()} [{category(v, kind)}]")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=100000, help="random vectors per format (default 100000)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--formats", default="nvfp4", help="comma list of nvfp4, mx16, mx32 for random vectors")
    ap.add_argument("--vectors", nargs="*", default=[], help="exported .vec files (skips random vectors)")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--variant", type=int, help="skip calibration and use this layout variant")
    ap.add_argument("--nvcc")
    ap.add_argument("--arch", help="sm_100a, sm_103a, sm_110a (Thor) or sm_101a (default: detected)")
    ap.add_argument("--out", default="tcgen05_results")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--show", type=int, default=8, help="disagreements to print per reference")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    fmt_kind = {"nvfp4": KIND_NVFP4, "mx16": KIND_MX16, "mx32": KIND_MX32}
    jobs = []  # (name, kind, vectors, {reference name: outputs})
    if args.vectors:
        for path in args.vectors:
            hdr, vecs, cols = read_vec_file(path)
            jobs.append((hdr.get("name", Path(path).stem), hdr["kind"], vecs, cols, hdr))
    else:
        for f in args.formats.split(","):
            k = fmt_kind[f]
            jobs.append((f"random_{f}_seed{args.seed}", k, random_vectors(args.n, args.seed, k), {}, {}))

    nvcc, ver, arch, gpu_name = select_toolchain(args.nvcc, args.arch)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="nvfp4_tcgen05_"))
    kinds = sorted({j[1] for j in jobs})
    print(f"{gpu_name}: nvcc {nvcc} (CUDA {ver[0]}.{ver[1]}), target {arch}, build dir {workdir}")
    exes = build(workdir, nvcc, arch, kinds)

    report = {"arch": arch, "cuda": f"{ver[0]}.{ver[1]}", "gpu": gpu_name,
              "time": time.strftime("%Y-%m-%d %H:%M:%S"), "sets": {}}
    variants = {}
    for k in kinds:
        print(f"\nlayout calibration, {KIND_NAMES[k]}:")
        if args.variant is not None:
            variants[k] = args.variant
            print(f"    using --variant {args.variant} without probing")
            continue
        ok = calibrate(exes[k], k, workdir, args.timeout, args.verbose)
        if len(ok) != 1:
            report["calibration_failed"] = {"kind": k, "passing_variants": ok}
            (outdir / "summary.json").write_text(json.dumps(report, indent=1))
            sys.exit(f"calibration for kind {k}: {len(ok)} variants passed all probes (need exactly 1). "
                     f"Rerun with -v and send the output; results were not produced.")
        variants[k] = ok[0]
        print(f"    -> variant {ok[0]}")
        report.setdefault("variants", {})[str(k)] = ok[0]
    if args.probe_only:
        (outdir / "summary.json").write_text(json.dumps(report, indent=1))
        return

    for name, kind, vecs, cols, hdr in jobs:
        t0 = time.time()
        gpu, msg = run_gpu(exes[kind], variants[kind], vecs, workdir, args.timeout)
        if gpu is None:
            sys.exit(f"{name}: GPU run failed: {msg}")
        t_gpu = time.time() - t0
        refs = {"mmasim": [model(v, kind, "mmasim") for v in vecs],
                "rtx5090": [model(v, kind, "rtx5090") for v in vecs]}
        for col, vals in cols.items():
            refs[f"file:{col}"] = vals
        print(f"  ({msg}; {t_gpu:.1f} s)")
        report["sets"][name] = compare(name, vecs, kind, gpu, refs, args.show)
        if hdr:
            report["sets"][name]["file_header"] = hdr
        (outdir / f"{name}.{arch}.out").write_bytes(struct.pack(f"<{len(gpu)}I", *gpu))
    (outdir / "summary.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {outdir}/summary.json and raw outputs ({outdir}/*.{arch}.out)")


if __name__ == "__main__":
    main()
