// Runs each test vector through one block-scaled mma.sync.m16n8k64 on an sm_120a tensor core.
//
// Every row of A carries the same 64 E2M1 elements and scales, and so does every column of B,
// so all 128 outputs equal the one dot product under test. Only the K layout of the
// fragments matters (CUTLASS MMA_Traits<SM120_16x8x64_TN_VS>):
//   A: lane%4 = t holds k = 8t..8t+7 in a0/a1 and k+32 in a2/a3 (low nibble first)
//   B: the same k pattern in b0/b1
//   scale register byte g scales k in [16g, 16g+16) (4X) or [32g, 32g+32) (2X)
//
//   nvfp4_mma <kind> <in.bin> <out.bin>
//   kind 0: e2m1 x e2m1, ue4m3 scales, scale_vec::4X (NVFP4)
//   kind 1: e2m1 x e2m1, ue8m0 scales, scale_vec::4X (MXFP4, block 16)
//   kind 2: e2m1 x e2m1, ue8m0 scales, scale_vec::2X (MXFP4, block 32)
//   in.bin: records {a[32], b[32], sfa[4], sfb[4], c_fp32[4]}; element k is nibble k.
//   out.bin: one fp32 bit pattern (d) per record.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct Rec {
    uint8_t  a[32], b[32], sfa[4], sfb[4];
    uint32_t c;
};
static_assert(sizeof(Rec) == 76, "record layout");

__device__ uint32_t ld32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

template <int KIND>
__global__ void run(const Rec* recs, uint32_t* out, int n) {
    int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (w >= n) return;  // n is checked per warp, so whole warps exit together
    const Rec& r = recs[w];
    int t = lane & 3;
    uint32_t a0 = ld32(r.a + 4 * t), a2 = ld32(r.a + 16 + 4 * t);
    uint32_t b0 = ld32(r.b + 4 * t), b1 = ld32(r.b + 16 + 4 * t);
    uint32_t sfa = ld32(r.sfa), sfb = ld32(r.sfb);
    float c = __uint_as_float(r.c);
    float d0, d1, d2, d3;
    const uint16_t id = 0;
    if constexpr (KIND == 0) {
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a0), "r"(a2), "r"(a2), "r"(b0), "r"(b1), "f"(c), "f"(c), "f"(c), "f"(c),
              "r"(sfa), "h"(id), "h"(id), "r"(sfb), "h"(id), "h"(id));
    } else if constexpr (KIND == 1) {
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a0), "r"(a2), "r"(a2), "r"(b0), "r"(b1), "f"(c), "f"(c), "f"(c), "f"(c),
              "r"(sfa), "h"(id), "h"(id), "r"(sfb), "h"(id), "h"(id));
    } else {
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a0), "r"(a2), "r"(a2), "r"(b0), "r"(b1), "f"(c), "f"(c), "f"(c), "f"(c),
              "r"(sfa & 0xFFFF), "h"(id), "h"(id), "r"(sfb & 0xFFFF), "h"(id), "h"(id));
    }
    // All four outputs of every lane must agree; flag the vector if they do not.
    uint32_t v = __float_as_uint(d0);
    bool same = v == __float_as_uint(d1) && v == __float_as_uint(d2) && v == __float_as_uint(d3);
    same = __all_sync(0xFFFFFFFF, same && v == __shfl_sync(0xFFFFFFFF, v, 0));
    if (lane == 0) out[w] = same ? v : 0x7FBADBAD;
}

#define CK(x)                                                                    \
    do {                                                                         \
        cudaError_t e = (x);                                                     \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            return 1;                                                            \
        }                                                                        \
    } while (0)

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <kind> <in.bin> <out.bin>\n", argv[0]);
        return 2;
    }
    int kind = atoi(argv[1]);
    FILE* f = fopen(argv[2], "rb");
    if (!f) return perror(argv[2]), 1;
    std::vector<Rec> recs;
    Rec r;
    while (fread(&r, sizeof r, 1, f) == 1) recs.push_back(r);
    fclose(f);
    int n = recs.size();

    Rec* d_recs;
    uint32_t* d_out;
    CK(cudaMalloc(&d_recs, sizeof(Rec) * n));
    CK(cudaMalloc(&d_out, 4 * n));
    CK(cudaMemcpy(d_recs, recs.data(), sizeof(Rec) * n, cudaMemcpyHostToDevice));
    int threads = 256, blocks = (n * 32 + threads - 1) / threads;
    if (kind == 0) run<0><<<blocks, threads>>>(d_recs, d_out, n);
    else if (kind == 1) run<1><<<blocks, threads>>>(d_recs, d_out, n);
    else run<2><<<blocks, threads>>>(d_recs, d_out, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> out(n);
    CK(cudaMemcpy(out.data(), d_out, 4 * n, cudaMemcpyDeviceToHost));

    f = fopen(argv[3], "wb");
    fwrite(out.data(), 4, n, f);
    fclose(f);
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    fprintf(stderr, "%s sm_%d%d: kind %d, %d vectors\n", p.name, p.major, p.minor, kind, n);
    return 0;
}
