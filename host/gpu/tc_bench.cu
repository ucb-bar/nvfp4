// Latency and throughput of mma.sync.m16n8k64 NVFP4 (block scale, UE4M3) on one SM of an sm_120a GPU.
//
// One block of W warps runs on one SM. Each warp issues ILP independent accumulator chains.
// W = 1, ILP = 1 is a dependent chain: cycles per MMA = latency. Warps w and w + 4 share an SM
// sub-partition (one tensor core), so W = 4 puts one warp on each of the four tensor cores.
//
//   tc_bench            prints one line per (W, ILP)
#include <cstdint>
#include <cstdio>

constexpr int MACS = 16 * 8 * 64;  // multiply-accumulates per m16n8k64

__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

template <int ILP>
__global__ void bench(int iters, float* sink, long long* cycles) {
    uint32_t e = 0x22222222u ^ (threadIdx.x & 1);  // E2M1 1.0 (and 1.0/0.5 mixes)
    uint32_t s = 0x38383838u;                       // UE4M3 1.0
    float d[ILP][4] = {};
    __syncthreads();
    long long t0 = clock64();
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < ILP; ++j) mma(d[j], e, e, e, e, e, e, s, s);
    }
    long long t1 = clock64();
    float acc = 0;
#pragma unroll
    for (int j = 0; j < ILP; ++j) acc += d[j][0] + d[j][1] + d[j][2] + d[j][3];
    sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
    if ((threadIdx.x & 31) == 0) cycles[threadIdx.x / 32] = t1 - t0;
}

template <int ILP>
void run(int warps, float* sink, long long* cyc_d) {
    const int iters = 4096;
    bench<ILP><<<1, 32 * warps>>>(iters, sink, cyc_d);  // warm-up
    bench<ILP><<<1, 32 * warps>>>(iters, sink, cyc_d);
    cudaDeviceSynchronize();
    long long cyc[32];
    cudaMemcpy(cyc, cyc_d, sizeof(long long) * warps, cudaMemcpyDeviceToHost);
    long long worst = 0;
    for (int w = 0; w < warps; ++w) worst = cyc[w] > worst ? cyc[w] : worst;
    double mmas = double(iters) * ILP * warps;
    printf("warps %2d  ilp %d  cycles/mma/warp %7.2f  MAC/clk/SM %7.1f\n", warps, ILP,
           double(worst) / (double(iters) * ILP), mmas * MACS / double(worst));
}

int main() {
    float* sink;
    long long* cyc;
    cudaMalloc(&sink, 32 * 32 * sizeof(float));
    cudaMalloc(&cyc, 32 * sizeof(long long));
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s sm_%d%d, %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
    for (int w : {1, 2, 4, 8, 16, 32}) {
        run<1>(w, sink, cyc);
        run<2>(w, sink, cyc);
        run<4>(w, sink, cyc);
        run<8>(w, sink, cyc);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
    return 0;
}
