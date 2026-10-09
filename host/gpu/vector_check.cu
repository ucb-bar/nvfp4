// GPU oracle for the vector unit: one record per line, op = vector_pkg opcode, or 15 for the
// FP32 primitives (add, mul, ex2, rcp and max of x[0] and y[0]). FTZ PTX operations throughout.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <stdexcept>

struct Record { uint32_t op, mask, scalar, old_max, old_sum, x[8], y[8]; };
struct Output { uint32_t result[8], reduction, maximum, sum, alpha; };
__device__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ float rcp(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ float ptx_rsqrt(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ float maximum(float x, float y) {
  float z; asm("max.ftz.f32 %0, %1, %2;" : "=f"(z) : "f"(x), "f"(y)); return z;
}
__global__ void run(const Record* records, Output* outputs, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  Record r = records[index]; Output o = {};
  float scalar = __uint_as_float(r.scalar), m = __uint_as_float(r.old_max), l = __uint_as_float(r.old_sum);
  if (l == 0.0f) m = -CUDART_INF_F;
  float a = 1.0f, running = 0.0f, reduce = r.op == 5 ? -CUDART_INF_F : 0.0f;
  if (r.op == 15) {
    float x = __uint_as_float(r.x[0]), y = __uint_as_float(r.y[0]);
    o.result[0] = __float_as_uint(__fadd_rn(x,y)); o.result[1] = __float_as_uint(__fmul_rn(x,y));
    o.result[2] = __float_as_uint(ex2(x)); o.result[3] = __float_as_uint(rcp(x));
    o.result[4] = __float_as_uint(maximum(x,y));
    outputs[index] = o; return;
  }
  if (r.op == 9) {
    for (int i = 0; i < 8; i++) if ((r.mask >> i) & 1) m = maximum(m, __uint_as_float(r.x[i]));
    a = l == 0.0f ? 0.0f : ex2(__fmul_rn(__fadd_rn(__uint_as_float(r.old_max), -m), scalar));
  }
  if (r.op == 10) scalar = scalar == 0.0f ? 0.0f : rcp(scalar);
  for (int i = 0; i < 8; i++) {
    float x = __uint_as_float(r.x[i]), y = __uint_as_float(r.y[i]), z = 0.0f;
    if ((r.mask >> i) & 1) switch (r.op) {
      case 0: z = __fadd_rn(x,y); break;
      case 1: z = __fadd_rn(x,-y); break;
      case 2: z = __fmul_rn(x,y); break;
      case 3: case 10: z = __fmul_rn(x,scalar); break;
      case 4: z = __fadd_rn(__fmul_rn(x,scalar),y); break;
      case 5: z = x; reduce = maximum(reduce,x); break;
      case 6: z = x; reduce = __fadd_rn(reduce,x); break;
      case 7: z = ex2(x); break;
      case 8: z = rcp(x); break;
      case 9: z = m == -CUDART_INF_F ? 0.0f : ex2(__fmul_rn(__fadd_rn(x,-m), scalar)); break;
      case 11: z = ptx_rsqrt(x); break;
      case 12: z = __fmul_rn(x, rcp(__fadd_rn(1.0f, ex2(__fmul_rn(x, -1.4426950408889634f))))); break;
      case 13: z = fabsf(x); break;
    }
    if (r.op == 9) running = __fadd_rn(running,z);
    o.result[i] = __float_as_uint(z);
  }
  o.reduction = __float_as_uint(reduce);
  o.maximum = __float_as_uint(m);
  o.sum = __float_as_uint(r.op == 9 ? __fadd_rn(__fmul_rn(a,l),running) : 0.0f);
  o.alpha = __float_as_uint(a);
  outputs[index] = o;
}
int main() {
  std::vector<Record> records;
  Record r;
  while (std::cin >> std::hex >> r.op >> r.mask >> r.scalar >> r.old_max >> r.old_sum) {
    for (auto& x : r.x) std::cin >> x;
    for (auto& y : r.y) std::cin >> y;
    records.push_back(r);
  }
  Record* device_records; Output* device_outputs;
  std::vector<Output> outputs(records.size());
  cudaMalloc(&device_records, records.size()*sizeof(Record));
  cudaMalloc(&device_outputs, records.size()*sizeof(Output));
  cudaMemcpy(device_records, records.data(), records.size()*sizeof(Record), cudaMemcpyHostToDevice);
  run<<<(records.size()+127)/128,128>>>(device_records,device_outputs,records.size());
  if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("CUDA oracle failed");
  cudaMemcpy(outputs.data(),device_outputs,outputs.size()*sizeof(Output),cudaMemcpyDeviceToHost);
  for (const auto& o : outputs) {
    for (auto x : o.result) std::cout << std::hex << x << ' ';
    std::cout << o.reduction << ' ' << o.maximum << ' ' << o.sum << ' ' << o.alpha << '\n';
  }
  cudaFree(device_records); cudaFree(device_outputs);
}
