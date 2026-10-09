// CUDA native E4M3/E2M1 conversion oracle, finite nearest-even NVFP4 blocks.
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <stdexcept>

struct Record { uint32_t inverse, x[16]; };
struct Output { uint64_t packed; uint32_t scale; };
__global__ void run(const Record* records, Output* outputs, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  Record r = records[index];
  float inverse = __uint_as_float(r.inverse), amax = 0;
  for (auto word : r.x) amax = fmaxf(amax, fabsf(__uint_as_float(word)));
  __nv_fp8_e4m3 scale(__fmul_rn(__fmul_rn(amax, inverse), 1.0f / 6.0f));
  float decoded = static_cast<float>(scale);
  Output result = {0, scale.__x};
  for (int lane = 0; lane < 16; lane++) {
    float x = __fmul_rn(__uint_as_float(r.x[lane]), inverse);
    __nv_fp4_e2m1 element(decoded == 0 ? copysignf(0.0f, x) : __fdiv_rn(x, decoded));
    result.packed |= uint64_t(element.__x & 15) << (4 * lane);
  }
  outputs[index] = result;
}
int main() {
  std::vector<Record> records;
  Record r;
  while (std::cin >> std::hex >> r.inverse) {
    for (auto& x : r.x) std::cin >> x;
    records.push_back(r);
  }
  Record* device_records;
  Output* device_outputs;
  std::vector<Output> outputs(records.size());
  cudaMalloc(&device_records, records.size() * sizeof(Record));
  cudaMalloc(&device_outputs, outputs.size() * sizeof(Output));
  cudaMemcpy(device_records, records.data(), records.size() * sizeof(Record), cudaMemcpyHostToDevice);
  run<<<(records.size() + 127) / 128, 128>>>(device_records, device_outputs, records.size());
  if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("CUDA quantization failed");
  cudaMemcpy(outputs.data(), device_outputs, outputs.size() * sizeof(Output), cudaMemcpyDeviceToHost);
  for (auto result : outputs) std::cout << std::hex << result.packed << ' ' << result.scale << '\n';
  cudaFree(device_records);
  cudaFree(device_outputs);
}
