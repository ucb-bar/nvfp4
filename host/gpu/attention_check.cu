// FP32 forward attention oracle with explicitly ordered operations and native RTX SFUs.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <stdexcept>
struct Row { uint32_t mask, scale, scores[24], values[24][8]; };
struct Output { uint32_t values[8]; };
__device__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;":"=f"(y):"f"(x)); return y; }
__device__ float rcp(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;":"=f"(y):"f"(x)); return y; }
__global__ void attention(const Row* rows, Output* results, int count) {
  int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index>=count) return;
  Row r=rows[index];
  float maximum=-CUDART_INF_F, sum=0, numerator[8]={};
  float scale=__uint_as_float(r.scale);
  for (int block=0;block<3;block++) {
    float next_max=maximum, probability[8];
    for (int k=0;k<8;k++) if ((r.mask>>(block*8+k))&1)
      next_max=fmaxf(next_max,__uint_as_float(r.scores[block*8+k]));
    float alpha=sum==0 ? 0 : ex2(__fmul_rn(__fadd_rn(maximum,-next_max),scale));
    float tile_sum=0;
    for (int k=0;k<8;k++) {
      probability[k]=((r.mask>>(block*8+k))&1) ?
        ex2(__fmul_rn(__fadd_rn(__uint_as_float(r.scores[block*8+k]),-next_max),scale)) : 0;
      tile_sum=__fadd_rn(tile_sum,probability[k]);
    }
    sum=__fadd_rn(__fmul_rn(alpha,sum),tile_sum);
    for (int column=0;column<8;column++) {
      float tile_pv=0;
      for (int k=0;k<8;k++) tile_pv=__fadd_rn(tile_pv,
        __fmul_rn(probability[k],__uint_as_float(r.values[block*8+k][column])));
      numerator[column]=__fadd_rn(__fmul_rn(alpha,numerator[column]),tile_pv);
    }
    maximum=next_max;
  }
  float inverse=sum==0 ? 0 : rcp(sum);
  for (int column=0;column<8;column++) results[index].values[column]=__float_as_uint(__fmul_rn(numerator[column],inverse));
}
int main() {
  std::vector<Row> rows; Row r;
  while (std::cin>>std::hex>>r.mask>>r.scale) {
    for (auto& x:r.scores) std::cin>>x;
    for (auto& row:r.values) for (auto& x:row) std::cin>>x;
    rows.push_back(r);
  }
  Row* input; Output* output;
  cudaMalloc(&input,rows.size()*sizeof(Row)); cudaMalloc(&output,rows.size()*sizeof(Output));
  cudaMemcpy(input,rows.data(),rows.size()*sizeof(Row),cudaMemcpyHostToDevice);
  attention<<<(rows.size()+127)/128,128>>>(input,output,rows.size());
  if (cudaDeviceSynchronize()!=cudaSuccess) throw std::runtime_error("CUDA attention failed");
  std::vector<Output> results(rows.size());
  cudaMemcpy(results.data(),output,results.size()*sizeof(Output),cudaMemcpyDeviceToHost);
  for (const auto& row:results) { for (auto x:row.values) std::cout<<std::hex<<x<<' '; std::cout<<'\n'; }
  cudaFree(input); cudaFree(output);
}
