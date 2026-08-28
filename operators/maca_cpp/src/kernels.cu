#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kElementwiseThreads = 256;
constexpr int kNormThreads = 128;
constexpr int kReductionWidth = 512;

__global__ void silu_mul_bf16_kernel(const __nv_bfloat16* gate,
                                     const __nv_bfloat16* up,
                                     __nv_bfloat16* output,
                                     int64_t elements) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float gate_value = __bfloat162float(gate[index]);
    const float up_value = __bfloat162float(up[index]);
    const __nv_bfloat16 silu =
        __float2bfloat16_rn(gate_value / (1.0f + __expf(-gate_value)));
    output[index] = __float2bfloat16_rn(__bfloat162float(silu) * up_value);
  }
}

__global__ void rms_norm_bf16_kernel(const __nv_bfloat16* input,
                                     const __nv_bfloat16* weight,
                                     __nv_bfloat16* output,
                                     int64_t columns,
                                     float eps) {
  __shared__ float reduction[kReductionWidth];
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * columns;

  for (int offset = threadIdx.x; offset < kReductionWidth; offset += blockDim.x) {
    float square_sum = 0.0f;
    for (int64_t chunk = 0; chunk * kReductionWidth < columns; ++chunk) {
      const int64_t column = chunk * kReductionWidth + offset;
      if (column < columns) {
        const float value = __bfloat162float(input[row_offset + column]);
        const __nv_bfloat16 square = __float2bfloat16_rn(value * value);
        square_sum += __bfloat162float(square);
      }
    }
    reduction[offset] = square_sum;
  }
  __syncthreads();

  for (int stride = kReductionWidth / 2; stride > 0; stride /= 2) {
    for (int offset = threadIdx.x; offset < stride; offset += blockDim.x) {
      reduction[offset] += reduction[offset + stride];
    }
    __syncthreads();
  }

  const __nv_bfloat16 mean =
      __float2bfloat16_rn(reduction[0] / static_cast<float>(columns));
  const __nv_bfloat16 inverse_rms =
      __float2bfloat16_rn(rsqrtf(__bfloat162float(mean) + eps));
  for (int offset = threadIdx.x; offset < kReductionWidth; offset += blockDim.x) {
    for (int64_t chunk = 0; chunk * kReductionWidth < columns; ++chunk) {
      const int64_t column = chunk * kReductionWidth + offset;
      if (column < columns) {
        const float value = __bfloat162float(input[row_offset + column]);
        const float scale = __bfloat162float(weight[column]);
        const __nv_bfloat16 normalized =
            __float2bfloat16_rn(value * __bfloat162float(inverse_rms));
        output[row_offset + column] =
            __float2bfloat16_rn(__bfloat162float(normalized) * scale);
      }
    }
  }
}

}  // namespace

void launch_silu_mul_bf16(const void* gate, const void* up, void* output,
                          int64_t elements, int device_index) {
  if (elements == 0) {
    return;
  }
  const int blocks =
      static_cast<int>((elements + kElementwiseThreads - 1) / kElementwiseThreads);
  auto stream = at::cuda::getCurrentCUDAStream(device_index);
  silu_mul_bf16_kernel<<<blocks, kElementwiseThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up),
      static_cast<__nv_bfloat16*>(output), elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_rms_norm_bf16(const void* input, const void* weight, void* output,
                          int64_t rows, int64_t columns, float eps,
                          int device_index) {
  if (rows == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream(device_index);
  rms_norm_bf16_kernel<<<static_cast<unsigned int>(rows), kNormThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), columns, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
