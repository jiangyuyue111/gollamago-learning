#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

void launch_silu_mul_bf16(const void* gate, const void* up, void* output,
                          int64_t elements, int device_index);
void launch_rms_norm_bf16(const void* input, const void* weight, void* output,
                          int64_t rows, int64_t columns, float eps,
                          int device_index);

namespace {

void check_maca_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on a CUDA-compatible MACA device");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor silu_mul(torch::Tensor gate, torch::Tensor up) {
  check_maca_tensor(gate, "gate");
  check_maca_tensor(up, "up");
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have identical shapes");
  TORCH_CHECK(gate.device() == up.device(), "gate and up must be on the same device");

  c10::cuda::CUDAGuard device_guard(gate.device());
  auto output = torch::empty_like(gate);
  launch_silu_mul_bf16(gate.data_ptr(), up.data_ptr(), output.data_ptr(),
                       gate.numel(), gate.get_device());
  return output;
}

torch::Tensor rms_norm(torch::Tensor input, torch::Tensor weight, double eps) {
  check_maca_tensor(input, "input");
  check_maca_tensor(weight, "weight");
  TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
  TORCH_CHECK(input.size(-1) == weight.numel(),
              "weight size must match the final input dimension");
  TORCH_CHECK(input.device() == weight.device(),
              "input and weight must be on the same device");
  TORCH_CHECK(input.size(-1) > 0, "the normalized dimension must be non-empty");
  TORCH_CHECK(eps >= 0.0, "eps must be non-negative");

  c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty_like(input);
  const int64_t columns = input.size(-1);
  const int64_t rows = input.numel() / columns;
  launch_rms_norm_bf16(input.data_ptr(), weight.data_ptr(), output.data_ptr(),
                       rows, columns, static_cast<float>(eps), input.get_device());
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("rms_norm", &rms_norm, "MXMACA BF16 RMSNorm");
  module.def("silu_mul", &silu_mul, "MXMACA BF16 fused SiLU multiply");
}
