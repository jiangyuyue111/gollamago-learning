import torch
import triton

from ninetoothed_gemm import BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_N, nt_gemm


M = N = K = 1024

def tflops(latency_ms):
    return 2 * M * N * K / (latency_ms * 1e-3) / 1e12


lhs = torch.randn((M, K), device="cuda", dtype=torch.float16)
rhs = torch.randn((K, N), device="cuda", dtype=torch.float16)
output = nt_gemm(lhs, rhs)
reference = torch.matmul(lhs, rhs)
assert torch.allclose(output, reference, atol=2e-2, rtol=2e-2)

ninetoothed_ms = triton.testing.do_bench(lambda: nt_gemm(lhs, rhs))
torch_ms = triton.testing.do_bench(lambda: torch.matmul(lhs, rhs))
print(f"block={BLOCK_SIZE_M}x{BLOCK_SIZE_N}x{BLOCK_SIZE_K}; correct=PASS")
print(f"NineToothed: {ninetoothed_ms:.4f} ms, {tflops(ninetoothed_ms):.3f} TFLOPS")
print(f"PyTorch:     {torch_ms:.4f} ms, {tflops(torch_ms):.3f} TFLOPS")
print(f"speedup_vs_PyTorch={torch_ms / ninetoothed_ms:.3f}x")
