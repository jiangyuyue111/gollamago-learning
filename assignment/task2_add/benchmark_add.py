import time, torch
from solution import tl_add_1d
SIZES = [1, 127, 1024, 100003, 1048576]
def timed(fn, warmup=10, repeat=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); start = time.perf_counter()
    for _ in range(repeat): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-start)*1e6/repeat
print("N\tcorrect\tTileLang(us)\tPyTorch(us)\tmax_abs_error")
for n in SIZES:
    a=torch.randn(n,device='cuda',dtype=torch.float16); b=torch.randn_like(a)
    k=tl_add_1d.compile(N=n,BLOCK_N=1024); out=k(a,b); ref=a+b
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print(f"{n}\tPASS\t{timed(lambda:k(a,b)):.2f}\t\t{timed(lambda:a+b):.2f}\t\t{(out-ref).abs().max().item():.3e}")
