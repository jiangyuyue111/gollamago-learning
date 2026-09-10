import tilelang
import tilelang.language as T

@tilelang.jit
def tl_softmax(A, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    A: T.Tensor((N, M), T.float32)
    B = T.empty((N, M), T.float32)
    # Step 1: ceildiv 覆盖任意 N；Step 2: 分配 tile fragment。
    # Step 3: ceildiv 扫描任意 M；Step 4: 越界填充 -inf。
    # Step 5: reduce_max；Step 6: exp2 稳定指数；Step 7: reduce_sum。
    # Step 8: online 更新 running max/sum（或 lse）。
    # Step 9: 第二遍扫描并归一化；Step 10: 只写回有效行列。
    # TODO: 完成上述 online Softmax 步骤。
    raise NotImplementedError("请根据步骤实现 Softmax")
