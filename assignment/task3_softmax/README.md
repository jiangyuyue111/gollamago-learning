# 任务三：TileLang Softmax Puzzle

实现二维矩阵按行 Softmax。输入 `A:[N,M] float32`，输出同 shape/dtype，沿最后一维计算稳定公式 `exp(x-max)/sum(exp(x-max))`。

要求：在 `solution.py` 实现 `tl_softmax(A,BLOCK_N,BLOCK_M)`；完成最大值归约、指数、求和归约和写回；综合 benchmark 使用 `(1,1)、(3,7)、(16,256)、(17,513)、(64,4096)` 五组配置。

运行：

```bash
cd /data/go-llama-go/assignment/task3_softmax
source /data/go-llama-go/setup_env.sh
python -m pytest -q test_softmax.py
python benchmark_softmax.py
```

提交:
运行`python benchmark_softmax.py`后的终端结果截图

## 九齿补充任务：GEMM

在 `ninetoothed_gemm.py` 中完成 `arrangement()` 和 `application()`，实现
`C = A @ B`：输入和输出使用 `float16`，中间结果使用 `float32` 累加。默认 block
size 为 `64 x 64 x 64`。

参考：[NineToothed Matrix Multiplication](https://github.com/InfiniTensor/ninetoothed/blob/b77f930dc6c8b016e09adf33570d55a7bc8376c1/docs/source/basics.rst)。

```bash
python -m pytest -q test_ninetoothed_gemm.py
python benchmark_ninetoothed_gemm.py
```

需要尝试自动调优时运行：

```bash
NINETOOTHED_AUTOTUNE=1 python benchmark_ninetoothed_gemm.py
```

测试使用 `M=N=K=512`；benchmark 对 `M=N=K=2^3` 到 `2^12` 的规模比较九齿与
`torch.mm` 的耗时。提交完成后的 `ninetoothed_gemm.py`、测试输出和基线性能数据。
