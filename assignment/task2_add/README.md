# 任务二：TileLang Vector Add Puzzle

实现 1-D `float16` 向量加法：`C[i]=A[i]+B[i]`。`1<=N<=1M`，必须处理尾块（N 不整除 BLOCK_N）。

要求：在 `solution.py` 实现 `tl_add_1d(A,B,BLOCK_N)`；使用 `T.Kernel`、`T.Parallel`；测试 N=1、127、1024、100003、1048576；与 PyTorch 逐元素校验；比较至少两种 BLOCK_N。

运行：

```bash
cd /data/go-llama-go/assignment/task2_add
source /data/go-llama-go/setup_env.sh
python "$TILELANG_ROOT/examples/quickstart.py"
python -m pytest -q test_add.py
python benchmark_add.py
```

`benchmark_add.py` 会对 N=1、127、1024、100003、1048576 逐一做正确性校验，并同时输出 TileLang 与 PyTorch 的平均耗时和最大误差。(不必要求TileLang比PyTorch更快)

提交:

运行 `python benchmark_add.py` 后的终端结果截图

## 九齿补充任务：Vector Add

这部分独立于上面的 TileLang 作业。在 `ninetoothed_add.py` 中完成 `application()`，实现同样的
一维 `float16` 向量加法并处理尾块；保持 `BLOCK_SIZE = 1024`，不做自动调优。

参考：[NineToothed Vector Addition](https://github.com/InfiniTensor/ninetoothed/blob/b77f930dc6c8b016e09adf33570d55a7bc8376c1/docs/source/basics.rst)。

```bash
python -m pip install ninetoothed
python -m pytest -q test_ninetoothed_add.py
python benchmark_ninetoothed_add.py
```

提交完成后的 `ninetoothed_add.py`、测试与 benchmark 输出，并用 3 到 5 句话记录九齿的
分块方式和开发体验。
