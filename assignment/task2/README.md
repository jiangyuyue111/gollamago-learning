# 任务二：TileLang Add 与 NineToothed Vector Add

## TileLang 任务：Vector Add

实现 1-D `float16` 向量加法：`C[i]=A[i]+B[i]`。`1<=N<=1M`，必须处理尾块（N 不整除 BLOCK_N）。

要求：在 `solution.py` 实现 `tl_add_1d(A,B,BLOCK_N)`；使用 `T.Kernel`、`T.Parallel`；测试 N=1、127、1024、100003、1048576；与 PyTorch 逐元素校验；比较至少两种 BLOCK_N。

运行：

```bash
cd /data/gollamago
source ./setup_env.sh
cd assignment/task2
python -m pytest -q test_add.py
python benchmark_add.py
```

`benchmark_add.py` 会对 N=1、127、1024、100003、1048576 逐一做正确性校验，并同时输出 TileLang 与 PyTorch 的平均耗时和最大误差。(不必要求TileLang比PyTorch更快)

提交:

运行 `python benchmark_add.py` 后的终端结果截图

## 九齿任务：Vector Add

在 `ninetoothed_add.py` 中完成 `arrangement()` 和 `application()`，实现一维
`float16` 向量加法并处理尾块。默认使用 `BLOCK_SIZE = 1024`。

`arrangement()` 负责按 block 对齐输入和输出，`application()` 负责完成 tile 内的逐元素
相加。`NINETOOTHED_AUTOTUNE=1` 时，block size 会在 256 到 1024 之间自动搜索。

参考：[NineToothed Vector Addition](https://github.com/InfiniTensor/ninetoothed/blob/b77f930dc6c8b016e09adf33570d55a7bc8376c1/docs/source/basics.rst)。

```bash
python -m pip install ninetoothed
python -m pytest -q test_ninetoothed_add.py
python benchmark_ninetoothed_add.py
```

需要尝试自动调优时运行：

```bash
NINETOOTHED_AUTOTUNE=1 python benchmark_ninetoothed_add.py
```

测试使用 `size=98432`；benchmark 对 `2^18` 到 `2^27` 的向量长度比较九齿与
PyTorch 的耗时。

提交完成`ninetoothed_add.py`的测试与 benchmark 输出截图，并用 3 到 5 句话记录九齿的
分块方式和开发体验。
