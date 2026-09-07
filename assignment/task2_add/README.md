# 任务二：TileLang Vector Add Puzzle

实现 1-D `float16` 向量加法：`C[i]=A[i]+B[i]`。`1<=N<=1M`，必须处理尾块（N 不整除 BLOCK_N）。

要求：在 `solution.py` 实现 `tl_add_1d(A,B,BLOCK_N)`；使用 `T.Kernel`、`T.Parallel`；测试 N=1、127、1024、100003、1048576；与 PyTorch 逐元素校验；比较至少两种 BLOCK_N。

运行：

```bash
cd /data/gollamago/assignment/task2_add
source /data/gollamago/setup_env.sh
python -m pytest -q test_add.py
python benchmark_add.py
```

`benchmark_add.py` 会对 N=1、127、1024、100003、1048576 逐一做正确性校验，并同时输出 TileLang 与 PyTorch 的平均耗时和最大误差。(不必要求TileLang比PyTorch更快)

提交:

运行 `python benchmark_add.py` 后的终端结果截图
