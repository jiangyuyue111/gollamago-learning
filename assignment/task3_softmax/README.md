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
