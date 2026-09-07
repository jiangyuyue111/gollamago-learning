# 任务三：NineToothed GEMM、TileLang Softmax 与 DSL 对比

本任务建立在任务二之上：用 NineToothed 完成 GEMM，用 TileLang 完成 Softmax，最后复用
任务二的两份 Add 做同一算子对比。

## 1. NineToothed GEMM

在 `ninetoothed_gemm.py` 中实现 `C = A @ B`。`A` 为 `[M,K]`，`B` 为 `[K,N]`；输入和
输出使用 `float16`，中间结果使用 `float32` 累加。

1. 阅读已经给出的矩阵分块和 `expand()`/`squeeze()` arrangement。
2. 在 `application()` 中沿 K 方向遍历 tile，使用 `ntl.dot()` 累加。
3. 将结果转换为 `float16` 并写回，保持固定的 `64 x 64 x 64` block size。

参考：[NineToothed Matrix Multiplication](https://github.com/InfiniTensor/ninetoothed/blob/b77f930dc6c8b016e09adf33570d55a7bc8376c1/docs/source/basics.rst)。

从仓库根目录执行：

```bash
source ./setup_env.sh
cd assignment/task3_softmax
python -m pytest -q test_gemm.py
python benchmark_gemm.py
```

测试包含方阵和非整块的非方阵。benchmark 使用 `M=N=K=1024`，先校验结果，再输出设备、
block size、首次运行耗时，以及 NineToothed 和 `torch.matmul` 的稳态端到端延迟、TFLOPS
和加速比；本任务不设置性能达标线。

## 2. TileLang Softmax

在 `solution.py` 中实现二维按行 Softmax。输入 `A:[N,M] float32`，使用稳定公式
`exp(x-max) / sum(exp(x-max))`，完成最大值归约、指数、求和归约和写回，并处理任意 N/M。

```bash
python -m pytest -q test_softmax.py
python benchmark_softmax.py
```

## 3. 同一算子对比

不能直接比较 GEMM 和 Softmax 的性能。返回任务二，用同一台设备、同一组输入和同一计时
方法，对比 NineToothed Add 与 TileLang Add：

```bash
cd ../task2_add
python benchmark_add.py
```

在提交说明中填写：

| 对比项 | NineToothed Add | TileLang Add |
|---|---|---|
| 核心代码结构 | arrangement / application | `T.Kernel` / `T.Parallel` |
| Tile、索引和尾块如何表达 | 待填写 | 待填写 |
| 稳态端到端延迟（us） | 待填写 | 待填写 |
| 相对 PyTorch 加速比 | 待填写 | 待填写 |
| 核心实现代码行数 | 待填写 | 待填写 |
| 首次正确运行耗时或调试次数 | 待填写 | 待填写 |

最后用一段话说明哪种 DSL 在 Add 任务中开发效率更高，以及判断依据。性能结论必须来自
相同设备、shape、dtype、输入、预热次数和计时方法。

## 提交内容

1. 完成后的 `ninetoothed_gemm.py` 和 `solution.py`。
2. GEMM、Softmax 的测试与 benchmark 输出截图。
3. 上述同算子对比表和简短结论。
