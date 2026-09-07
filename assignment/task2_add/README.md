# 任务二：NineToothed 与 TileLang 基础算子

使用 NineToothed 和 TileLang 分别实现一维向量加法，熟悉两种 DSL 的基本开发流程。
输入为等长的 `float16` CUDA tensor，输出满足 `C[i] = A[i] + B[i]`，并正确处理尾块。

## 1. 准备环境并运行官方示例

以下命令从仓库根目录开始，并在同一个 shell 中执行。MXMACA 版 PyTorch 仍使用
`cuda` 作为设备名。

```bash
python -m pip install ninetoothed
python -m pip show ninetoothed
python -c "import ninetoothed; print('NineToothed import: OK')"

source ./setup_env.sh
python -c "import tilelang; print('TileLang:', tilelang.__version__)"
python "$TILELANG_ROOT/examples/quickstart.py"
```

如果 `setup_env.sh` 没有找到 `tilelang-metax`，请先按
[TileLang-MetaX 的 MACA 安装说明](https://github.com/tile-ai/tilelang-metax/blob/dev/docs/get_started/Installation_maca.md)
完成构建。提交时保留版本信息和 quickstart 成功运行的终端输出。

## 2. 完成两份 Add

### NineToothed

在 `ninetoothed_add.py` 中完成 `application()`，用 arrange-and-apply 范式计算两个 tile
的和。保持 `BLOCK_SIZE = 1024`，不做自动调优，也不能用 PyTorch 算子代替 kernel。

参考：[NineToothed Vector Addition](https://github.com/InfiniTensor/ninetoothed/blob/b77f930dc6c8b016e09adf33570d55a7bc8376c1/docs/source/basics.rst)。

### TileLang

在 `solution.py` 中完成 `tl_add_1d(A, B, BLOCK_N)`：使用 `T.Kernel` 和 `T.Parallel`
处理 tile，计算全局索引并保护尾块。

参考：[TileLang Vector Add](https://www.tilelang.com/programming_guides/language_basics.html#id7)。

## 3. 验证与对比

继续在已经加载环境的 shell 中，从仓库根目录执行：

```bash
cd assignment/task2_add
python -m pytest -q test_add.py
python benchmark_add.py
```

同一个测试会检查两份 Add，覆盖 `N=1、127、1024、100003、1048576`。benchmark 使用
相同输入和计时方法，对比 NineToothed、PyTorch，以及 `BLOCK_N=256/1024` 的 TileLang
稳态端到端调用延迟；首次编译不计入结果，也不要求自定义 kernel 必须快于 PyTorch。

## 提交内容

1. 完成后的 `ninetoothed_add.py` 和 `solution.py`。
2. TileLang quickstart、正确性测试和 benchmark 的终端输出截图。
3. 用 3 到 5 句话比较两种 DSL 的代码结构和开发体验，并说明 TileLang 的 tile 级表达与
   传统 GPU 逐线程、显式索引编程的差异。
