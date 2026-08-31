# Llama 算子练习仓库

这是一个小型 Llama 推理项目，用来练习 PyTorch、TileLang 和 MXMACA 算子。
模型已经接好统一的算子调用流程，学员主要编写 kernel。

## 快速开始

安装依赖：

```shell
python -m pip install -r requirements.txt
```

需要模型时，先下载 `meta-llama/Llama-3.2-1B` 到 `models/Llama-3.2-1B`，再运行：


## 选择后端

`--backend` 决定使用哪份算子代码：

| 后端 | 用途 |
|---|---|
| `torch` | 参考实现和 CPU 基线 |
| `tilelang` | TileLang kernel |
| `maca_cpp` | MXMACA `.maca` kernel |
| `NineToothed` |  |

`--target` 可选 `auto`、`cuda`、`maca`。MACA 版 PyTorch 仍使用 `--device cuda`。
如果后端、target、扩展或某个算子不可用，或者 kernel 运行出错，框架会打印警告并自动使用 PyTorch。
推理最终输出的 `registered_operators` 会列出当前后端已接入的算子；值为 `torch_fallback`
表示该算子暂时仍调用 PyTorch reference。

TileLang测试：

先加载预装的 TileLang 开发环境。脚本会自动发现 `/app/tilelang-metax`：

```shell
source ./setup_env.sh
# 自定义源码位置：TILELANG_ROOT=/path/to/tilelang-metax source ./setup_env.sh
```

```shell
python infer.py --model models/Llama-3.2-1B --prompts "Hello" \
  --max-new-tokens 1 --backend tilelang --target maca --device cuda
```

MXMACA测试：

```shell
python operators/maca_cpp/setup.py build_ext --inplace
python infer.py --model models/Llama-3.2-1B --prompts "Hello" \
  --max-new-tokens 1 --backend maca_cpp --target maca --device cuda
```

## 学员要改什么

框架已经预置 `rms_norm` 和 `rope` 的调用、注册和测试入口。当前 TileLang
`rms_norm` 仅用于演示算子接入流程，并非性能最优实现；学员可以在此基础上继续优化
线程布局、访存和归约方式。

优化范围不限于现有示例或文档推荐的某一个算子。学员可以根据自己的能力和目标，自行
选择更多适合的模型算子进行实现和优化。RoPE 的接口是：

```python
rope(input, sin_table, cos_table) -> output
```

学员只需要实现对应的 TileLang 或 MXMACA kernel，不需要修改 `llama.py`、注册表或
命令行。还没有实现的槽位会暂时使用 PyTorch reference 并打印警告，所以示例可以直接
跑通；这种状态不能用于性能结论。完整说明见 [`operators/INTEGRATION.md`](operators/INTEGRATION.md)。

## 测试和性能

下面提供轻量性能对比，统一生成 16 个 token，使用 1 次 warmup、3 次测量，并保持模型、
prompt、seed、精度和设备完全一致。该配置用于快速反馈，结果波动较大，不作为正式性能结论。
下面命令假设在 MACA 机器上运行，模型目录是 `models/Llama-3.2-1B`。

先测 PyTorch 基线：

```shell
python infer.py \
  --model models/Llama-3.2-1B \
  --prompts "Hello" \
  --max-new-tokens 16 \
  --backend torch --target maca --device cuda \
  --num-warmup-iterations 1 \
  --num-profiling-iterations 3 \
  --seed 0 \
  --output-json benchmarks/results/torch_maca.json
```

再测 TileLang：

```shell
python infer.py \
  --model models/Llama-3.2-1B \
  --prompts "Hello" \
  --max-new-tokens 16 \
  --backend tilelang --target maca --device cuda \
  --num-warmup-iterations 1 \
  --num-profiling-iterations 3 \
  --seed 0 \
  --output-json benchmarks/results/tilelang_maca.json
```

MXMACA 原生算子需要先构建扩展：

```shell
python operators/maca_cpp/setup.py build_ext --inplace
python infer.py \
  --model models/Llama-3.2-1B \
  --prompts "Hello" \
  --max-new-tokens 16 \
  --backend maca_cpp --target maca --device cuda \
  --num-warmup-iterations 1 \
  --num-profiling-iterations 3 \
  --seed 0 \
  --output-json benchmarks/results/maca_cpp_maca.json
```

比较 TileLang 和 PyTorch：

```shell
python benchmarks/compare_results.py \
  benchmarks/results/torch_maca.json \
  benchmarks/results/tilelang_maca.json \
  --output-json benchmarks/results/torch_vs_tilelang_maca.json
```

比较结果中的 `modified_operators` 会列出候选后端相对于基线实际接入的原生算子；
`torch_fallback` 不会被计为优化算子。

比较 MXMACA 原生实现和 PyTorch：

```shell
python benchmarks/compare_results.py \
  benchmarks/results/torch_maca.json \
  benchmarks/results/maca_cpp_maca.json \
  --output-json benchmarks/results/torch_vs_maca_cpp_maca.json
```

比较器会检查测试条件和生成的 token IDs。首次 TileLang 调用包含 JIT 编译，不能直接
拿第一次运行的时间评价性能。如果 RoPE 还在使用临时 PyTorch reference，也不能把该
结果当作完整后端性能；应先实现对应 kernel。

## 目录

```text
infer.py                         推理入口
llama.py                         Llama 模型和算子调用点
backends.py                      后端与 target 配置
operators/registry.py            算子注册和分发
operators/torch_ops.py           PyTorch 参考实现
operators/tilelang_ops.py        TileLang 实现槽位
operators/maca_cpp/              MXMACA 构建脚本和源码
operators/INTEGRATION.md         接入教程和推荐练习
benchmarks/compare_results.py    性能结果比较
tests/                           单元测试和加速器测试
```
