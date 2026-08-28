# 九源 x MXMACA 高性能算子开发实战营

本仓库提供 Llama 3 推理脚手架、PyTorch 性能基线、TileLang 示例算子、MXMACA
C++ 示例算子，以及九齿算子的统一接入点。后端选择会真正进入 Llama 算子调用链，缺失
实现时直接报错，不会静默回退 PyTorch。

## 学习任务

从九齿、TileLang、MACA C++ 中选择至少两种实现方式，实现并优化至少两个算子。
至少一个算子必须集成到本仓库的 Llama 推理流程。

提交内容必须包含：

1. 算子源码和构建说明；
2. 与 PyTorch reference 对齐的正确性测试；
3. 独立算子和端到端 Llama 推理性能数据；
4. 优化方法、测试环境和复现命令；
5. 相同输入下的生成 token IDs。

## 健韬负责事项

下表只列任务表中 Owner 为“健韬”或“嘉成 + 健韬”的事项。

| 原序号 | 负责方式 | 任务 | 实现位置 | 如何使用或验收 | 状态 |
|---|---|---|---|---|---|
| 2 | 独立负责 | 支持 MXMACA 后端 | `backends.py`、`infer.py`、`tests/test_backends.py` | 推理时传入 `--target maca --device cuda`；使用 `--backend torch`、`tilelang` 或 `maca_cpp` 选择实现。详见[后端与硬件 target](#后端与硬件-target)。 | 已完成 |
| 5 | 独立负责 | 预留 MXMACA 算子接入点，并提供可构建的 C++ 示例 | `operators/maca_cpp/__init__.py`、`setup.py`、`src/`、`README.md` | 先运行 `python operators/maca_cpp/setup.py build_ext --inplace`，再使用 `--backend maca_cpp --target maca`。详见[MXMACA C++ 示例](#mxmaca-c-示例)。 | 已完成 |
| 6 | 独立负责 | 提供 TileLang 算子示例 | `operators/tilelang_ops.py`、`tests/test_backend_integration.py`、`llama.py` | 使用 `--backend tilelang --target maca` 运行；执行 `RUN_ACCELERATOR_TESTS=1 pytest -m accelerator -k tilelang` 验证。详见[TileLang 示例](#tilelang-示例)。 | 已完成 |
| 10 | 共同负责 | 明确学习任务并提供 Llama 集成框架 | `operators/registry.py`、`operators/torch_ops.py`、`llama.py` | 新实现调用 `register_operator(backend, name, implementation)` 注册统一接口，再通过 `infer.py --backend <name>` 进行 Llama 端到端验证。 | 已完成 |
| 11 | 共同负责 | 提供基线数据和性能计算方式 | `infer.py`、`benchmarks/compare_results.py`、`benchmarks/results/` | 按固定参数运行 Torch、TileLang、MACA C++，再用比较器校验条件和 token IDs。详见[正式性能测试](#正式性能测试)。 | 已完成 |

统一算子接口为 `rms_norm(input, weight, eps)` 和 `silu_mul(gate, up)`。完整加速器
验收命令为：

```shell
python operators/maca_cpp/setup.py build_ext --inplace
RUN_ACCELERATOR_TESTS=1 pytest -q
```

当前 MetaX C500 环境结果为 `19 passed`。

## 仓库结构

```text
infer.py                         推理和端到端性能入口
llama.py                         基础 Llama 3 模型
backends.py                      实现后端与硬件 target 配置
operators/registry.py            统一算子注册和分发
operators/torch_ops.py           PyTorch reference
operators/tilelang_ops.py        TileLang RMSNorm、SiLU x Gate 示例
operators/maca_cpp/              MXMACA C++ RMSNorm、SiLU x Gate 及构建脚本
operators/jiuchi/                九齿扩展接入点
benchmarks/                      基线协议和结果比较工具
tests/                           单元测试和加速器测试
```

当前接入模型的统一算子签名为：

```python
rms_norm(input, weight, eps) -> output
silu_mul(gate, up) -> output
```

新增实现需通过 `operators.register_operator()` 注册相同签名。具体扩展约定见
`operators/maca_cpp/README.md` 和 `operators/jiuchi/README.md`。

## 安装与模型下载

```shell
python -m pip install -r requirements.txt
```

`meta-llama/Llama-3.2-1B` 是受限模型。先在 Hugging Face 模型页面接受许可，创建
read token 并登录：

```shell
hf auth login
hf download meta-llama/Llama-3.2-1B --local-dir models/Llama-3.2-1B
```

不要把 token 或下载的权重提交到仓库。

## 后端与硬件 target

`--backend` 表示算子实现方式：

| 参数 | 实现 |
|---|---|
| `torch` | PyTorch reference 和基线 |
| `tilelang` | 仓库提供的 TileLang 示例 |
| `maca_cpp` | 仓库提供的 MXMACA C++ 扩展 |
| `jiuchi` | 学员提供的 `jiuchi_kernels` 包 |

`--target` 表示编译/运行平台，可选 `auto`、`cuda`、`maca`。MACA 版 PyTorch
通过 CUDA 兼容接口提供设备，因此 MXMACA 上仍使用 `--device cuda`。

`backends.py` 会根据 `torch.version.maca` 自动识别 MACA 环境，将 `maca` target
传给 TileLang，并限制 `maca_cpp` 只能在 MACA target 下运行。推理计时前后会同步
设备；自定义后端缺少算子时直接报错，不会静默回退 PyTorch。

PyTorch 基线：

```shell
python infer.py --model models/Llama-3.2-1B --prompts "Hello" \
  --max-new-tokens 1 --backend torch --target maca --device cuda
```

在 MXMACA 上运行 TileLang 示例：

```shell
python infer.py --model models/Llama-3.2-1B --prompts "Hello" \
  --max-new-tokens 1 --backend tilelang --target maca --device cuda
```

构建并运行 MXMACA C++ 示例：

```shell
python operators/maca_cpp/setup.py build_ext --inplace
python infer.py --model models/Llama-3.2-1B --prompts "Hello" \
  --max-new-tokens 1 --backend maca_cpp --target maca --device cuda
```

在 NVIDIA GPU 上只需把 `--target maca` 改为 `--target cuda`。也可以使用
`--target auto` 自动检测。纯 CPU 冒烟测试使用 `--backend torch --device cpu`。

以上生成 1 个 token 且没有 warmup 的命令只用于确认后端能加载、kernel 能编译、
Llama 能生成结果。TileLang 首次调用包含 JIT 编译时间，不能使用冒烟结果评价性能。

## TileLang 示例

`operators/tilelang_ops.py` 包含两个可运行示例：

- RMSNorm：展示分块读取、FP32 reduction 和权重融合；
- SiLU x Gate：把 SiLU 激活和逐元素乘法融合为一个 kernel。

两个算子分别替换 `llama.RMSNorm.forward()` 和 `llama.MLP.forward()` 中的
对应 PyTorch 计算。TileLang 在首次使用时 JIT 编译，编译时间不计入预热后的性能
样本。

## MXMACA C++ 示例

`operators/maca_cpp/` 提供可构建的 BF16 扩展源码：

- RMSNorm：每行一个 block，使用 FP32 shared-memory reduction，并融合权重乘法；
- SiLU x Gate：在一个 elementwise kernel 内完成激活和乘法；
- kernel 使用 PyTorch 当前 CUDA/MACA stream；
- 输入检查要求连续 BF16 tensor，并保持输出 shape、dtype 和 device；
- 为保持端到端生成 token 一致，kernel 显式保留 PyTorch reference 的 BF16 舍入边界。

构建命令必须从仓库根目录执行：

```shell
python operators/maca_cpp/setup.py build_ext --inplace
```

生成的 `build/` 和 `*.so` 是本机编译产物，不应提交。完整接口和构建说明见
`operators/maca_cpp/README.md`。

## 正确性测试

不需要 GPU 的测试：

```shell
pytest
```

在 CUDA 或 MXMACA 环境运行真实 kernel 编译和数值测试：

```shell
RUN_ACCELERATOR_TESTS=1 pytest -m accelerator
```

只验证 MXMACA TileLang：

```shell
RUN_ACCELERATOR_TESTS=1 pytest -m accelerator -k tilelang
```

构建扩展后，只验证 MXMACA C++：

```shell
python operators/maca_cpp/setup.py build_ext --inplace
RUN_ACCELERATOR_TESTS=1 pytest -m accelerator -k maca_cpp
```

当前 MetaX C500 环境中，TileLang 和 MXMACA C++ 各有两个真实加速器测试；完整结果
为 `19 passed`。其中 MXMACA C++ 算子使用零容差与 PyTorch BF16 reference 对齐。

## 正式性能测试

课程统一条件为 BF16、固定模型和 prompt、固定 seed、warmup 3 次、profiling 10 次。
结果至少记录平均延迟、P50/P90、tokens/s、峰值显存、环境版本及生成 token IDs。

先运行 PyTorch 基线：

```shell
python infer.py \
  --model models/Llama-3.2-1B \
  --prompts "Hello" \
  --max-new-tokens 64 \
  --backend torch \
  --target maca \
  --device cuda \
  --num-warmup-iterations 3 \
  --num-profiling-iterations 10 \
  --seed 0 \
  --output-json benchmarks/results/torch_maca.json
```

使用完全相同条件运行 TileLang：

```shell
python infer.py \
  --model models/Llama-3.2-1B \
  --prompts "Hello" \
  --max-new-tokens 64 \
  --backend tilelang \
  --target maca \
  --device cuda \
  --num-warmup-iterations 3 \
  --num-profiling-iterations 10 \
  --seed 0 \
  --output-json benchmarks/results/tilelang_maca.json
```

构建扩展后，使用完全相同条件运行 MXMACA C++：

```shell
python operators/maca_cpp/setup.py build_ext --inplace
python infer.py \
  --model models/Llama-3.2-1B \
  --prompts "Hello" \
  --max-new-tokens 64 \
  --backend maca_cpp \
  --target maca \
  --device cuda \
  --num-warmup-iterations 3 \
  --num-profiling-iterations 10 \
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

比较 MXMACA C++ 和 PyTorch：

```shell
python benchmarks/compare_results.py \
  benchmarks/results/torch_maca.json \
  benchmarks/results/maca_cpp_maca.json \
  --output-json benchmarks/results/torch_vs_maca_cpp_maca.json
```

三种实现必须使用相同模型、prompt、精度、设备、seed、batch size、输入长度和生成
长度。比较器会检查环境字段和生成 token IDs，条件或输出不一致时拒绝比较。三次
warmup 用于排除 JIT 和运行时初始化，十次 profiling 用于计算稳定统计量。

计算口径：

```text
tokens/s = batch_size * 每条序列生成 token 数 / 平均耗时
加速比 = 优化版 tokens/s / PyTorch 基线 tokens/s
性能提升率 = (加速比 - 1) * 100%
```

2026-08-28 在 MetaX C500、PyTorch `2.8.0+metax3.5.3.9`、MACA `3.5.3.9`
环境按上述命令测得：

| 实现 | 平均延迟 | P50 | P90 | tokens/s |
|---|---:|---:|---:|---:|
| PyTorch 基线 | 727.14 ms | 705.70 ms | 745.78 ms | 88.02 |
| TileLang 示例 | 1001.66 ms | 964.30 ms | 1140.50 ms | 63.89 |
| MXMACA C++ | 682.44 ms | 646.48 ms | 751.83 ms | 93.78 |

三个后端生成的 64 个 token IDs 完全一致。TileLang 加速比为 `0.726x`，性能提升率
为 `-27.41%`；MXMACA C++ 加速比为 `1.065x`，性能提升率为 `6.55%`。当前示例
尚未达到课程的 80% 性能要求，仍需继续优化。正式 JSON 位于
`benchmarks/results/`，更完整的测量协议见 `benchmarks/README.md`。

## 评分

基础要求为性能提升率达到 80%，同时保持输出 token IDs 一致。满足基础要求的提交
再按课程公布的进阶筛选规则评审。

## 限制条件

性能提升必须来自九齿、TileLang 或 MACA C++ 算子实现及合理的算子融合。直接调用
更高层 PyTorch 融合实现替换作业算子，或通过 KV cache 等系统级改动改变基线工作量，
不计入本次算子优化成绩。如对改动边界有疑问，应在提交前向课程助教确认。
