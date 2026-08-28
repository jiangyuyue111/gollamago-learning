# MXMACA 原生算子实现

本目录提供可以直接构建的 MXMACA 扩展，不再只是接口占位。示例只包含一个 BF16
RMSNorm device kernel，并通过 PyTorch Python 扩展接入 Llama 推理。

## 已实现算子

```python
rms_norm(input, weight, eps) -> output
```

| 算子 | 实现方式 |
|---|---|
| `rms_norm` | 每行一个 block，FP32 shared-memory reduction，融合归一化与权重乘法 |

源码结构：

```text
operators/maca_cpp/
├── __init__.py          registry 注册适配
├── setup.py             PyTorch 扩展及原生 mxcc 构建脚本
└── src/
    ├── bindings.cpp     参数检查、输出分配和 Python 绑定
    └── rms_norm.maca    MXMACA device kernel 与 launch 函数
```

## 环境要求

- MACA Toolkit，环境变量 `MACA_PATH` 指向安装目录；
- MACA 版 PyTorch，`torch.version.maca` 不为空；
- 可用的 `mxcc` 编译器；
- 可用的 MetaX GPU。

当前环境使用 `/opt/maca/mxgpu_llvm/bin/mxcc`。`setup.py` 直接使用
`mxcc -x maca -offload-arch native` 编译 `rms_norm.maca`，不再把设备源码作为
CUDA `.cu` 文件交给兼容层。可通过 `MACA_PATH`、`MXCC` 和 `MACA_ARCH` 覆盖路径及架构。

## 构建

在仓库根目录执行：

```shell
python operators/maca_cpp/setup.py build_ext --inplace
```

编译产物会生成在 `operators/maca_cpp/` 下，并作为
`operators.maca_cpp.maca_kernels` 导入。`build/` 和 `*.so` 是本机生成物，已被
`.gitignore` 排除；提交仓库时只提交源码和构建脚本。

确认扩展加载成功：

```shell
python -c "from operators.maca_cpp import maca_kernels; print(maca_kernels.__file__)"
```

## 输入约束

- tensor 必须位于 CUDA 兼容的 MACA device；
- 当前实现支持连续存储的 BF16 tensor；
- `rms_norm` 的 `weight` 必须是一维，长度等于输入最后一维；
- 输出保持输入的 shape、dtype 和 device；
- 不支持的输入会明确报错，不会回退到 PyTorch。

kernel 在 PyTorch 当前 CUDA/MACA stream 上执行，因此能够遵守模型已有的 stream
依赖关系。

## 正确性测试

构建完成后运行：

```shell
RUN_ACCELERATOR_TESTS=1 pytest -m accelerator -k maca_cpp
```

测试将 RMSNorm 与 PyTorch reference 对齐。当前 MetaX C500 环境结果为：

```text
1 passed
```

## Llama 端到端验证

```shell
python infer.py \
  --model models/Llama-3.2-1B \
  --prompts "Hello" \
  --max-new-tokens 1 \
  --backend maca_cpp \
  --target maca \
  --device cuda
```

当前验证生成 token ID `11`，文本为 `Hello,`，与 PyTorch 和 TileLang 后端一致。
正式性能测量应使用 `benchmarks/README.md` 规定的预热、测量次数和固定输入条件。
