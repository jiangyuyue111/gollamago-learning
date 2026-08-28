# 基线与性能测量

`*_smoke.json` 只用于确认端到端推理和输出一致性。首次 TileLang 调用包含 JIT
编译时间，冒烟结果不能作为正式性能数据。正式测量必须包含预热，并使用下述统一参数。

## 正式性能测试

正式测试使用 3 次预热和 10 次测量，避免把 TileLang 首次 JIT 编译时间计入稳定性能。
在 MXMACA 机器上先运行 PyTorch 基线：

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

再使用完全相同的模型、prompt、生成长度、seed、设备和迭代参数运行 TileLang：

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

构建扩展后，以相同条件运行 MXMACA C++：

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

分别运行比较器：

```shell
python benchmarks/compare_results.py \
  benchmarks/results/torch_maca.json \
  benchmarks/results/tilelang_maca.json \
  --output-json benchmarks/results/torch_vs_tilelang_maca.json

python benchmarks/compare_results.py \
  benchmarks/results/torch_maca.json \
  benchmarks/results/maca_cpp_maca.json \
  --output-json benchmarks/results/torch_vs_maca_cpp_maca.json
```

比较器会检查设备、软件版本、输入条件和生成 token IDs。条件不一致或输出不一致时，
结果会被拒绝，不能用于性能结论。

所有结果使用相同模型、prompt、随机种子、batch size、输入长度、生成长度、设备和
软件版本。基线固定使用 PyTorch 算子，正式测试命令见上方“正式性能测试”章节。

计算口径：

```text
tokens/s = batch_size * 每条序列生成 token 数 / 平均耗时
加速比 = 优化版 tokens/s / PyTorch 基线 tokens/s
性能提升率 = (加速比 - 1) * 100%
```

比较器会检查测试环境字段和生成 token IDs。任一不一致时结果无效。基线 JSON 应由
课程指定的 MXMACA 机器生成并随仓库发布，不应复制其他设备的数据。

## 仓库实测结果

2026-08-28 在 MetaX C500、PyTorch `2.8.0+metax3.5.3.9`、MACA `3.5.3.9`
环境按上述命令测得：

| 实现 | 平均延迟 | P50 | P90 | tokens/s |
|---|---:|---:|---:|---:|
| PyTorch 基线 | 727.14 ms | 705.70 ms | 745.78 ms | 88.02 |
| TileLang 示例 | 1001.66 ms | 964.30 ms | 1140.50 ms | 63.89 |
| MXMACA C++ | 682.44 ms | 646.48 ms | 751.83 ms | 93.78 |

三个后端生成的 64 个 token IDs 完全一致。TileLang 示例加速比为 `0.726x`，性能
提升率为 `-27.41%`；MXMACA C++ 加速比为 `1.065x`，性能提升率为 `6.55%`。两个
示例都不代表已达到课程的 80% 要求。原始结果及比较 JSON 位于 `results/`。
