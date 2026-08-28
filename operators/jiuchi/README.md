# 九齿算子接入点

安装并导入名为 `jiuchi_kernels` 的包，提供以下统一接口：

```python
rms_norm(input, weight, eps) -> output
```

算子必须保持输入的 shape、dtype 和 device。仓库不会静默回退 PyTorch，缺失
接口会直接报错，以保证性能测试确实运行了提交的算子。
