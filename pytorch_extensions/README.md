# pytorch_extensions/

## 职责

存放 PyTorch C++/CUDA extension：从 Python API、ATen tensor 检查、编译打包，到 CPU fallback、CUDA 实现、autograd 和测试。

## 推荐布局

```text
pytorch_extensions/
└── <extension_name>/
    ├── pyproject.toml 或 setup.py
    ├── __init__.py
    ├── binding.cpp
    ├── kernel.cu / cpu.cpp
    ├── ops.py
    └── tests/
```

## 实现规范

检查 device、dtype、shape、contiguous 要求和 stream；不要在库代码中隐式搬运数据。为 CPU/GPU/不同 dtype 设计清晰错误信息，若参与训练则实现并测试 backward。构建时固定编译参数并记录 PyTorch、CUDA、编译器和 Compute Capability。

## 完成标准

可 editable install、可 import、CPU fallback 可用、GPU 路径有数值与梯度测试，并能在干净环境中重建。
