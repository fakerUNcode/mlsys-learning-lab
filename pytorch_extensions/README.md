# PyTorch Extensions

把 C++/CUDA 实现安全地接入 PyTorch，覆盖 Python API、ATen Tensor 检查、构建打包、CPU fallback、CUDA 路径和自动化测试。

## 当前状态

该模块处于设计阶段，尚无可安装扩展。阶段 1 正在准备 C++ 生命周期、ABI、CMake 和错误处理基础，见[阶段 1](../learning/stage-01/README.md)。

## 目录约定

```text
pytorch_extensions/
└── <extension_name>/
    ├── pyproject.toml
    ├── binding.cpp
    ├── cpu.cpp
    ├── kernel.cu
    ├── ops.py
    └── tests/
```

## 接口约定

- 显式检查 device、dtype、shape、layout 和 contiguous 要求。
- 遵循 PyTorch 当前 stream，不在库代码中隐式搬运 Tensor。
- 检查 CUDA API 和 kernel launch 错误，并转换为清楚的上层错误。
- 若算子参与训练，需要实现并测试 backward。
- 记录 PyTorch、CUDA、编译器、ABI 和 Compute Capability。
- Python 引用、C++ 对象和 Device buffer 的所有权必须清楚。

## 验证矩阵

至少覆盖 CPU、CUDA、支持的 dtype、典型与边界 shape、非连续输入、错误输入和梯度检查。无 GPU 环境应明确 skip GPU 测试。

## 完成标准

扩展可以 editable install 和 import；CPU fallback 与 GPU 路径结果一致；错误信息可操作；可在干净环境重建；性能结论包含 PyTorch reference 和完整环境。
