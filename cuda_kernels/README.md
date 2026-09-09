# cuda_kernels/

## 职责

研究 CUDA kernel 从线程映射、内存访问、共享内存、同步、占用率到性能分析的完整链路。每个实验应同时提供可读的 baseline 和有明确假设的优化版本。

## 推荐布局

```text
cuda_kernels/
├── README.md
├── <topic>/
│   ├── kernel.cu
│   ├── reference.py 或 reference.cpp
│   ├── test_<topic>.py
│   └── notes.md
└── CMakeLists.txt（当需要独立 C++ 构建时）
```

## 验证重点

先与 NumPy/PyTorch reference 做多形状、多 dtype 比较，再测吞吐和显存。记录 GPU 型号、Compute Capability、block/grid、寄存器、shared memory、occupancy，以及 Nsight Compute/Systems 的关键指标。注意边界条件、非连续 tensor、对齐和错误检查。

## 完成标准

代码能在目标 Compute Capability 上编译，数值误差阈值明确，benchmark 可重复，优化结论有 profiler 证据。
