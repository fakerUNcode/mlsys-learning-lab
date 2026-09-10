# CUDA Kernels

从可读 baseline 出发，学习线程映射、内存访问、同步、占用率和 profiler 驱动的 CUDA kernel 优化。

## 当前状态

该模块目前处于规划阶段，尚无可构建的自定义 kernel。已有 CUDA 概念笔记位于 [`Operator-notes/CUDA`](../Operator-notes/CUDA/)，导航见[前置资料](../learning/stage-00/before-learning/README.md)。

## 实验阶梯

1. vector add 与 ReLU；
2. reduction 与 prefix sum；
3. transpose 与访存合并；
4. tiled GEMM；
5. Softmax、LayerNorm 与融合算子。

每个主题先实现清楚的 baseline，再根据 Nsight 或 benchmark 证据添加优化版本。

## 目录约定

```text
cuda_kernels/
├── README.md
├── <topic>/
│   ├── kernel.cu
│   ├── reference.py
│   ├── test_<topic>.py
│   └── notes.md
└── CMakeLists.txt
```

## 验证要求

- 与 NumPy 或 PyTorch reference 比较多个 shape 和 dtype。
- 覆盖空输入、小输入、非整除 block 尺寸和边界索引。
- 检查每个 CUDA API 与 kernel launch 的错误。
- 记录 GPU、Compute Capability、grid/block、寄存器和 shared memory。
- 区分 kernel 时间、传输时间和端到端时间。
- 使用 profiler 解释 occupancy、访存、分支和 stall。

## 完成标准

代码能在声明的 GPU 架构上构建；误差阈值明确；测试和 benchmark 可从仓库根目录运行；优化结论有 baseline 与 profiler 证据。
