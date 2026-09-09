# MLSys Learning Lab 全部相关知识点学习报告

## 摘要

本项目把“模型计算”拆成可观察的系统层次：数学算子 → tensor/内存 → kernel → runtime/扩展 → 图编译 → 推理服务 → 基准与报告。学习目标是能解释性能，而不只是调用 API。

## 知识地图

| 层次 | 核心知识 | 在本仓库的落点 |
| --- | --- | --- |
| 数值计算 | dtype、误差、稳定性、广播、布局 | `tests/`, `inference/` |
| GPU 硬件 | SM、warp、occupancy、寄存器、shared/global memory | `cuda_kernels/` |
| CUDA 编程 | grid/block、同步、stream、异步拷贝、错误检查 | `cuda_kernels/` |
| PyTorch 系统 | eager、autograd、ATen、dispatcher、tensor metadata | `pytorch_extensions/` |
| 编译构建 | C++ ABI、CMake、Ninja、CUDA arch、wheel/editable install | `pytorch_extensions/`, `scripts/` |
| 编译器 | graph capture、IR、fusion、tiling、layout、codegen、动态 shape | `compiler/` |
| 推理系统 | batching、并发、warmup、KV cache、量化、显存复用 | `inference/` |
| 性能工程 | roofline、带宽、算术强度、profiling、分位数 | `benchmarks/`, `reports/` |
| 工程复现 | Git、环境快照、随机种子、配置、CI、artifact | 全仓库 |

## 关键概念与应回答的问题

### 1. 正确性先于性能

CPU/PyTorch reference 是 oracle。需要明确绝对/相对误差、NaN/Inf、边界尺寸、dtype 和非连续输入。优化可能改变求和顺序，因此要区分严格 bitwise 一致与数值等价。

### 2. GPU 性能模型

一次 kernel 的瓶颈通常来自计算吞吐、全局内存带宽、访存合并、同步或 launch overhead。用算术强度和 roofline 建立假设，再用 profiler 验证。occupancy 不是越高越好，它会与寄存器、shared memory、ILP 和实际延迟隐藏能力共同决定结果。

### 3. CUDA 执行与内存

理解 host/device、stream、event、同步点、warp divergence、coalescing、shared-memory bank conflict、atomic contention 和 kernel fusion。任何 GPU benchmark 都要避免把异步 launch 当作完成时间。

### 4. PyTorch 扩展边界

扩展必须处理 device、dtype、shape、contiguous、生命周期和当前 stream；训练算子还需要 backward 与 autograd 语义。构建成功不代表 ABI、运行时 CUDA 和目标 GPU 都兼容。

### 5. 编译器抽象

图捕获把 Python/算子调用转换成可分析的 IR；优化 pass 必须满足语义和别名约束。融合减少 launch 和中间 tensor，但可能增加寄存器压力或降低通用性。要分别测编译开销与 steady-state。

### 6. 推理性能

latency、throughput、batch、并发、尾延迟和显存互相制约。量化、编译、算子融合和 KV cache 都有精度、启动成本或内存权衡。服务端指标必须定义请求边界和排队时间。

### 7. Benchmark 科学

固定环境和输入；先 warmup；使用多次采样和 p50/p95/p99；报告方差和异常；对比 baseline；保存原始数据。一个漂亮的单点 speedup 不足以支持一般性结论。

## 推荐学习路线

1. 完成环境自检，理解驱动、CUDA runtime、toolkit 和 PyTorch 的区别。
2. 写 CPU reference 与最简单 CUDA elementwise kernel，验证线程映射和同步计时。
3. 逐步加入 tiled GEMM/reduction，使用 profiler 观察带宽、occupancy 和分支。
4. 将 kernel 封装为 PyTorch extension，加入 CPU fallback、dtype/shape 检查和测试。
5. 搭建 eager 推理 baseline，测量 batch、精度和显存，再尝试 compile/fusion/量化。
6. 保存 FX/TorchInductor/Triton/MLIR 的图或 IR，完成一个可解释的变换。
7. 用统一 benchmark 和报告模板复盘：假设、证据、局限、下一步。

## 每个实验的验收问题

- 我能否在另一台相近环境按 README 重现结果？
- 是否有 reference、边界测试和明确误差阈值？
- 测量是否排除了 warmup、异步和 I/O 偏差？
- 是否知道瓶颈证据，而非只知道最终耗时？
- 优化是否在不同输入规模、dtype 或 batch 下仍成立？
- 报告是否记录了 commit、环境、命令和参数？

## 结论

本仓库的最终能力目标是建立闭环：提出系统假设，写出最小实现，证明正确，进行可重复测量，用 profiler/IR/硬件指标解释结果，再将结论沉淀成可复现报告。剩余任务应按根目录 README 的模块清单逐项落地。
