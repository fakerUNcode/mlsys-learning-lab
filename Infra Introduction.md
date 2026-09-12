# Infra学习指南

面向准备进入 AI Infra、CUDA 算子、推理优化或 ML Systems 方向的学习者。本路线将目标拆成两个时期：实习前建立可投递的核心能力，实习后再扩展到训推系统、分布式、编译器和硬件。

> 当前阶段：阶段 1，C++ 必需基础；第 00～03 节已完成，下一步学习第 04 节模板与类型特性。
>
> 预计投递：约一年后。
>
> 优先方向：CUDA、量化、并行计算。
>
> 学习记录：[每日记录与总进度](daily-record/README.md)。

## 路线原则

“从模型到芯片”是长期方向，不是实习前必须完成的清单。实习前只追求一条短而完整的能力链：

```text
C++ 最低闭环
→ GPU 并行模型
→ CUDA 算子
→ Nsight 分析
→ PyTorch/Triton 接入
→ 推理量化
→ 综合项目
```

每个阶段必须留下四类证据：

1. **代码证据**：能够构建、运行和测试；
2. **数据证据**：包含 benchmark、profiling 或误差数据；
3. **解释证据**：说明瓶颈、取舍和失败实验；
4. **复现证据**：记录环境、命令、版本、硬件和 Git commit。

不以“看完课程”作为完成标准。完成意味着能够独立实现简化版本、解释运行过程、用工具定位问题，并在输入或约束改变时调整方案。

## 双段总览

| 时期 | 目标 | 主要成果 |
| --- | --- | --- |
| 实习前 | 获得 CUDA/推理优化岗位所需的核心实践能力 | CUDA 算子、PyTorch 扩展、量化实验、作品集 |
| 实习后 | 从局部算子深入完整 ML Systems | 训推 Runtime、分布式、编译系统、硬件协同 |

# 实习前

建议用 10 个月完成主体学习，保留最后 2 个月用于项目打磨、复习和投递。时间是上限参考；达到验收线后即可进入下一阶段。

## 阶段总览

| 阶段 | 用时 | 核心产出 | 通过标志 |
| --- | ---: | --- | --- |
| 0. 工具基线 | 1～2 周 | 环境快照与可信 benchmark | 能定位环境和计时问题 |
| 1. C++必需 | 3～4 周 | GPU 资源包装与 C++ 小项目 | 能解释生命周期、构建和错误 |
| 2. 并行基础 | 3～4 周 | CPU/GPU microbenchmark | 能判断主要性能限制 |
| 3. CUDA核心 | 6～8 周 | Vector Add、Reduce、Transpose | 能独立写对并测准 kernel |
| 4. 算子优化 | 6～8 周 | Softmax、Norm 与 GEMM 教学版 | 能用 Nsight 解释收益 |
| 5. 框架接入 | 4～5 周 | PyTorch 扩展与 Triton 对照 | 算子可安装、测试和调用 |
| 6. 推理量化 | 5～6 周 | INT8/PTQ 实验与误差报告 | 能解释精度和性能取舍 |
| 7. 主线项目 | 6～8 周 | 完整 CUDA 推理算子项目 | 可作为简历主项目 |
| 8. 投递准备 | 4～6 周 | 简历、讲稿和面试复盘 | 能清楚讲述设计与证据 |

总用时约 38～51 周。阶段可以重叠少量复习，但同一时期只保留一个主项目。

## 阶段0

目标：让实验可以复现，并建立正确的性能测量习惯。

### 必学

- Linux/WSL、Git、Python 虚拟环境；
- NVIDIA 驱动、CUDA Runtime、Toolkit 和 PyTorch CUDA 的关系；
- CMake、Ninja、`nvcc` 的职责；
- warmup、同步、重复采样和统计量；
- CPU wall time、CUDA Event 和 profiler 时间的区别。

### 产出

- 一条命令输出硬件、驱动、CUDA、PyTorch、编译器和 commit；
- CPU/GPU Vector Add benchmark；
- 一份环境和计时报告。

### 验收

- 能解释“PyTorch 可用 GPU”和“本机可编译 CUDA”为什么不同；
- 性能数字能追溯到输入、环境、命令和代码版本；
- 能发现未同步造成的错误计时。

## 阶段1

目标：只学习 CUDA Host 代码与 PyTorch 扩展真正需要的 C++。

### 必学

- RAII、对象生命周期、拷贝与移动；
- `unique_ptr`、`shared_ptr`、`weak_ptr`；
- `vector`、`map`、迭代器和常用算法；
- 模板、泛型和类型 traits 的阅读能力；
- `optional`、`variant`、`string_view`、结构化绑定；
- exception、error code 和 CUDA 错误边界；
- ABI、动态库、符号、CMake、CTest 和 Sanitizer。

> 注：`map`/`unordered_map` 根据任务选用。学习重点是容器语义，不是背诵全部 STL API。

### 略学

- allocator 的职责和内存池思想；
- C++20 concept 的接口约束；
- 协程的暂停与恢复模型；
- ABI 的常见兼容风险。

略学内容只需“能读懂、能查资料”，暂不自行实现复杂 allocator 或协程框架。

### 暂缓

- 通用 RPC 框架；
- 完整线程池和复杂任务取消；
- 网络服务治理；
- 模板元编程技巧题。

### 产出

实现一个小型 C++ GPU 资源模型：用 RAII 模拟或包装 Buffer、Stream、Event 的所有权，支持移动、错误传播、单元测试和 Sanitizer 构建。

### 验收

- 能解释 GPU buffer 为什么通常禁止浅拷贝；
- 能为 workspace、共享模型和模型缓存选择智能指针；
- 能排查一次 `.so` 缺失符号或动态库查找问题；
- CTest、ASan 和 UBSan 通过。

当前材料见 [阶段 1](learning/stage-01/README.md)。

## 阶段2

目标：建立分析 CUDA 性能所需的最小体系结构模型。

### 必学

- CPU cache line、局部性和 SIMD 基本思想；
- GPU SM、Warp、Block、Grid；
- SIMT 与 SIMD 的区别；
- Global/Shared Memory、寄存器和同步；
- 内存带宽、延迟、算术强度和 Roofline；
- coalescing、bank conflict、warp divergence；
- false sharing 作为 CPU 并行对照。

### 产出

完成连续/跨步访存、CPU/GPU Vector Add 和简单矩阵运算 microbenchmark，记录数据规模、延迟和有效带宽。

### 验收

能根据数据判断瓶颈主要来自计算、带宽、延迟、同步还是 launch overhead，并给出下一步实验，而不是直接猜优化方案。

## 阶段3

目标：能够独立编写正确、可测量的 CUDA kernel。

### 必学

- 一维和多维线程索引；
- grid-stride loop 与边界处理；
- `cudaMalloc`、`cudaMemcpy`、Stream 和 Event；
- kernel 异步执行和错误检查；
- shared memory、同步和原子操作；
- warp shuffle 的基本使用；
- Compute Sanitizer 与 Nsight Systems 入门。

### 算子阶梯

```text
Vector Add
→ ReLU
→ Reduction
→ Transpose
```

每个算子都保留 PyTorch/CPU reference、naive kernel、边界测试和 benchmark。

### 验收

- 覆盖空输入、小输入和非整除 block 尺寸；
- 能区分 H2D、kernel、D2H 和端到端时间；
- 能定位一次 Device 越界或异步错误；
- 能解释 block size 变化为何可能改善或损害性能。

## 阶段4

目标：从“会写 kernel”进入“用证据优化算子”。

### 必学

- Nsight Systems 时间线；
- Nsight Compute 的吞吐、访存、occupancy 和 warp stall；
- reduction 的 shared memory 与 shuffle 优化；
- transpose 的合并访问和 bank conflict；
- Softmax 的数值稳定性；
- LayerNorm/RMSNorm 的归约、访存和融合；
- GEMM tiling 与 Tensor Core 的基本原理。

### 范围

GEMM 只实现教学版本并与 cuBLAS 对照，不把“超过 cuBLAS”设为验收目标。Attention 只理解计算与内存瓶颈，不在本阶段复刻完整 FlashAttention。

### 产出

- 一个 Reduction 优化阶梯；
- 一个 Softmax 或 RMSNorm 优化项目；
- Nsight 报告与性能分析。

### 验收

每次优化都能回答：改变了什么、影响哪个硬件瓶颈、数据是否支持判断、在哪些 shape 上失效。

## 阶段5

目标：让底层算子成为 PyTorch 可以安全调用的组件。

### 必学

- PyTorch C++/CUDA Extension 构建；
- ATen Tensor 的 device、dtype、shape、stride；
- CPU fallback 与 CUDA dispatch；
- 当前 CUDA Stream 和异步语义；
- autograd 的使用边界；
- Triton program model 与自动调优入门。

### 产出

为同一算子提供三种实现：PyTorch reference、C++/CUDA extension 和 Triton，并用相同输入矩阵验证正确性和性能。

### 验收

- 扩展可安装、导入和重建；
- 错误输入给出明确提示；
- 支持约定的 dtype、shape 和非连续输入策略；
- 性能对比不隐藏编译、搬运或同步成本。

## 阶段6

目标：理解量化如何同时改变数值误差、内存流量和硬件执行。

### 前置基础

只需掌握线性映射和误差概念：

\[
q
\overset{\text{缩放取整}}{=}
\operatorname{round}
\left(
\frac{x}{s}
\right)
+z
\]

\[
\hat{x}
\overset{\text{反量化}}{=}
s(q-z)
\]

### 符号说明

- \(x\)：原始浮点值。
- \(q\)：量化后的整数值。
- \(s\)：scale，缩放因子。
- \(z\)：zero-point，零点。
- \(\hat{x}\)：反量化近似值。
- \(\operatorname{round}\)：取整算子。

### 必学

- FP32、FP16、BF16 和 INT8 的表示差异；
- 对称/非对称量化；
- per-tensor 与 per-channel；
- PTQ、校准、饱和和舍入误差；
- 权重大小、显存带宽、计算吞吐与反量化开销；
- 量化前后的误差与端到端性能测量。

### 暂缓

- 完整 QAT 训练工程；
- GPTQ、AWQ 的源码级复现；
- FP8 分布式训练；
- 面向特定芯片的极限量化格式。

### 产出

实现一个 INT8 量化/反量化实验，并完成量化线性层或 weight-only 推理对照，报告模型大小、误差、延迟和吞吐。

### 验收

能解释精度下降来自哪里，性能收益是否被反量化、数据搬运或小输入开销抵消，并明确结论的设备与 shape 范围。

### 直观理解

量化像把精密刻度改成较粗刻度：数字更省空间、搬运更快，但必须记录换算比例。刻度太粗会丢失细节；换算工作太多时，省下的搬运时间也可能被抵消。

## 阶段7

目标：把前六阶段收束为一个可以展示、复现和深入追问的项目。

### 项目主题

推荐主题：**PyTorch 可调用的 CUDA 推理算子实验室**。

最小范围：

1. PyTorch reference；
2. naive CUDA kernel；
3. 至少一个有证据的优化版本；
4. FP16/BF16 中的一种低精度路径；
5. INT8 或 weight-only 量化路径；
6. C++/CUDA Extension；
7. 可选 Triton 对照；
8. 正确性、边界和误差测试；
9. CUDA Event benchmark；
10. Nsight 分析和复现文档。

算子顺序建议为 Reduction → Softmax → RMSNorm → 量化线性层。优先完成闭环，不强求全部实现。

### 验收

- 新环境可以根据 README 构建和运行；
- baseline、优化和量化路径可以公平比较；
- 结论包含失败案例和适用边界；
- 项目能在 10 分钟内讲清问题、设计、证据和下一步。

## 阶段8

目标：把项目能力转化为可投递、可面试的证据。

### 必做

- 清理仓库入口、构建命令和依赖；
- 固定一组可复现 benchmark；
- 完成架构图、性能表和关键 profiler 截图；
- 为主项目准备 3 分钟和 10 分钟讲稿；
- 复习 C++、CUDA、操作系统和并行计算高频问题；
- 从真实岗位描述反查缺口；
- 小批量投递并根据反馈迭代。

### 验收

面对“为什么更快”“为什么这样管理内存”“量化损失在哪里”“换一张 GPU 是否仍成立”等问题，可以用项目数据回答，而不是只复述概念。

# 实习后

实习后不再按固定顺序通关全部内容，而是根据工作任务选择一条主线，其余路线作为补充。

## 训推系统

关注模型从加载到稳定服务的完整执行过程：

- Transformer 与 MoE 的执行结构；
- KV Cache、Paged Attention 和 Continuous Batching；
- TTFT、TPOT、吞吐、P99 和显存；
- vLLM、SGLang 等系统的调度与扩展机制；
- 训练 Runtime、算子融合、activation checkpoint 和混合精度；
- 性能、稳定性和资源利用率的联合优化。

建议产出：修改真实推理或训练模块，并通过压测、profiling 和故障案例证明收益。

## 分布式

关注多卡、多机环境中的通信、调度和可靠性：

- NCCL collective 与拓扑；
- DP、TP、PP、EP 和 ZeRO；
- 通信与计算重叠；
- 参数、梯度、激活和 KV Cache 的切分；
- 调度、限流、监控、容错和故障恢复；
- NUMA、PCIe、NVLink 与网络对性能的影响。

建议产出：一个可观测的多 GPU 训推实验，包含扩展效率、通信占比和故障注入报告。

## 编译系统

关注从模型图到硬件代码的自动变换：

- FX、TorchDynamo、AOTAutograd 和 TorchInductor；
- Triton 调度与代码生成；
- AST、IR、SSA、Pass 和数据流分析；
- LLVM IR、优化与后端；
- MLIR Dialect、Rewrite、Lowering；
- shape、layout、fusion 和 memory planning。

建议产出：一个小型张量编译器或 MLIR lowering 项目，保存变换前后 IR、等价性测试和性能结果。

## 硬件协同

关注算子、编译器与芯片之间的接口：

- GPU SM、scheduler、register file 和内存层次；
- Tensor Core、MMA 指令与数据布局；
- Cache、TLB、PCIe、NVLink 和 HBM；
- ISA、SASS/PTX 和编译结果分析；
- Roofline 与更细粒度的性能模型；
- 面向硬件约束的 kernel、量化和编译策略。

建议产出：选取一个算子，从模型语义、IR、PTX/SASS、硬件计数器到端到端性能完成纵向分析。

## 长期能力

长期目标不是把所有工具都学一遍，而是能沿下面的链路定位问题：

```text
模型结构
→ 计算图
→ 算子与量化
→ CUDA/Triton
→ Runtime 与调度
→ 分布式训推
→ 编译器
→ GPU 体系结构
```

当问题发生在任意一层时，能判断它是否真正源于该层，能向相邻层追踪，并用代码与数据验证判断。

# 执行方法

## 每周节奏

建议每周只围绕一个可验证问题推进：

```text
提出问题
→ 阅读最少资料
→ 写 baseline
→ 加正确性测试
→ 测量和 profiling
→ 解释结果
→ 提交代码和报告
```

推荐时间分配：

| 工作 | 比例 |
| --- | ---: |
| 编码与调试 | 45% |
| 测试与测量 | 20% |
| 阅读资料 | 20% |
| 报告与复盘 | 15% |

## 停止规则

以下情况应停止扩展范围：

- 当前实验还没有正确性测试；
- benchmark 没有 warmup 或同步；
- 同时维护超过一个主项目；
- 只增加新算子，不分析已有结果；
- 为了完整而学习当前项目用不到的框架；
- 无法用自己的话说明本周获得的证据。

## 调整规则

每四周复盘一次：

1. 本月新增了哪些可运行证据？
2. 哪个知识点实际阻塞了项目？
3. 哪些内容只是因为“以后可能有用”而加入？
4. 下一月能否删除至少一个低优先任务？
5. 当前项目是否更接近可投递状态？

岗位描述用于校准方向，不用于无限增加关键词。只有重复出现在目标岗位、并且阻塞当前项目的能力，才提升为近期主线。

## 当前行动

当前只推进阶段 1：

1. 学习 [`unique_ptr`、`shared_ptr`、`weak_ptr`](learning/stage-01/lessons/02-smart-pointer-ownership/README.md) 并运行专题示例；
2. 解释独占所有权转移、强引用计数和弱引用过期检查；
3. 继续学习 STL、模板与 C++17、错误处理、ABI 和构建测试；
4. 用 RAII 和移动语义实现一个 GPU 资源模型；
5. 通过 CTest、ASan 和 UBSan；
6. 达到阶段验收线后进入 GPU 并行基础，不扩展 RPC 或复杂协程。

## 资料入口

- [项目 README](README.md)
- [学习入口](learning/README.md)
- [部署指南](DEPLOYMENT_GUIDE.md)
- [CUDA 文档](https://docs.nvidia.com/cuda/)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)
- [PyTorch 扩展](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [Triton 教程](https://triton-lang.org/main/getting-started/tutorials/)
- [LLVM 教程](https://llvm.org/docs/tutorial/)
- [MLIR 教程](https://mlir.llvm.org/docs/Tutorials/Toy/)
