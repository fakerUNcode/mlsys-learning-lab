# Infra学习指南

> 适用对象：已经具备 Python、深度学习基础、C/Linux 基础、后端开发经验，并开始学习 CUDA/Triton，希望进入推理系统、AI 性能工程、Runtime 或 AI 编译器方向的学习者。
>
> 编写时间：2026-09-08
>
> 总体目标：在 9～15 个月内形成一条可验证的能力链：
>
> **模型/框架 → 计算图 → 算子 → CUDA/Triton → Runtime/调度 → 推理服务 → Profiling/性能优化 → LLVM/MLIR 编译**

推荐主线为：

> **C++/Linux 工程 → CUDA 性能闭环 → PyTorch 扩展 → Transformer 推理 → 分布式/服务化 → 编译原理 → LLVM/MLIR → 综合项目**

强化学习、GAN、传统 CNN 细节、NAS 等内容暂时放在支线。只保留能够帮助你理解模型推理、量化、算子和系统负载的部分。

##  学习与验收总原则

每个阶段都必须留下四类证据：

1. **代码证据**：可构建、可运行、可测试的仓库；
2. **数据证据**：Benchmark、Profiling、资源占用、正确率；
3. **解释证据**：一篇说明瓶颈、设计取舍和失败实验的技术文档；
4. **复现证据**：固定环境、启动命令、版本、硬件型号和结果。

不要用“看完视频/读完源码”作为完成标准。完成标准应是：

> 能够独立实现一个简化版本，能解释它为什么这样工作，能用工具证明瓶颈，并能在限制条件变化时做出合理修改。

所有性能数据都必须注明：GPU 型号、驱动/CUDA 版本、编译选项、输入规模、数据类型、预热次数、重复次数和统计方式。不同 GPU 的绝对数值不能直接比较，应优先比较同一环境下的相对提升。

## 阶段总览

| 阶段 | 建议用时 | 核心产出 | 通过标志 |
|---|---:|---|---|
| 0. 环境与基线 | 1 周 | 可复现实验环境和性能基线 | 能独立定位环境/版本问题 |
| 1. C++ 与 Linux Runtime | 6～8 周 | 线程池、任务队列、简易 RPC/执行器 | 能定位内存、线程、锁和 IO 问题 |
| 2. 体系结构与并行性能 | 3～4 周 | CPU/GPU microbenchmark | 能判断 compute-bound/memory-bound |
| 3. CUDA 性能工程 | 8～10 周 | GEMM、Reduce、Softmax、LayerNorm、Attention | 能用 Nsight 解释优化收益 |
| 4. PyTorch/Triton 算子集成 | 4～6 周 | C++/CUDA Extension + Triton kernel | 正确性、autograd/编译兼容和性能均达标 |
| 5. LLM 推理系统 | 8～10 周 | vLLM/SGLang 压测与改造项目 | 能分析 TTFT、TPOT、吞吐、P99 和 KV Cache |
| 6. AI Infra 与分布式 | 6～8 周 | 服务化、监控、限流、故障恢复 | 能稳定运行并解释资源/故障行为 |
| 7. 编译原理与 LLVM | 6～8 周 | 小型表达式/张量编译器 | 能读写 AST、IR、SSA 和 Pass |
| 8. MLIR/图编译 | 8～12 周 | MLIR Dialect/Pass/Lowering 项目 | 能完成一个端到端 lowering |
| 9. 综合作品集 | 4～6 周 | 端到端推理优化项目 | 可作为简历和面试主项目 |

##  阶段 0：环境、基线与工作方法

### 需要学习

- Linux/WSL 下的 CUDA、编译器、CMake、Python 虚拟环境；
- Git 分支、tag、子模块和可复现构建；
- Docker 基础；
- `nvidia-smi`、CUDA Toolkit、PyTorch CUDA 版本的关系；
- Benchmark 的预热、同步、重复运行和统计方法。

### 必做事情

建立一个统一仓库，例如 `mlsys-learning-lab`，目录至少包括：

```text
benchmarks/
cuda_kernels/
pytorch_extensions/
inference/
compiler/
reports/
scripts/
environment.yml or pyproject.toml
README.md
```

完成一个环境自检脚本，输出：

- GPU 名称、Compute Capability、显存；
- 驱动和 CUDA Runtime 版本；
- PyTorch 版本、CUDA 是否可用；
- 编译器、CMake、Ninja 版本；
- 当前 Git commit 和运行参数。

### 注意事项

- 不要只记录“CUDA 版本”，还要记录 driver、PyTorch wheel 和 GPU 架构；
- kernel 计时时必须使用 CUDA Event 或 Nsight，不要直接用 CPU wall clock 包住异步调用；
- 所有实验至少预热 10～50 次，再重复 100～1000 次；
- 对延迟同时报告 mean、median、P95/P99，避免只报告一个最好成绩。

### 达标线

- 新机器或新环境能在 30 分钟内完成安装；
- 一条命令能跑通正确性测试和 benchmark；
- 任何结果都能追溯到 commit、硬件、版本和参数；
- 能解释 CPU 计时、CUDA Event 计时和 profiler 结果为什么不同。

## 阶段 1：C++ 与 Linux Runtime

###  C++ 知识清单

- RAII、对象生命周期、拷贝/移动构造；
- `unique_ptr`、`shared_ptr`、`weak_ptr` 的使用边界；
- STL 容器、迭代器、算法和 allocator 基础；
- 模板、泛型编程和类型 traits；
- C++17：`optional`、`variant`、`string_view`、结构化绑定；
- C++20：concept、协程只需初步了解；
- exception 与 error code 的取舍；
- ABI、动态库、符号和链接；
- CMake、单元测试、Sanitizer。

###  Linux 与并发知识清单

- 进程、线程、虚拟内存、页表、mmap；
- 文件描述符、阻塞/非阻塞 IO、epoll；
- mutex、condition variable、semaphore、spinlock、atomic；
- false sharing、cache line、锁竞争；
- TCP、HTTP、RPC 和超时；
- `gdb`、`strace`、`perf`、AddressSanitizer、ThreadSanitizer。

###  必做项目：C++ 异步执行器

实现一个可复用的异步执行器，功能按顺序增加：

1. 固定大小线程池；
2. 有界任务队列；
3. `submit()` 返回 Future；
4. 优雅停止和任务取消；
5. 超时、重试和任务优先级；
6. 指标：队列长度、活跃线程、成功/失败数、任务等待时间；
7. 一个简单的 HTTP/RPC 入口；
8. 用 C++ 执行矩阵乘或模拟推理任务。

### 注意事项

- 队列满时必须定义行为：阻塞、拒绝或丢弃，不能静默覆盖；
- 停止流程必须考虑生产者、消费者和正在执行的任务；
- 不要用 `detach()` 逃避生命周期管理；
- 记录任务等待时间与执行时间，二者是不同瓶颈；
- 用 TSan 检查数据竞争，用 ASan 检查越界和 use-after-free；
- 用 `perf` 验证锁竞争，而不是凭感觉优化。

### 达标线

- 单元测试覆盖正常、满队列、超时、停止、异常任务等情况；
- 通过 ASan、TSan；
- 能解释线程数增加后吞吐为何不一定增加；
- 能用 perf 找到至少一个真实瓶颈并完成优化；
- 代码具备清晰的 ownership、错误处理和 CMake 构建方式。

## 阶段 2：计算机体系结构与并行性能

### 需要学习

- Cache line、L1/L2/L3、TLB、预取；
- SIMD、AVX2/AVX-512 的基本思想；
- 分支预测和分支失效；
- 内存带宽、访问延迟、NUMA；
- GPU 的 SM、Warp、寄存器、共享内存、全局内存；
- SIMT 与 CPU SIMD 的区别；
- Roofline Model 和算术强度；
- 数据并行、流水线并行、模型并行。

### 必做项目：CPU/GPU Microbenchmark

实现以下对比：

- 连续访问和跨步访问；
- AoS 与 SoA；
- 分支密集和无分支代码；
- CPU 单线程、多线程和 SIMD；
- GPU naive kernel 与 tiled kernel；
- 不同矩阵规模下的算术强度。

每组实验都要画出：数据规模—延迟、数据规模—带宽或吞吐的关系。

### 达标线

- 能根据实验判断程序主要受计算、内存带宽、延迟、同步还是启动开销限制；
- 能用 Roofline 解释优化方向；
- 能解释为什么增加线程后可能变慢；
- 能解释 CPU cache miss、GPU memory transaction、occupancy 之间的关系。

##  阶段 3：CUDA 性能工程

这是你当前最值得集中投入的阶段。你已有 CUDA 笔记，但需要将空白和薄弱部分补全。

### 必须补齐的 CUDA 基础

- CUDA API 错误检查；
- kernel 异步执行和显式同步；
- Grid-stride loop；
- 三维索引和边界处理；
- Stream、Event、异步 memcpy；
- Unified Memory 的适用边界；
- 原子操作；
- Register、spill、local memory；
- Constant memory 和只读数据路径。

###  必须掌握的性能主题

- Coalesced memory access；
- Shared memory 与 bank conflict；
- Warp divergence；
- Warp shuffle、vote、match 指令；
- Occupancy；
- Kernel launch overhead；
- Double buffering 和计算/传输重叠；
- FP16、BF16、混合精度；
- Tensor Core/WMMA 的基本使用；
- cuBLAS/cuDNN 与手写 kernel 的边界。

### 必做项目 A：高质量 Reduction

从 naive 版本开始，依次实现：

1. 全局内存归约；
2. 共享内存归约；
3. 交错寻址版本；
4. 无发散版本；
5. warp shuffle 版本；
6. 多 block、多 kernel 级联版本；
7. 与 PyTorch `sum`、CUB 或其他库比较。

记录：正确性、不同输入规模、不同 block size、吞吐、占用率、warp stall 和内存访问指标。

### 必做项目 B：GEMM 优化阶梯

实现并比较：

```text
CPU naive
→ CUDA naive
→ shared-memory tiling
→ register tiling
→ vectorized load
→ double buffering
→ cuBLAS
→ optional Tensor Core
```

至少覆盖方阵和非方阵、边界不能整除 tile 的情况。

### 必做项目 C：Softmax、LayerNorm、Fused Attention

顺序建议：

1. Row-wise Softmax；
2. LayerNorm/RMSNorm；
3. 融合 bias + activation；
4. 简化版 scaled dot-product attention；
5. 处理变长输入或 mask。

重点观察：中间张量是否写回显存、数值稳定性、寄存器压力、shared memory 使用和融合收益。

### Profiling 工具路线

- `cuda-memcheck`/Compute Sanitizer：先查正确性；
- Nsight Systems：看 CPU/GPU 时间线、Stream、kernel 启动和同步；
- Nsight Compute：看单 kernel 的 SOL、访存、warp stall、occupancy、source correlation；
- PyTorch Profiler：看 Python/算子/设备之间的整体关系。

Nsight Compute 官方文档提供 profiling guide、CLI、UI、报告比较和 Python 报告接口，可作为主要工具手册：[Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)。

### 注意事项

- 不要先优化再测量；
- 不要把 occupancy 当作越高越好，最终目标是有效吞吐/延迟；
- 不要只看 kernel duration，要看端到端时间和数据搬运；
- 不要用不稳定的随机输入掩盖 NaN、Inf 和边界错误；
- FP16/BF16 优化必须与 FP32 参考实现比较误差；
- 任何“提升 X 倍”都要说明 baseline、输入规模、数据类型和测量方法。

### 达标线

- 能从零写出带错误检查、边界处理和测试的 CUDA kernel；
- Reduction、GEMM、Softmax、LayerNorm 至少各有一个可复现实现；
- 对一个 kernel 完成两轮以上 profiling 驱动优化；
- 同一 GPU 上至少有一个算子相对 baseline 提升 1.5 倍以上，或明确证明已经接近库函数/硬件上限；
- 能写出一份 2～5 页的性能报告，解释每个优化为什么有效或无效；
- 正确性误差阈值、性能结果和失败实验均有记录。

## 阶段 4：PyTorch、C++/CUDA Extension 与 Triton

### 需要学习

- Tensor、Storage、Stride、contiguous；
- dispatcher、operator registration、device dispatch；
- autograd、fake/meta kernel、`torch.compile` 兼容性；
- C++ Extension 的编译、加载和 ABI；
- Triton 的 program ID、block pointer、mask、reduction、autotune；
- Python baseline、Triton kernel、CUDA kernel、库函数四者的比较方法。

PyTorch 官方扩展教程目前包含自定义 C++/CUDA Operator、dispatcher 和 `torch.library` 路线，建议优先按官方推荐接口复现，而不是从旧式绑定方式开始：[PyTorch Extending](https://docs.pytorch.org/tutorials/extension.html) 和 [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)。

Triton 官方教程建议按 Vector Addition、Fused Softmax、Matrix Multiplication、Layer Normalization、Fused Attention 的顺序学习：[Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)。

### 必做项目：同一算子的三种实现

选择 LayerNorm 或 Softmax，实现：

1. PyTorch 参考版本；
2. Triton 版本；
3. C++/CUDA Extension 版本。

加入：

- CPU/GPU 正确性测试；
- 随机输入和极端输入；
- dtype 测试；
- `torch.library.opcheck` 或等价检查；
- autograd 检查；
- 不同 shape 的 benchmark；
- `torch.compile` 兼容性测试；
- 失败 shape 的明确报错。

### 达标线

- 能独立完成 PyTorch 注册、编译、加载和测试；
- 自定义算子结果与参考实现误差在预先定义的范围内；
- 能解释 stride、layout 和 contiguous 对 kernel 的影响；
- 至少一个 Triton kernel 在目标 shape 上达到 PyTorch baseline 的 0.8～1.2 倍以上，并能解释未达到更高性能的原因；
- 能区分 Python 调度开销、kernel 开销、内存访问和编译开销。

## 阶段 5：Transformer 与 LLM 推理系统

###  必须掌握的模型执行知识

- Transformer decoder 单层计算顺序；
- Prefill 与 Decode 的计算差异；
- MHA、MQA、GQA；
- KV Cache 的 shape、布局、增长和回收；
- Attention mask、causal mask、sliding window；
- Sampling、停止条件和 streaming；
- FP16/BF16/INT8/FP8 的基本取舍；
- weight-only、activation、KV Cache 量化的区别。

### 必须掌握的系统知识

- Continuous Batching；
- Paged KV Cache；
- Prefix Cache；
- Chunked Prefill；
- Admission control 和限流；
- 请求队列、调度、超时和取消；
- tensor parallel、pipeline parallel 的基本通信路径；
- TTFT、TPOT、throughput、concurrency、P50/P95/P99。

###  必做项目 A：推理基准平台

使用 vLLM 或 SGLang 部署一个可运行模型，编写 benchmark 客户端，控制：

- 输入 token 长度；
- 输出 token 长度；
- 并发数；
- 请求到达速率；
- streaming 开关；
- batch 或 cache 配置；
- 数据类型和量化配置。

输出：

- TTFT；
- TPOT/ITL；
- 总吞吐和 output token 吞吐；
- P50/P95/P99；
- GPU 利用率；
- 显存和 KV Cache 使用；
- 错误率、超时率和队列等待时间。

### 必做项目 B：简化版 Paged KV Cache

不要一开始直接修改大型推理框架。先用 Python 或 C++ 实现一个教学版：

- 固定大小 KV block；
- block table；
- 请求到 block 的映射；
- 分配、释放和复用；
- 不同长度请求的碎片率比较；
- contiguous cache 与 paged cache 的对比。

然后阅读 vLLM 的实现，把教学版概念映射到真实系统。

PagedAttention 的原始论文适合用来理解“分页式 KV Cache”的设计动机：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)。

### 必做项目 C：修改一个真实推理模块

从以下内容中选一个：

- scheduler 指标；
- KV Cache 统计；
- 请求限流；
- prefix cache 实验；
- chunked prefill 实验；
- attention backend 对比；
- benchmark/metrics 改进。

要求先建立 baseline，再修改，再回归测试和压测。不要只提交“能跑”的代码，要说明在什么 workload 下收益、在哪些场景回退。

### 注意事项

- TTFT 和 TPOT 不能混成一个 latency；
- offline benchmark 和 online serving benchmark 结论不同；
- 不同 prompt 分布会显著改变结论；
- 不能只测满载吞吐，要测低并发延迟和突发流量；
- 修改框架时固定 commit，避免 upstream 变化导致无法复现。

### 达标线

- 能解释 Prefill/Decode 的瓶颈差异；
- 能根据 KV Cache 估算并发上限；
- 能独立完成一组可重复的压测报告；
- 能定位一次 P99 延迟上升的原因；
- 至少完成一个真实推理框架模块的修改或实验；
- 具备一个可公开展示的推理优化项目。

## 阶段 6：AI Infra 与分布式系统

### 需要学习

- Docker 镜像、GPU runtime、健康检查；
- Kubernetes Pod、Deployment、Service、ConfigMap、Secret；
- GPU 资源请求、节点标签、taint/toleration；
- Prometheus 指标、Grafana、日志和 tracing；
- 限流、排队、重试、熔断、超时、幂等；
- 模型版本、灰度、回滚；
- NCCL、NVLink、PCIe、RDMA 的基本关系；
- data/tensor/pipeline parallel 的基本通信模式。

### 必做项目：单机多实例推理平台原型

实现：

- 模型注册和版本信息；
- 启动多个推理实例；
- GPU/端口分配；
- 请求路由和限流；
- readiness/liveness；
- Prometheus 指标；
- 失败重试和超时；
- 灰度发布和回滚；
- 压测脚本和故障注入。

初期可以在 Docker Compose 或单机 Kubernetes 上完成，不必一开始搭建复杂云集群。

### 故障注入清单

- 模型加载失败；
- GPU 显存不足；
- 一个实例无响应；
- 请求超时；
- 队列持续增长；
- GPU 温度或利用率异常；
- 指标服务不可用。

### 达标线

- 能从日志和指标定位请求失败原因；
- 故障实例不会继续接收新请求；
- 重试不会造成无限放大；
- 有明确的 P99、错误率和队列长度告警；
- 能解释多实例吞吐提升受限于什么；
- 至少完成一次灰度和一次回滚演练。

##  阶段 7：编译原理与 LLVM

### 需要学习

- Lexer、Parser、AST；
- 符号表和类型检查；
- CFG、基本块、SSA；
- 常量折叠、死代码消除、公共子表达式；
- 数据流分析；
- IR 设计；
- 指令选择、寄存器分配、代码生成的基本概念；
- LLVM IR、Module、Function、BasicBlock、Value、Pass。

### 必做项目：张量表达式编译器

支持以下表达式：

```text
C = A + B
D = relu(C)
E = matmul(D, W)
```

分阶段实现：

1. 词法和语法；
2. AST；
3. shape/type checking；
4. 简单 tensor IR；
5. 常量折叠；
6. elementwise fusion；
7. 打印 IR；
8. 输出 C++/CUDA 伪代码或 LLVM IR。

LLVM 官方 Kaleidoscope 教程适合作为前端、AST、IR 和代码生成的复现路线，但必须使用与你安装的 LLVM 版本匹配的教程版本：[LLVM Kaleidoscope](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl03.html)。

### 达标线

- 能解释 AST、CFG、SSA 的用途和关系；
- 能实现至少两个优化 Pass；
- 能展示优化前后的 IR 差异；
- 能把一个简单表达式 lower 到 LLVM IR 或可执行代码；
- 能解释为什么图优化和 kernel 优化不是同一个层次的问题。

##  阶段 8：MLIR、图优化与 AI 编译器

### 需要学习

- MLIR Context、Module、Operation、Region、Block；
- Dialect、Operation、Type、Attribute；
- ODS/TableGen；
- Rewrite Pattern、Canonicalization；
- Pass 管理；
- shape inference；
- tensor/memref；
- affine/linalg；
- bufferization；
- tiling、fusion、vectorization；
- LLVM lowering；
- layout 和 memory space。

### 官方复现路线

按 MLIR Toy Tutorial 的顺序完成：

1. AST 和 Toy 语言；
2. 生成基础 MLIR；
3. 写高层 rewrite；
4. 用 interface 做通用变换；
5. lower 到 affine/linalg；
6. lower 到 LLVM；
7. 增加一种自定义类型。

官方教程明确覆盖了 Dialect、Rewrite Pattern、Interface、部分 lowering 和 LLVM code generation，适合作为第一条可复现路线：[MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)。

### 必做项目：Elementwise 图编译器

实现一个小型模型图编译流程：

```text
JSON/ONNX-like graph
→ Graph IR
→ shape inference
→ constant folding
→ elementwise fusion
→ layout decision
→ bufferization
→ linalg/affine
→ LLVM 或 CUDA 目标
```

第一版只支持：Add、Mul、Relu、Matmul、Transpose。重点不是支持很多算子，而是把完整 pipeline 跑通。

### 注意事项

- 不要一开始就阅读整个 MLIR 源码；先跑通 Toy Tutorial；
- 每个 Pass 都要有输入 IR、输出 IR 和测试；
- 把合法性约束写成 verifier 或测试，不要只依赖人工检查；
- 明确区分 tensor-level、buffer-level 和 hardware-level 优化；
- Fusion 可能增加寄存器压力，不能假设融合总是更快；
- Tiling、layout、vectorization 必须结合目标硬件验证。

### 达标线

- 能定义一个简单 Dialect 或扩展一个已有 Dialect；
- 能写一个 Pattern Rewrite 和一个 Pass；
- 能完成至少一次 tensor→memref 或 linalg→LLVM 的 lowering；
- 能对一个图做融合并验证数值一致性；
- 能用 benchmark 证明优化对某些 shape 有益、对另一些 shape 可能有害。

## 阶段 9：综合作品集项目

最终建议形成三个互相连接的项目，而不是很多零散 demo。

### 项目一：Kernel Lab

包含：

- Reduction；
- GEMM；
- Softmax；
- LayerNorm/RMSNorm；
- Attention；
- CUDA 与 Triton 双版本；
- correctness、benchmark、Nsight 报告。

验收：至少一个真实模型相关算子在固定 shape 上有可解释的性能提升。

### 项目二：Inference Lab

包含：

- vLLM/SGLang 部署；
- benchmark 客户端；
- TTFT/TPOT/P99；
- KV Cache 实验；
- 调度或 cache 模块改造；
- Prometheus/Grafana 指标；
- 故障和回滚演练。

验收：能从端到端压测结果定位到 kernel、显存、调度或服务层的瓶颈。

### 项目三：Mini Compiler

包含：

- Graph IR；
- shape inference；
- fusion pass；
- layout 或 tiling pass；
- MLIR/LLVM lowering；
- 至少一个目标 kernel；
- 数值和性能回归测试。

验收：输入一个小型计算图，自动输出优化后的 IR 和可执行目标，并能展示优化前后的差异。

## 每周执行模板

建议每周投入 12～18 小时：

- 3 小时：阅读概念和官方文档；
- 6～8 小时：实现项目；
- 2～3 小时：测试、benchmark、profiling；
- 1～2 小时：整理实验报告和复盘；
- 1～2 小时：阅读真实项目源码。

每周必须完成一个可验证闭环：

```text
提出问题
→ 写 baseline
→ 设计实验
→ 测量
→ 修改
→ 复测
→ 解释结果
→ 记录失败实验
```

推荐每周记录：

- 本周新增的一个核心概念；
- 一个最小可运行实现；
- 一个性能数据表或时间线；
- 一个失败实验；
- 一个仍未解释的问题；
- 下周要验证的假设。

##  什么时候算真正达到目标

你可以认为自己具备“推理系统/性能工程初级到中级”的能力，当你能够独立完成以下任务：

1. 写一个 C++/CUDA/Triton 算子并处理异常、边界和测试；
2. 用 Nsight 或 PyTorch Profiler 找到瓶颈，而不是凭经验猜；
3. 解释 Transformer Prefill、Decode、KV Cache 和 Continuous Batching；
4. 部署模型并报告 TTFT、TPOT、吞吐和 P99；
5. 修改推理框架的一个模块并完成回归压测；
6. 实现一个简单计算图和优化 Pass；
7. 阅读并解释一段 PyTorch、vLLM、Triton 或 MLIR 的核心代码；
8. 面对性能下降时，能按“服务—调度—Runtime—kernel—硬件”逐层定位。

## 你现在的直接行动顺序

如果从今天开始执行，建议严格按下面顺序：

1. 补全 CUDA 错误检查、Coalesced Access、Warp Shuffle、Occupancy、Stream；
2. 完成 Reduction、GEMM、Softmax、LayerNorm 四个 benchmark；
3. 用 PyTorch C++/CUDA Extension 封装 LayerNorm 或 Softmax；
4. 用 Triton 重写同一算子并比较；
5. 实现 C++ 线程池和异步任务队列；
6. 部署 vLLM/SGLang，完成完整压测报告；
7. 实现教学版 Paged KV Cache；
8. 阅读并修改一个真实推理框架模块；
9. 学习 LLVM Kaleidoscope，完成张量表达式编译器；
10. 完成 MLIR Toy Tutorial，再做自己的 Elementwise 图编译器。

最重要的约束是：在 CUDA 性能闭环和推理系统项目完成之前，不要把主要精力转移到大规模 MLIR 源码阅读，也不要继续无边界扩展深度学习理论目录。

## 参考入口

- [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [PyTorch Extending Documentation](https://docs.pytorch.org/tutorials/extension.html)
- [PyTorch Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [Triton Official Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Triton GitHub](https://github.com/triton-lang/triton)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Documentation](https://docs.vllm.ai/)
- [LLVM Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/)
- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
