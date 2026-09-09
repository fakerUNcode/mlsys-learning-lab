# MLSys Learning Lab 自学指南

## 0. 这份指南解决什么问题

如果你刚开始接触 GPU，只知道“GPU 可以加速深度学习”，但不知道从哪里写第一行代码，这份指南就是你的项目入口。

你不需要一开始就理解 CUDA 的所有细节。你只需要按照“观察现象 → 写最小程序 → 检查正确性 → 测量速度 → 找到瓶颈 → 优化 → 写报告”的顺序推进。每完成一个阶段，就会多理解一层机器学习系统。

## 1. 这个项目用于什么

`mlsys-learning-lab` 是一个机器学习系统实验室。它用一个统一仓库把模型计算背后的不同层次连接起来：

```text
Python / PyTorch 模型
        ↓
算子与 Tensor（数据、形状、dtype、内存布局）
        ↓
CUDA kernel（GPU 上真正执行的程序）
        ↓
PyTorch 扩展与运行时
        ↓
编译器优化（图、IR、融合、代码生成）
        ↓
推理系统与性能 benchmark
        ↓
实验报告与可复现结论
```

换句话说，这不是一个单纯“训练模型”的项目，而是用来回答以下问题：

- 一段程序为什么能在 GPU 上运行？
- GPU 到底比 CPU 快在哪里？
- 一个 CUDA kernel 如何把工作分给成千上万个线程？
- 为什么同样的数学公式，不同写法速度差异很大？
- PyTorch 如何调用 C++/CUDA 代码？
- 编译器如何自动改写和优化计算图？
- 怎样用可信的方法证明一个版本真的更快？

## 2. 你能在这里干什么

每个目录代表一个学习方向：

| 目录 | 你会做什么 | 最终产物 |
| --- | --- | --- |
| `benchmarks/` | 学习如何公平测量速度、吞吐和显存 | 可复现 benchmark 与结果表 |
| `cuda_kernels/` | 编写 GPU kernel，理解线程、内存和同步 | baseline kernel、优化 kernel、正确性测试 |
| `pytorch_extensions/` | 把 C++/CUDA 代码接入 PyTorch | 可安装的 extension 与 Python API |
| `inference/` | 测量模型推理的延迟、吞吐和显存 | 推理 runner 与性能报告 |
| `compiler/` | 观察计算图/IR，尝试融合或编译优化 | IR 快照、优化前后对比 |
| `reports/` | 记录环境、方法、数据和结论 | 可以被别人复现的实验报告 |
| `tests/` | 防止优化破坏正确性 | 单元测试、数值测试、回归测试 |

## 3. 你需要先具备什么

### 必需的基础

- 会使用终端：进入目录、运行命令、查看文件。
- 知道变量、函数、循环、数组和基本 Python 语法。
- 理解二进制、内存、进程、编译和库这些基本概念。
- 会阅读报错，并能根据文件名和行号定位问题。
- 知道 Git 的基本用法：`status`、`diff`、`log`。

### 不要求一开始就会的内容

- 不要求你已经会 CUDA。
- 不要求你会写复杂的 C++ 模板。
- 不要求你理解深度学习全部数学。
- 不要求你一开始会使用 profiler 或编译器。

本项目会在需要时逐步引入这些知识。

## 4. 先建立三个最重要的概念

### 4.1 CPU 和 GPU 的区别

CPU 通常有较少但很强的核心，适合复杂控制逻辑和少量任务。GPU 有大量相对简单的计算核心，适合把同一种操作同时应用到大量数据。

例如向量加法：

```text
c[i] = a[i] + b[i]
```

CPU 可以依次或少量并行地计算多个 `i`；GPU 可以让大量线程分别负责不同的 `i`。当数据足够大、工作足够规则时，GPU 才可能体现优势。

### 4.2 Tensor 是什么

Tensor 可以先理解成带有额外信息的多维数组。除了数据本身，还要关注：

- `shape`：每一维有多长，例如 `(batch, channels, height, width)`。
- `dtype`：每个元素的类型，例如 `float32`、`float16`、`int32`。
- `device`：数据在 CPU 还是 GPU。
- `layout/stride`：多维索引如何映射到内存。

很多 GPU 错误和性能问题，实际上来自 shape、dtype、device 或内存布局不匹配。

### 4.3 正确不等于快速

第一目标是得到正确结果。第二目标才是加速。一个错误但很快的 kernel 没有价值；一个正确但没有测量依据的“优化”也不能证明成功。

每个实验都必须有：

```text
reference（可信结果）
→ candidate（待验证实现）
→ correctness test（结果比较）
→ benchmark（速度测量）
→ report（解释原因）
```

## 5. 项目使用方法：固定闭环

每次开始新实验，都按下面的闭环操作：

### 第一步：确认环境

```bash
cd ~/projects/mlsys-learning-lab
source .venv/bin/activate
./scripts/check_environment.sh --purpose=<experiment-name>
```

把输出保存到报告中。它告诉你本次实验使用了哪个 GPU、驱动、CUDA、PyTorch、编译器和 Git commit。

### 第二步：提出一个小问题

不要一开始做“优化整个大模型”。从一个可以验证的小问题开始，例如：

> 对两个长度为 N 的向量做加法，CUDA 实现是否比 CPU 实现快？N 多大时才值得使用 GPU？

好问题必须有明确输入、输出和比较方法。

### 第三步：写 reference

先用 NumPy 或 PyTorch 写最容易相信的版本。它不必最快，但必须清晰。

### 第四步：写最小 candidate

先让 GPU 版本运行起来，再考虑优化。第一次实现只需要覆盖一种 dtype、一个简单 shape 和一个设备。

### 第五步：检查正确性

比较 reference 和 candidate 的输出，使用明确的绝对误差或相对误差阈值。测试正常输入、空输入、小尺寸、非整除线程块尺寸和极值输入。

### 第六步：测量性能

GPU 工作通常是异步提交的。不能只用普通的 Python 开始/结束时间判断 kernel 用时；需要使用 CUDA event 或明确同步。测量前先 warmup，重复多次，并记录 p50/p95 或平均值与波动。

### 第七步：解释结果

问自己：瓶颈是计算、显存带宽、访存不合并、同步、线程分歧、kernel launch，还是数据搬运？没有解释的 speedup 只是一个数字。

### 第八步：写报告

使用 `reports/` 的模板思路记录：问题、假设、环境、命令、参数、结果、分析、局限和下一步。

## 6. 分阶段学习任务

### 阶段 0：环境和工具（半天到一天）

目标：能独立运行项目，并理解自检输出。

任务：

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
./scripts/check_environment.sh --purpose=stage-0
pytest -q
git status
```

详细的名词解释、实测输出、两道验收题及解析见 [reports/stage0_environment.md](reports/stage0_environment.md)。

阶段 0 还要阅读报告第 12 节，完成一次 benchmark 方法练习：明确 warmup、同步位置、重复次数、统计量，以及 kernel 时间和端到端时间的区别。

你需要能解释：Python 虚拟环境、PyTorch、CUDA Runtime、CUDA Toolkit、驱动、`nvcc`、CMake 和 Ninja 分别是什么。

验收：能说清楚“PyTorch 能否使用 GPU”和“系统是否能编译 CUDA 代码”是两个相关但不同的问题。

### 阶段 1：用 PyTorch 观察 CPU/GPU（一天）

目标：理解 Tensor 的 shape、dtype、device 和基本计时。

建议实验：向量加法、矩阵乘法、ReLU、归约求和。

需要完成：

- 同一计算分别在 CPU 和 GPU 上运行。
- 打印输入输出的 shape、dtype、device。
- 比较结果是否一致。
- 改变数据规模，观察 GPU 何时开始有优势。

验收：你能解释为什么小数据可能 GPU 更慢，以及为什么 GPU 计时需要同步。

### 阶段 2：第一个 CUDA kernel（两到三天）

目标：理解一个线程如何对应一个数据元素。

建议从 elementwise vector add 开始，然后尝试：

1. 向量加法。
2. ReLU。
3. 行级归约。
4. 简单矩阵转置。

每个实验都要有 CPU/PyTorch reference、CUDA candidate、边界测试和 benchmark。

必须理解的词：grid、block、thread、warp、global memory、shared memory、同步、kernel launch。

验收：能画出 `grid → block → thread` 的层次，并解释一个线程如何计算自己的数组下标。

### 阶段 3：性能分析与优化（两到四天）

目标：不靠猜测优化，而是根据数据定位瓶颈。

依次尝试：

- 改变 block size。
- 检查全局内存访问是否合并。
- 使用 shared memory。
- 减少不必要的同步和中间结果。
- 比较不同 dtype。
- 使用 profiler 观察 kernel 时间、显存吞吐和 occupancy。

验收：每次优化都能回答“改变了什么、为什么可能有效、数据是否支持结论”。

### 阶段 4：PyTorch C++/CUDA extension（两到四天）

目标：理解 Python 调用底层实现的边界。

需要完成一个简单算子，例如 custom ReLU 或 vector add：

- Python API。
- C++ binding。
- CPU fallback。
- CUDA 实现。
- device、dtype、shape 检查。
- 与 PyTorch reference 的数值测试。
- 可 editable install。

验收：另一个人只看安装说明，就能构建并调用你的 extension；错误输入会得到清楚的报错。

### 阶段 5：推理实验（两到四天）

目标：从“单个 kernel 很快”扩展到“整个请求很快”。

建立一个推理 baseline，测量：

- 首次运行时间。
- warmup 后的 steady-state latency。
- p50/p95/p99 latency。
- 不同 batch size 的吞吐。
- 峰值显存。
- CPU 到 GPU 的数据搬运时间。

再尝试一种优化，例如 `torch.compile`、半精度、算子融合或 batch 调整。

验收：能区分编译时间、模型加载时间、数据准备时间和实际 GPU 执行时间。

### 阶段 6：编译器和计算图（两到五天）

目标：理解框架如何把多条算子调用变成可优化的计算图。

需要观察：

- eager 模式执行了哪些算子。
- 图捕获后 IR 如何表示计算。
- 哪些算子可以融合。
- 动态 shape 或控制流为什么可能阻碍编译。
- 编译时间和运行时收益如何权衡。

验收：完成一个优化前后对比，并保存图/IR、命令、性能和正确性结果。

## 7. 你最终需要完成的项目清单

### 最低完成版

- [ ] 通过环境自检。
- [ ] 完成一个 PyTorch CPU/GPU 对比实验。
- [ ] 完成一个 CUDA kernel。
- [ ] 为 kernel 编写正确性测试。
- [ ] 完成一次有 warmup 和同步的 benchmark。
- [ ] 完成一个 PyTorch extension 或可运行的底层算子实验。
- [ ] 完成一个推理 latency/throughput 实验。
- [ ] 完成一个编译器或计算图实验。
- [ ] 为每个实验写报告。
- [ ] 用 Git 保存每个阶段的 commit。

### 进阶完成版

- [ ] 使用 profiler 找到并解释瓶颈。
- [ ] 支持多个 dtype、shape 和 batch size。
- [ ] 增加 CPU fallback 和错误输入测试。
- [ ] 增加自动化 benchmark 汇总。
- [ ] 增加 CI 或至少一条可重复的自动测试命令。
- [ ] 比较多个优化方案，而不是只比较一个版本。

## 8. 初学者最容易踩的坑

### 只看 GPU 利用率判断快慢

GPU 利用率高不等于程序高效；可能只是等待内存或运行了不必要的工作。要结合 kernel 时间、带宽、occupancy 和端到端延迟。

### 没有 warmup 就开始计时

首次运行可能包含 CUDA context 初始化、内存分配或编译时间。要把初始化、编译和稳定运行分开报告。

### 忘记 CUDA 异步执行

Python 代码返回不代表 GPU 已经完成。计时前后需要 CUDA event 或同步。

### 只测试一个输入

一个 shape 上有效的优化可能在其他 shape、dtype 或 batch 上变差。至少测试小、中、大三种规模和边界尺寸。

### 一开始就优化复杂模型

复杂模型的问题太多，难以判断收益来自哪里。先做向量加法、归约、矩阵乘法等最小实验。

### 看到报错就盲目改代码

先记录完整命令、错误类型、设备、dtype、shape、commit 和环境自检输出，再定位问题。

## 9. 每天学习时的建议流程

```text
阅读一个概念
→ 写 20~50 行最小代码
→ 用一个小输入验证
→ 打印 shape/device/dtype
→ 写一个反例或边界测试
→ 测量并记录结果
→ 用自己的话写三句话总结
```

每天不要同时引入多个变量。例如先固定 dtype，只改变 batch；先固定 shape，只改变 block size。这样你才能知道结果为什么变化。

## 10. 完成标准

当你能做到下面这些事情，就说明已经完成了本项目的入门目标：

1. 能解释一个 Tensor 从 Python 到 GPU kernel 的大致路径。
2. 能写出一个简单 CUDA kernel，并让它通过数值测试。
3. 能正确测量 GPU kernel，而不是测到异步提交时间。
4. 能根据 profiler 或 benchmark 数据提出优化假设。
5. 能把底层实现封装成 PyTorch 可调用的接口。
6. 能区分 kernel 性能、模型推理性能和端到端服务性能。
7. 能用一份报告让别人复现实验并理解你的结论。

这就是从 GPU 小白进入机器学习系统工程的第一条完整路径。
