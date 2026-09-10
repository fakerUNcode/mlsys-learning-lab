# MLSys Learning Lab

一个以证据驱动方式学习机器学习系统的实验仓库，覆盖 C++/Linux Runtime、CUDA kernel、PyTorch 扩展、推理系统、性能基准与 AI 编译器。

> 项目正在持续建设。目前处于阶段 1：C++ 与 Linux Runtime。目录存在不代表对应实现已经完成，请以各模块的“当前状态”为准。

## 项目目标

这里不只收集代码或笔记。每个实验都应形成一条可复现的证据链：

```text
问题与假设 → baseline → 正确性测试 → 性能测量 → profiler 证据 → 结论与复盘
```

长期主线为 C++/Linux → CUDA → PyTorch/Triton → LLM 推理 → AI Infra → LLVM/MLIR。完整规划见 [Infra 学习指南](Infra%20Introduction.md)，当前讲义从 [learning/README.md](learning/README.md) 进入。

## 当前状态

| 模块 | 状态 | 当前内容 |
| --- | --- | --- |
| 学习材料 | 进行中 | 阶段 0 索引、阶段 1 C++ 讲义与示例 |
| 环境检查 | 可用 | GPU、CUDA、PyTorch、编译器和 Git 快照 |
| Benchmark | 初版可用 | PyTorch CPU/GPU 向量加法基准 |
| CUDA kernel | 规划中 | 文档骨架与既有 CUDA 专题笔记 |
| PyTorch 扩展 | 规划中 | 接口和验证约定 |
| 推理系统 | 规划中 | 指标与目录约定 |
| 编译器实验 | 规划中 | 图、IR 和 Pass 实验约定 |

## 快速开始

要求 Python 3.10+。GPU 实验还需要兼容的 NVIDIA 驱动；只有编译 CUDA 源码时才要求完整 CUDA Toolkit。

```bash
git clone git@github.com:fakerUNcode/mlsys-learning-lab.git
cd mlsys-learning-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[benchmark]'
bash scripts/check_environment.sh --purpose=bootstrap
python benchmarks/benchmark_vector_add.py \
  --device cpu --n 1024 --warmup 1 --iters 2
```

PyTorch 相关依赖可用 `python -m pip install -e '.[torch,benchmark]'` 安装。CUDA wheel 需要按本机环境选择，详见[部署指南](DEPLOYMENT_GUIDE.md)。

## 首个实验

无 GPU 时先运行 CPU 路径：

```bash
python benchmarks/benchmark_vector_add.py \
  --device cpu --n 1048576 --warmup 5 --iters 20
```

有可用 NVIDIA GPU 和 CUDA 版 PyTorch 时：

```bash
python benchmarks/benchmark_vector_add.py \
  --device cuda --n 16777216 --warmup 20 --iters 100
```

## 项目结构

| 路径 | 内容 | 文档 |
| --- | --- | --- |
| `learning/` | 分阶段讲义、实例与练习 | [学习入口](learning/README.md) |
| `benchmarks/` | 计时、输入矩阵和结果规范 | [基准测试](benchmarks/README.md) |
| `cuda_kernels/` | CUDA kernel 与性能实验 | [CUDA 算子](cuda_kernels/README.md) |
| `pytorch_extensions/` | C++/CUDA extension | [PyTorch 扩展](pytorch_extensions/README.md) |
| `inference/` | 模型执行与服务性能 | [推理实验](inference/README.md) |
| `compiler/` | 图、IR、Pass 与代码生成 | [编译器实验](compiler/README.md) |
| `tests/` | 跨模块回归测试 | [测试约定](tests/README.md) |
| `reports/` | 环境、数据与实验结论 | [报告规范](reports/README.md) |
| `scripts/` | 环境检查与自动化入口 | [脚本说明](scripts/README.md) |
| `Operator-notes/` | 既有 CUDA 专题笔记 | [阶段 0 索引](learning/stage-00/before-learning/README.md) |

## 实验规范

- 先验证正确性，再讨论性能。
- GPU 计时必须处理异步执行，区分 device time 与端到端时间。
- 测量前 warmup，多次采样，并报告统计量而非单次最好结果。
- 记录输入、随机种子、硬件、驱动、依赖、编译选项和 Git commit。
- 优化同时保留 baseline，并说明收益、代价与适用边界。
- 不提交大型权重、构建产物和原始 profiler 文件；报告中保留复现方法。

## 参与方式

欢迎通过 Issue 提交可复现的问题、实验建议或资料纠错。代码贡献应保持单一主题，并包含背景、根目录运行命令、正确性测试；性能改动还需包含 baseline、环境和原始指标。

提交前至少运行：

```bash
git diff --check
python benchmarks/benchmark_vector_add.py \
  --device cpu --n 1024 --warmup 1 --iters 2
```

Python 测试加入后，统一入口为 `pytest -q`；当前仓库尚未收集到 Python 测试。

涉及阶段 1 C++ 示例时，再运行：

```bash
cmake -S learning/stage-01/examples \
  -B /tmp/mlsys-stage1-build -DENABLE_SANITIZERS=ON
cmake --build /tmp/mlsys-stage1-build
ctest --test-dir /tmp/mlsys-stage1-build --output-on-failure
```

## 路线文档

- [Infra 学习指南](Infra%20Introduction.md)：阶段路线与验收标准。
- [学习入口](learning/README.md)：当前阶段化材料。
- [部署指南](DEPLOYMENT_GUIDE.md)：安装、运行与排障。
- [学习报告](LEARNING_REPORT.md)：已有知识总结。

## 许可状态

仓库目前尚未添加开源许可证。在许可证明确前，源码可公开阅读，但复用、分发和衍生使用不应被默认视为已授权。
