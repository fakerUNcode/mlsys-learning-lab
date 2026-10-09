# MLSys Learning Lab

一个面向机器学习系统初学者的可运行学习仓库。材料从张量与 C++ 基础延伸到 CUDA、PyTorch 扩展、推理系统、编译器和性能测量；每个实验尽量同时提供实现、正确性证据、运行方法和适用边界。

仓库的重点是把“写出程序”连到“解释程序为何正确、性能受什么限制”。目前内容以教程和小型验证实验为主，不是生产级推理框架，也不提供普遍适用的性能保证。

## 适合谁

- 正在学习 PyTorch、C++、CUDA 或 ML Systems 的开发者。
- 希望通过小程序理解张量布局、GPU 执行、内存和推理性能的人。
- 需要可复现的基准与报告范例的实验者。

## 内容与状态

| 主题 | 仓库内容 | 当前边界 |
| --- | --- | --- |
| C++ | 多文件构建、对象生命周期、智能指针、STL 与 Buffer 所有权练习 | 教学示例，不构成通用库 |
| PyTorch Tensor 与梯度 | shape、索引、广播、stride、别名和手算梯度验证 | 小型 CPU/GPU 实验；不替代完整训练课程 |
| CUDA | Host/Device、线程索引、内存层级、归约、scan、矩阵乘法与 profiling 笔记 | 部分内容为讲义示例；执行结果受 GPU、驱动和 Toolkit 版本影响 |
| 基准测试 | Vector Add、环境快照和计时规范 | 现有 Vector Add 使用 PyTorch；不是自定义 CUDA kernel 的性能结论 |
| 推理与编译器 | 推理服务、扩展、图和编译器主题的学习入口 | 文档与基础目录为主，功能按各模块状态逐步实现 |

## 快速开始

要求 Python 3.10 或更高版本。CPU 基础实验无需 NVIDIA GPU。克隆地址可替换为本项目的 HTTPS 或 SSH 地址：

```bash
git clone <repository-url> mlsys-learning-lab
cd mlsys-learning-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[benchmark]'
bash scripts/check_environment.sh --purpose=bootstrap
```

运行一个小型 CPU 基准：

```bash
python benchmarks/benchmark_vector_add.py \
  --device cpu --n 1024 --warmup 1 --iters 2
```

PyTorch 实验可安装 `python -m pip install -e '.[torch,benchmark]'`。CUDA 版 PyTorch、NVIDIA 驱动和系统 CUDA Toolkit 是不同组件；需要编译 `.cu` 源码时才必须安装匹配的 Toolkit。安装细节与排障见[部署指南](DEPLOYMENT_GUIDE.md)。

## 从哪里开始

1. 阅读[学习材料索引](learning/README.md)，按已有基础选择 C++ 或 Tensor 路线。
2. 按具体目录 README 中的命令运行实验，先理解输入、输出和正确性条件。
3. 对 GPU 计时先 warmup 并同步；区分 kernel 时间和端到端时间。
4. 把环境、命令、参数、原始结果和限制写入实验报告。

新读者可以从 [PyTorch 与模型基础章节目录](learning/stage-02/README.md) 或 [C++ 示例目录](learning/stage-01/examples/README.md) 开始。CUDA 专题索引见 [CUDA 学习材料](learning/stage-00/before-learning/README.md)。

## 目录导航

| 路径 | 用途 | 入口 |
| --- | --- | --- |
| `learning/` | 分阶段教程、程序导读和练习 | [学习索引](learning/README.md) |
| `Operator-notes/` | CUDA 与 GPU 架构专题笔记 | [CUDA 学习材料](learning/stage-00/before-learning/README.md) |
| `benchmarks/` | 基准脚本与测量规范 | [基准说明](benchmarks/README.md) |
| `cuda_kernels/` | CUDA kernel 实验 | [CUDA 算子](cuda_kernels/README.md) |
| `pytorch_extensions/` | PyTorch C++/CUDA 扩展 | [扩展说明](pytorch_extensions/README.md) |
| `inference/` | 推理执行与服务实验 | [推理说明](inference/README.md) |
| `compiler/` | 图、IR、Pass 和代码生成实验 | [编译器说明](compiler/README.md) |
| `tests/` | 测试入口和跨模块测试约定 | [测试说明](tests/README.md) |
| `reports/` | 环境记录与实验结论 | [报告规范](reports/README.md) |
| `scripts/` | 环境检查等辅助脚本 | [脚本说明](scripts/README.md) |

## 实验与贡献约定

- 先检查语义和正确性，再讨论性能；优化实验保留可比较的 baseline。
- 报告记录输入、随机种子、软件/硬件环境、构建选项和 Git commit。环境信息只记录复现所需内容，不记录个人账号、主机名、访问令牌或本机私有路径。
- GPU 测量需处理异步执行，进行 warmup 和多次采样，并说明统计方法、测量范围及误差来源。
- 文档中的公式使用普通文本或代码格式，避免依赖特定 Markdown 数学扩展。
- Issue 和 PR 请提供可复现步骤。代码改动应附相应正确性验证；性能改动还应提供 baseline、环境和原始指标。

常用检查：

```bash
git diff --check
python benchmarks/benchmark_vector_add.py \
  --device cpu --n 1024 --warmup 1 --iters 2
```

C++ 示例的 CMake/CTest 命令见 [测试说明](tests/README.md)。

## 许可证

当前仓库尚未包含开源许可证。添加许可证之前，公开可见不代表已授予复制、修改、分发或商用权限；复用前请确认许可状态。
