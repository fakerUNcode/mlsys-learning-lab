# MLSys Learning Lab

统一的机器学习系统学习与实验仓库，覆盖 GPU/CUDA kernel、PyTorch C++/CUDA 扩展、推理系统、编译器和性能基准。仓库的目标不是只收集代码，而是让每个实验都能被复现、测量、解释和沉淀为报告。

## 当前状态与还剩哪些任务

仓库骨架、Python 项目配置、环境自检脚本和模块级文档已建立。当前仍需按个人学习路线逐步补齐可运行实验：

1. 在 `benchmarks/` 建立统一的输入规模、warmup、重复次数、计时同步和结果保存规范。
2. 在 `cuda_kernels/` 完成至少一个 baseline kernel 与一个优化版本，并记录 occupancy、带宽、访存和正确性结果。
3. 在 `pytorch_extensions/` 完成一个可安装、可测试的 C++/CUDA extension，覆盖 CPU fallback 与错误处理。
4. 在 `inference/` 建立端到端推理基线，测量 latency、throughput、显存和 batch size 的关系。
5. 在 `compiler/` 选择一个编译器/IR 实验（如 FX、TorchInductor、Triton 或 MLIR），保存 IR、优化前后代码和分析。
6. 为以上模块补充 `tests/` 中的单元测试、数值校验和最小 CI；必要时增加真实 GPU 测试标记。
7. 在 `reports/` 逐项写实验报告，记录硬件、软件版本、命令、原始结果、结论和下一步。
8. 在真实 NVIDIA 环境执行 `bash scripts/check_environment.sh`，把输出作为每次实验的环境快照。

这里的“完成”标准是：代码能运行、结果能复现、基线有对照、结论有数据，而不是目录中仅有示例文件。

## 建立文件大纲与项目总架构

```text
mlsys-learning-lab/
├── benchmarks/              # 基准测试、计时工具、结果格式与实验矩阵
├── cuda_kernels/             # CUDA kernel、launch 配置、正确性与性能实验
├── pytorch_extensions/       # PyTorch C++/CUDA extension 与 Python 封装
├── inference/                # 模型推理、服务化、batch/精度/显存实验
├── compiler/                 # 图捕获、IR、算子融合、代码生成与编译实验
├── reports/                  # 各实验报告、原始结果和复盘材料
├── scripts/                  # 环境检查、运行、汇总和自动化脚本
├── tests/                    # 跨模块测试与回归测试
├── pyproject.toml            # Python 版本、依赖和构建元数据
├── DEPLOYMENT_GUIDE.md       # 项目介绍、安装部署和运行手册
├── LEARNING_REPORT.md        # 相关知识点的系统学习报告
└── README.md                 # 总览、架构、剩余任务和导航
```

## 全局工作流

```text
环境自检 → 选择问题与 baseline → 实现 → 正确性验证 → 性能测量
    ↑                                             ↓
报告复现 ← 保存命令/版本/数据/图表 ← 分析瓶颈与优化假设
```

各模块 README 给出该目录的职责、推荐文件布局、输入输出、验证指标和完成标准。跨模块的公共约定是：先保证正确性，再做性能优化；所有 GPU 计时都要同步；所有结论都要带硬件、软件版本和运行参数。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[benchmark]'
bash scripts/check_environment.sh --purpose=bootstrap
pytest -q
```

若需要 PyTorch，请按本机 CUDA 驱动和官方安装矩阵安装匹配的 `torch`，再运行自检脚本。完整部署、运行、排障和报告规范见 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)；知识路线见 [LEARNING_REPORT.md](LEARNING_REPORT.md)。

如果你是 GPU 初学者，请先阅读 [SELF_STUDY_GUIDE.md](SELF_STUDY_GUIDE.md)，按照阶段任务推进，不要直接跳到复杂 kernel 或大模型优化。

## 目录导航

- [benchmarks/README.md](benchmarks/README.md)
- [cuda_kernels/README.md](cuda_kernels/README.md)
- [pytorch_extensions/README.md](pytorch_extensions/README.md)
- [inference/README.md](inference/README.md)
- [compiler/README.md](compiler/README.md)
- [reports/README.md](reports/README.md)
- [reports/stage0_environment.md](reports/stage0_environment.md)
- [scripts/README.md](scripts/README.md)
- [tests/README.md](tests/README.md)
