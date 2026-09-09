# 项目介绍与部署运行说明书

## 1. 项目介绍

`mlsys-learning-lab` 是一个面向机器学习系统的可复现实验仓库。它把底层 CUDA kernel、PyTorch 扩展、推理性能、编译器优化和 benchmark 放在同一个工作流中，用统一的环境快照和报告规范连接起来。

## 2. 环境要求

- Linux、Python 3.10 或更高版本。
- 基础实验需要 Python、pip、C/C++ 编译器；CMake/Ninja 用于原生构建。
- GPU 实验需要 NVIDIA GPU、兼容驱动和 CUDA toolkit；PyTorch wheel 自带的 runtime 不等同于系统 `nvcc` toolkit。
- PyTorch、CUDA toolkit 与驱动版本必须按官方兼容矩阵匹配；无 GPU 时可运行 CPU 文档、测试和部分编译实验。

## 3. 安装

```bash
git clone <repository-url> mlsys-learning-lab
cd mlsys-learning-lab
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python -m pip install -e '.[benchmark]'
```

如需 PyTorch，先根据目标 CUDA 版本安装对应发行包，再确认：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

## 4. 部署后自检

```bash
bash scripts/check_environment.sh --purpose=post-install
```

重点检查：GPU 名称/Compute Capability/显存、驱动、PyTorch CUDA runtime、CUDA 可用性、`nvcc`、编译器、CMake、Ninja 和 Git commit。把输出保存到实验报告或 artifact 中；脚本是只读诊断，不会修改系统。

## 5. 运行顺序

```bash
pytest -q
python -m pip install -e '.[benchmark]'
# 按各目录 README 的命令运行具体实验
```

推荐顺序是“正确性 → baseline → profiling → 优化 → 回归 benchmark → 报告”。GPU 计时要进行 warmup 和同步；比较时固定输入、随机种子、batch、dtype 和设备状态。

## 6. 常见问题

- `torch.cuda.is_available()` 为 false：检查驱动、PyTorch wheel、容器 GPU 透传和 `nvidia-smi`，不要只看系统是否安装了 CUDA。
- `nvcc` 找不到：安装 CUDA toolkit 或修正 PATH；这不一定影响只使用 PyTorch wheel 的推理实验。
- extension 编译失败：核对 gcc/CUDA/PyTorch 组合、Compute Capability、头文件路径和 Ninja/CMake 版本。
- 结果波动大：增加 warmup/重复次数，固定频率或记录机器负载，分别报告分位数而非只报告平均值。
- 显存不足：减小 batch/输入尺寸，确认没有缓存或重复模型副本，并记录峰值显存。

## 7. 交付检查清单

环境自检输出已保存；代码在干净环境可安装；测试通过；baseline 和优化版本都可运行；报告包含 commit、命令、参数、硬件和原始数据；大模型权重、编译产物和临时结果不直接提交到 Git。
