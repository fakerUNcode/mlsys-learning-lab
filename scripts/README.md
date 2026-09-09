# scripts/

## 职责

放置跨模块自动化入口，保持脚本短小、可组合、默认只读。脚本不应把机器相关路径硬编码进仓库。

## 已有脚本

### `check_environment.sh`

输出系统、Git commit/分支/工作区、GPU 名称、Compute Capability、显存、NVIDIA 驱动、`nvcc`/PyTorch CUDA runtime、PyTorch CUDA 可用性、编译器、CMake、Ninja、Python，以及本次运行参数。没有 GPU 或工具时会明确打印 unavailable/not found，而不是静默失败。

```bash
bash scripts/check_environment.sh --purpose=kernel-benchmark --iters=1000
PYTHON_BIN=python3 bash scripts/check_environment.sh
```

后续可增加统一 benchmark runner、报告汇总和 CI 检查，但应保持所有命令可从仓库根目录执行。
