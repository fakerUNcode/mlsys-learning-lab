# T01：当前设备环境验收

本章记录机器环境与 CUDA 工具链的验收结果。实验材料集中在验证报告，通用环境采集脚本仍由仓库 `scripts/` 维护。

- [T01 环境验收报告](../../../reports/stage-02/2026-09-27-t01-environment.md)：记录 WSL/GPU、PyTorch CUDA runtime、Triton、`nvcc`、原生 CUDA kernel 和 profiler 检查及其边界。
- [环境检查脚本](../../../scripts/check_environment.sh)：从仓库根目录执行 `PYTHON_BIN=.venv/bin/python bash scripts/check_environment.sh --purpose=stage-0`。

报告区分了“能运行 PyTorch/Triton GPU 操作”和“已安装完整 CUDA Toolkit 并能用 `nvcc` 编译 `.cu`”两类证据。复查当前机器时重新运行检查；报告中的设备、驱动、显存和软件版本是 2026-09-27 的快照。
