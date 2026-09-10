# Scripts

保存跨模块自动化入口。脚本应短小、可组合、可从仓库根目录运行，并避免硬编码机器相关路径。

## 当前脚本

### 环境检查

[`check_environment.sh`](check_environment.sh) 输出系统、Git 状态、GPU、Compute Capability、显存、驱动、`nvcc`、PyTorch CUDA Runtime、编译器、CMake、Ninja、Python 和实验参数。

```bash
bash scripts/check_environment.sh \
  --purpose=kernel-benchmark --iters=1000
```

指定 Python：

```bash
PYTHON_BIN=.venv/bin/python \
  bash scripts/check_environment.sh --purpose=bootstrap
```

缺少 GPU 或工具时，脚本会输出 `unavailable` 或 `not found`，便于 CPU-only 环境保存真实快照。

## 脚本约定

- 默认行为安全，破坏性操作必须显式传参。
- 使用非零退出码表示真正失败，并写清诊断信息。
- 参数提供 `--help`，路径以仓库根目录为基准。
- 不输出访问令牌、私钥或其他敏感环境变量。
- 新脚本需记录用途、依赖、输入、输出和示例。

## 后续计划

后续可增加统一 benchmark runner、结果汇总和 CI 检查；在实现存在之前不提供占位命令。
