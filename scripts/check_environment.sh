#!/usr/bin/env bash

# 只读环境诊断脚本。
#
# 用途：
#   在每次实验前保存“这次到底使用了什么环境”，避免结果脱离
#   GPU、驱动、CUDA、PyTorch、编译器和 Git commit 后无法复现。
#
# 示例：
#   bash scripts/check_environment.sh --purpose=stage1
#
# 可选环境变量：
#   PYTHON_BIN=.venv/bin/python bash scripts/check_environment.sh
#
# 脚本不会安装、删除或修改软件；额外参数只会原样记录到报告。

# -u：读取未定义变量时立即报错。
# 本脚本没有开启 -e，因为某些诊断命令失败不应中止整份报告；
# 例如没有 NVIDIA GPU 时，仍应继续输出编译器和 Python 信息。
set -u

# 优先级：
# 1. 调用者显式提供 PYTHON_BIN；
# 2. PATH 中存在 python；
# 3. 尝试 python3。
if [ -n "${PYTHON_BIN:-}" ]; then
    # 冒号命令不执行操作；这里明确表示保留调用者传入的值。
    : "${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

# BASH_SOURCE[0] 是当前脚本文件的位置。
# 先得到绝对脚本目录，再向上一级得到仓库根目录。
# 这样无论从哪个工作目录调用，Git 检查都指向同一仓库。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 打印统一章节标题。
# 将函数参数放进 "%s" 而不是格式字符串，可避免意外格式解析。
section() {
    printf '\n=== %s ===\n' "$1"
}

# 输出某个工具的第一行版本信息。
# "$1" 用于 command -v 探测命令是否存在；
# "$@" 保留命令及其后续参数，例如 gcc --version。
version_or_missing() {
    if command -v "$1" >/dev/null 2>&1; then
        "$@" 2>&1 | head -n 1
    else
        printf '%s: not found\n' "$1"
    fi
}

section "Run"

# date -Is 输出 ISO 8601 时间；不支持时退回普通 date。
printf 'timestamp: %s\n' "$(date -Is 2>/dev/null || date)"
printf 'working directory: %s\n' "$PWD"
printf 'script: %s\n' "$0"

# %q 以 shell 可重新读取的形式转义参数，空格不会破坏记录。
printf 'arguments:'
if [ "$#" -eq 0 ]; then
    printf ' (none)\n'
else
    printf ' %q' "$@"
    printf '\n'
fi

section "Git"

# 同时检查 git 命令和目标目录是否真是 Git 工作树。
if command -v git >/dev/null 2>&1 && git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'root: %s\n' "$ROOT_DIR"
    printf 'commit: %s\n' "$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf 'unknown')"
    printf 'branch: %s\n' "$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || printf 'unknown')"

    # status --short 的多行输出压成分号分隔的一行，方便嵌入报告。
    printf 'worktree: %s\n' "$(git -C "$ROOT_DIR" status --short | tr '\n' ';' || printf 'unknown')"
else
    printf 'git: not available or repository not detected\n'
fi

section "System"

# uname 失败不阻塞其他检查。
uname -a 2>/dev/null || true

# 某些发行版没有 lsb_release，因此先探测。
if command -v lsb_release >/dev/null 2>&1; then
    lsb_release -ds 2>/dev/null || true
fi

section "Compiler and build tools"

# version_or_missing 会在工具缺失时输出明确的 not found。
version_or_missing gcc --version
version_or_missing g++ --version
version_or_missing nvcc --version
version_or_missing cmake --version
version_or_missing ninja --version

section "Python"

version_or_missing "$PYTHON_BIN" --version
version_or_missing "$PYTHON_BIN" -m pip --version

section "NVIDIA"

if command -v nvidia-smi >/dev/null 2>&1; then
    # 新版 nvidia-smi 支持 compute_cap 字段。
    # 如果第一条查询失败，退回不含 compute_cap 的兼容查询。
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv 2>/dev/null \
        || nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null \
        || true
else
    printf 'nvidia-smi: not found (GPU/driver details unavailable)\n'
fi

section "PyTorch and CUDA runtime"

if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    # 使用选定的 Python 执行内嵌脚本。
    # <<'PY' 的引号会禁止 shell 展开 Python 内容中的 $ 等字符。
    "$PYTHON_BIN" - <<'PY'
try:
    import torch

    # PyTorch 包版本与它编译时对应的 CUDA Runtime 版本。
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch CUDA runtime:", torch.version.cuda or "none")
    print("cuDNN:", torch.backends.cudnn.version() or "none")

    # 只有 CUDA 真正可用时才枚举设备，避免无 GPU 环境抛错。
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            capability = torch.cuda.get_device_capability(index)
            properties = torch.cuda.get_device_properties(index)
            total = properties.total_memory

            print(f"GPU {index}: {name}")
            print(
                "  compute capability: "
                f"{capability[0]}.{capability[1]}"
            )
            print(f"  memory: {total / (1024**3):.2f} GiB")
except Exception as exc:
    # 环境检查应报告失败原因，而不是让整份 shell 报告中断。
    print("PyTorch check failed:", repr(exc))
PY
else
    printf '%s: not found (PyTorch check skipped)\n' "$PYTHON_BIN"
fi
