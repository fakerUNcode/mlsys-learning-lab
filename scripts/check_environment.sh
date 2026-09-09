#!/usr/bin/env bash

# Read-only environment diagnostic. Extra arguments are intentionally preserved
# in the report so that a result can be reproduced later.
set -u

if [ -n "${PYTHON_BIN:-}" ]; then
    : "${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

section() { printf '\n=== %s ===\n' "$1"; }
version_or_missing() {
    if command -v "$1" >/dev/null 2>&1; then
        "$@" 2>&1 | head -n 1
    else
        printf '%s: not found\n' "$1"
    fi
}

section "Run"
printf 'timestamp: %s\n' "$(date -Is 2>/dev/null || date)"
printf 'working directory: %s\n' "$PWD"
printf 'script: %s\n' "$0"
printf 'arguments:'
if [ "$#" -eq 0 ]; then printf ' (none)\n'; else printf ' %q' "$@"; printf '\n'; fi

section "Git"
if command -v git >/dev/null 2>&1 && git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'root: %s\n' "$ROOT_DIR"
    printf 'commit: %s\n' "$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf 'unknown')"
    printf 'branch: %s\n' "$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || printf 'unknown')"
    printf 'worktree: %s\n' "$(git -C "$ROOT_DIR" status --short | tr '\n' ';' || printf 'unknown')"
else
    printf 'git: not available or repository not detected\n'
fi

section "System"
uname -a 2>/dev/null || true
if command -v lsb_release >/dev/null 2>&1; then lsb_release -ds 2>/dev/null || true; fi

section "Compiler and build tools"
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
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv 2>/dev/null || \
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || true
else
    printf 'nvidia-smi: not found (GPU/driver details unavailable)\n'
fi

section "PyTorch and CUDA runtime"
if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
"$PYTHON_BIN" - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch CUDA runtime:", torch.version.cuda or "none")
    print("cuDNN:", torch.backends.cudnn.version() or "none")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            capability = torch.cuda.get_device_capability(index)
            total = torch.cuda.get_device_properties(index).total_memory
            print(f"GPU {index}: {name}")
            print(f"  compute capability: {capability[0]}.{capability[1]}")
            print(f"  memory: {total / (1024**3):.2f} GiB")
except Exception as exc:
    print("PyTorch check failed:", repr(exc))
PY
else
    printf '%s: not found (PyTorch check skipped)\n' "$PYTHON_BIN"
fi
