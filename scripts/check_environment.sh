#!/usr/bin/env bash

set -e

echo "=== System ==="
uname -a
lsb_release -a 2>/dev/null || true

echo
echo "=== Compiler ==="
g++ --version | head -n 1
cmake --version | head -n 1
ninja --version

echo
echo "=== Python ==="
python --version
pip --version

echo
echo "=== NVIDIA ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "nvidia-smi not found"
fi

echo
echo "=== PyTorch ==="
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda:", torch.version.cuda)

    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
except Exception as e:
    print("PyTorch check failed:", repr(e))
PY
