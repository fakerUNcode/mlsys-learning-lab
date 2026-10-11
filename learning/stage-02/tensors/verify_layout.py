"""旧路径兼容入口：转调 T03 正式目录中的 layout verifier。"""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "batches" / "b01-environment-tensor-layout" / "t03-tensor-layout-and-storage" / "verify_layout.py"),
    run_name="__main__",
)
