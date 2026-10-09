"""旧路径兼容入口：转调 T02 正式目录中的 shape verifier。"""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "t02-tensor-basics" / "verify_shapes.py"),
    run_name="__main__",
)
