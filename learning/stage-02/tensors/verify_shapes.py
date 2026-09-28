"""Backward-compatible entry point for the T02 shape verifier."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).parent / "t02" / "verify_shapes.py"), run_name="__main__")
