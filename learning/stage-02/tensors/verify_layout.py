"""Backward-compatible entry point for the T03 layout verifier."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).parent / "t03" / "verify_layout.py"), run_name="__main__")
