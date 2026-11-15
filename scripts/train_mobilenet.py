"""Compatibility entry point; see README.md for the unified pipeline."""
import sys
import runpy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    runpy.run_module("wastewise.finetune", run_name="__main__")
