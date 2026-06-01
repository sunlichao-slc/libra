#!/usr/bin/env python3
"""Set dataset: cub200 in config.yaml, then run train.py."""
import runpy
import sys
from pathlib import Path

target = Path(__file__).resolve().parent / "train.py"
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name="__main__")
