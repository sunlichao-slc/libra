#!/usr/bin/env python3
"""Recover images from embedding files using a trained denoiser checkpoint."""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from rar_diffusion.infer import main

if __name__ == "__main__":
    main()

