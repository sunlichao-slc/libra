"""Run a legacy script from the repository root by relative path."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root (parent of scripts/)
LEGACY_ROOT = ROOT / "scripts" / "legacy"


def run(relative_path: str) -> None:
    candidates = [
        LEGACY_ROOT / relative_path,
        ROOT / relative_path,
    ]
    target = next((p for p in candidates if p.is_file()), None)
    if target is None:
        searched = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Legacy script not found: {relative_path}. Searched: {searched}")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
