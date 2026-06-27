"""Compatibility wrapper for the coarse kill-type dataset builder."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_kill_type_dataset import build_kill_type_dataset, main

build_weapon_dataset = build_kill_type_dataset


if __name__ == "__main__":
    raise SystemExit(main())
