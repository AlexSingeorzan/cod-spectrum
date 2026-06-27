"""Compatibility wrapper for the coarse kill-type recognition sample."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sample_kill_type_recognition import build_sample_dataset, main


if __name__ == "__main__":
    raise SystemExit(main())
