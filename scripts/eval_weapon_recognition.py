"""Compatibility wrapper for coarse kill-type recognition evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_kill_type_recognition import evaluate, main


if __name__ == "__main__":
    raise SystemExit(main())
