from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings


@dataclass(frozen=True)
class HudProfile:
    name: str
    base_resolution: tuple[int, int]
    regions: dict[str, dict[str, float]]
    ocr_hints: dict[str, dict[str, Any]]


def load_hud_profile(name: str) -> HudProfile:
    path = get_settings().data_dir / "configs" / "hud_profiles" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"HUD profile not found: {path}")
    payload = json.loads(path.read_text())
    return HudProfile(
        name=payload["name"],
        base_resolution=tuple(payload["base_resolution"]),
        regions=payload["regions"],
        ocr_hints=payload.get("ocr_hints", {}),
    )


def region_pixels(region: dict[str, float], width: int, height: int) -> tuple[int, int, int, int]:
    x = max(0, min(width, round(region["x"] * width)))
    y = max(0, min(height, round(region["y"] * height)))
    w = max(1, round(region["w"] * width))
    h = max(1, round(region["h"] * height))
    return x, y, min(w, width - x), min(h, height - y)


def crop_region(frame: np.ndarray, region: dict[str, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x, y, w, h = region_pixels(region, width, height)
    return frame[y:y + h, x:x + w]

