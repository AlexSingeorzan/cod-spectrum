"""Minimap player-marker detection (classical CV) + an occupancy heatmap.

This implements the `MinimapDetector` contract from `hardpoint_breakdown.py` with
a classical, training-free color/shape detector. It is honest about its ceiling:

  * It reliably LOCALISES bright player markers (the arrow + number blips) on the
    broadcast minimap, giving real positions and an occupancy heatmap.
  * Team classification is BEST-EFFORT (a marker ringed by saturated red is
    tagged `enemy`, otherwise `observed`) and carries low confidence — the
    compound arrow+white-number markers and broadcast compression make exact
    per-player team tracking unreliable without a trained model.
  * The broadcast minimap shows the OBSERVED team plus radar-visible enemies, so
    detections carry `observed_team`/`visibility`; hidden opponents are never
    invented (the README's rule).

Upgrade path (documented, not claimed done): replace `ClassicalMinimapDetector`
with a YOLO model trained on labelled minimap crops. The interface, evidence
persistence and downstream heatmap/spawn derivation stay identical — only the
`detect()` body changes. That is the real route to per-player, team-accurate
positions and detected (not inferred) spawn flips.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import get_settings

# Minimap disc in the CDL_2026 broadcast (fractions of frame). Bottom-left,
# trimmed to avoid the player-cam nameplate to the right and the rug below.
MINIMAP_REGION = {"x": 0.012, "y": 0.715, "w": 0.140, "h": 0.260}


@dataclass(frozen=True)
class Detection:
    x: float          # normalised 0..1 within the minimap
    y: float
    team: str         # "enemy" | "observed"  (low confidence)
    confidence: float
    area: int

    def as_dict(self) -> dict:
        return {"x": round(self.x, 4), "y": round(self.y, 4), "team": self.team,
                "confidence": round(self.confidence, 3), "area": self.area}


def crop_minimap(frame: np.ndarray, region: dict | None = None) -> np.ndarray:
    region = region or MINIMAP_REGION
    h, w = frame.shape[:2]
    x0, y0 = int(region["x"] * w), int(region["y"] * h)
    return frame[y0:y0 + int(region["h"] * h), x0:x0 + int(region["w"] * w)]


class ClassicalMinimapDetector:
    """MinimapDetector implementation. `detect()` returns player-marker
    Detections; `observed_team` is carried so callers never infer hidden units."""

    def detect(self, frame: np.ndarray, hud_profile: dict | None = None) -> list[dict]:
        region = (hud_profile or {}).get("regions", {}).get("minimap", MINIMAP_REGION)
        crop = crop_minimap(frame, region)
        return [d.as_dict() for d in self._detect_crop(crop)]

    def _detect_crop(self, crop: np.ndarray) -> list[Detection]:
        hh, ww = crop.shape[:2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # static non-map regions: orange rug/nameplate bleed
        rug = cv2.dilate(cv2.inRange(hsv, (9, 90, 90), (26, 255, 255)), np.ones((3, 3), np.uint8))
        keep = cv2.bitwise_not(rug)
        red = (cv2.inRange(hsv, (0, 95, 95), (9, 255, 255)) | cv2.inRange(hsv, (166, 95, 95), (180, 255, 255))) & keep
        bright = cv2.inRange(hsv, (0, 0, 205), (180, 60, 255)) & keep
        dets: list[Detection] = []
        # bright markers = player arrows/number blips (observed team mostly)
        for mask, base_team, conf in [(red, "enemy", 0.5), (bright, "observed", 0.4)]:
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = int(cv2.contourArea(c))
                x, y, w, h = cv2.boundingRect(c)
                aspect = w / h if h else 9
                if not (7 <= area <= 170 and 0.4 <= aspect <= 2.6 and max(w, h) <= 22):
                    continue
                cx, cy = x + w / 2, y + h / 2
                # team tint: SATURATED red in a ring around the marker -> enemy.
                # (must check saturation: white/dark pixels report hue 0 too.)
                ring = hsv[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3]
                if ring.size:
                    hue, sat = ring[..., 0], ring[..., 1]
                    redness = float(np.mean(((hue < 10) | (hue > 168)) & (sat > 90)))
                else:
                    redness = 0.0
                team = "enemy" if (base_team == "enemy" or redness > 0.35) else "observed"
                dets.append(Detection(cx / ww, cy / hh, team, conf, area))
        return _dedupe(dets)


def _dedupe(dets: list[Detection], min_dist: float = 0.05) -> list[Detection]:
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda z: z.confidence, reverse=True):
        if all((d.x - k.x) ** 2 + (d.y - k.y) ** 2 > min_dist ** 2 for k in kept):
            kept.append(d)
    return kept


@dataclass
class HeatmapResult:
    grid: list[list[float]]
    size: int
    frames: int
    detections: int
    note: str = field(default="classical CV marker localisation; team tint is low-confidence")


def occupancy_heatmap(frame_detections: list[list[dict]], size: int = 16) -> HeatmapResult:
    """Aggregate per-frame detections into a normalised size×size occupancy grid."""
    grid = np.zeros((size, size), dtype=float)
    total = 0
    for dets in frame_detections:
        for d in dets:
            gx = min(size - 1, int(d["x"] * size))
            gy = min(size - 1, int(d["y"] * size))
            grid[gy][gx] += 1
            total += 1
    if grid.max() > 0:
        grid = grid / grid.max()
    return HeatmapResult(grid=grid.round(3).tolist(), size=size,
                         frames=len(frame_detections), detections=total)


def heatmap_path() -> Path:
    return get_settings().data_dir / "reports" / "lat_van_hp_minimap.json"


def saved_heatmaps() -> dict:
    """Precomputed per-hill occupancy heatmaps (needs the local VOD to (re)build,
    so the result is cached to disk and shipped for the UI)."""
    path = heatmap_path()
    return json.loads(path.read_text()) if path.exists() else {}
