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
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from ..config import get_settings
from ..events import Evidence, GameEvent, PositionEvent, Provenance, SourceKind

# Minimap disc in the CDL_2026 broadcast (fractions of frame). Bottom-left,
# trimmed to avoid the player-cam nameplate to the right and the rug below.
MINIMAP_REGION = {"x": 0.012, "y": 0.715, "w": 0.140, "h": 0.260}
MODEL_NAME = "minimap_classical_marker_detector"
MODEL_VERSION = "0.2.0"
TRAINING_DATASET = "none_classical_cv"
PROCESSING_VERSION = "phase6_minimap_contract_v1"
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
EVALUATION_METRICS = {
    "real_labelled_map_accuracy": None,
    "reason": "classical minimap detector is a contract baseline; no player-resolved labelled eval yet",
}
Visibility = Literal["observed_team", "radar_visible_enemy", "unknown"]


@dataclass(frozen=True)
class Detection:
    x: float          # normalised box centre 0..1 within the minimap
    y: float
    w: float          # normalised box size (for YOLO pre-labels)
    h: float
    team: str         # "enemy" | "observed"  (low confidence)
    confidence: float
    area: int
    observed_team: str | None = None
    visibility: Visibility = "unknown"
    human_review_status: str = "unreviewed"

    def as_dict(self) -> dict:
        return {"x": round(self.x, 4), "y": round(self.y, 4), "w": round(self.w, 4),
                "h": round(self.h, 4), "team": self.team,
                "confidence": round(self.confidence, 3), "area": self.area}

    def rich_dict(self) -> dict[str, Any]:
        return {
            **self.as_dict(),
            "bbox_xywh_norm": [
                round(max(0.0, self.x - self.w / 2), 4),
                round(max(0.0, self.y - self.h / 2), 4),
                round(self.w, 4),
                round(self.h, 4),
            ],
            "observed_team": self.observed_team,
            "visibility": self.visibility,
            "human_review_status": self.human_review_status,
        }


@dataclass(frozen=True)
class MinimapFrameResult:
    detections: list[Detection]
    video_timestamp_seconds: float
    frame_index: int | None
    frame_path: str | None
    crop_path: str | None
    crop_box: dict[str, float]
    observed_team: str | None
    model_name: str
    model_version: str
    training_dataset: str
    latency_ms: float
    failure_reason: str | None
    fallback_used: bool
    evaluation_metrics: dict[str, Any]
    processing_version: str

    @property
    def accepted(self) -> bool:
        return self.failure_reason is None

    def accepted_detections(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> list[Detection]:
        return [d for d in self.detections if d.confidence >= confidence_threshold]

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_timestamp_seconds": self.video_timestamp_seconds,
            "frame_index": self.frame_index,
            "frame_path": self.frame_path,
            "crop_path": self.crop_path,
            "crop_box": self.crop_box,
            "observed_team": self.observed_team,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "training_dataset": self.training_dataset,
            "latency_ms": round(self.latency_ms, 3),
            "failure_reason": self.failure_reason,
            "fallback_used": self.fallback_used,
            "evaluation_metrics": self.evaluation_metrics,
            "processing_version": self.processing_version,
            "detections": [d.rich_dict() for d in self.detections],
        }


def crop_minimap(frame: np.ndarray, region: dict | None = None) -> np.ndarray:
    region = region or MINIMAP_REGION
    h, w = frame.shape[:2]
    x0, y0 = int(region["x"] * w), int(region["y"] * h)
    return frame[y0:y0 + int(region["h"] * h), x0:x0 + int(region["w"] * w)]


class ClassicalMinimapDetector:
    """MinimapDetector implementation. `detect()` returns player-marker
    Detections; `observed_team` is carried so callers never infer hidden units."""

    def detect(self, frame: np.ndarray, hud_profile: dict | None = None) -> list[dict]:
        """Legacy compact detector output used by the YOLO dataset seeder."""
        return [d.as_dict() for d in self.read_frame(frame, hud_profile).detections]

    def read_frame(
        self,
        frame: np.ndarray,
        hud_profile: dict | None = None,
        *,
        frame_index: int | None = None,
        video_timestamp_seconds: float = 0.0,
        frame_path: str | None = None,
        crop_path: str | None = None,
        observed_team: str | None = None,
        human_review_status: str = "unreviewed",
    ) -> MinimapFrameResult:
        start = time.perf_counter()
        region = (hud_profile or {}).get("regions", {}).get("minimap", MINIMAP_REGION)
        if frame is None or frame.size == 0:
            return self._result(
                [],
                start,
                video_timestamp_seconds,
                frame_index,
                frame_path,
                crop_path,
                region,
                observed_team,
                "frame_unreadable",
            )
        crop = crop_minimap(frame, region)
        if crop.size == 0:
            return self._result(
                [],
                start,
                video_timestamp_seconds,
                frame_index,
                frame_path,
                crop_path,
                region,
                observed_team,
                "minimap_crop_empty",
            )
        detections = [
            replace(
                d,
                observed_team=observed_team,
                visibility="radar_visible_enemy" if d.team == "enemy" else "observed_team",
                human_review_status=human_review_status,
            )
            for d in self._detect_crop(crop)
        ]
        return self._result(
            detections,
            start,
            video_timestamp_seconds,
            frame_index,
            frame_path,
            crop_path,
            region,
            observed_team,
            None,
        )

    def _result(
        self,
        detections: list[Detection],
        start: float,
        video_timestamp_seconds: float,
        frame_index: int | None,
        frame_path: str | None,
        crop_path: str | None,
        crop_box: dict[str, float],
        observed_team: str | None,
        failure_reason: str | None,
    ) -> MinimapFrameResult:
        return MinimapFrameResult(
            detections=detections,
            video_timestamp_seconds=video_timestamp_seconds,
            frame_index=frame_index,
            frame_path=frame_path,
            crop_path=crop_path,
            crop_box=crop_box,
            observed_team=observed_team,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            training_dataset=TRAINING_DATASET,
            latency_ms=(time.perf_counter() - start) * 1000,
            failure_reason=failure_reason,
            fallback_used=False,
            evaluation_metrics=EVALUATION_METRICS,
            processing_version=PROCESSING_VERSION,
        )

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
                dets.append(Detection(cx / ww, cy / hh, w / ww, h / hh, team, conf, area))
        return _dedupe(dets)


def position_events_from_minimap_result(
    result: MinimapFrameResult,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    broadcast_id: int | None = None,
    match_id: int | None = None,
    map_id: int | None = None,
) -> list[GameEvent]:
    """Convert accepted minimap detections into evidence-backed PositionEvents.

    No visual evidence means no event. Low-confidence detections are withheld.
    Hidden opponents are never inferred; enemy markers are only radar-visible
    observations on the broadcast minimap.
    """
    if not result.accepted or not (result.frame_path or result.crop_path):
        return []
    events: list[GameEvent] = []
    for detection in result.accepted_detections(confidence_threshold):
        team = result.observed_team if detection.team == "observed" else None
        events.append(
            GameEvent(
                broadcast_id=broadcast_id,
                match_id=match_id,
                map_id=map_id,
                video_timestamp_seconds=result.video_timestamp_seconds,
                confidence=detection.confidence,
                provenance=Provenance(
                    source=SourceKind.HEURISTIC,
                    model_name=result.model_name,
                    model_version=result.model_version,
                    producer="minimap-detector",
                    note="classical minimap marker detector; YOLO can replace detector body later",
                ),
                evidence=Evidence(
                    video_timestamp_seconds=result.video_timestamp_seconds,
                    frame_index=result.frame_index,
                    frame_path=result.frame_path,
                    crop_path=result.crop_path,
                ),
                payload=PositionEvent(
                    x=detection.x,
                    y=detection.y,
                    team=team,
                    detector="minimap",
                    observed_team=result.observed_team,
                    attributes={
                        "source": "minimap_detection",
                        "team_marker": detection.team,
                        "visibility": detection.visibility,
                        "bbox_xywh_norm": detection.rich_dict()["bbox_xywh_norm"],
                        "area": detection.area,
                        "crop_box": result.crop_box,
                        "training_dataset": result.training_dataset,
                        "evaluation_metrics": result.evaluation_metrics,
                        "processing_version": result.processing_version,
                        "human_review_status": detection.human_review_status,
                    },
                ),
                tags=["minimap", "position"],
            )
        )
    return events


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
