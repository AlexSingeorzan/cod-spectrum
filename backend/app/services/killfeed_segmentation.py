"""Killfeed row segmentation - Phase 4 Stage B.

This stage separates a detected killfeed row into field-level evidence regions:
attacker text, weapon icon, victim text, and optional indicators. It does not read
names or classify weapons. Low-confidence fields are returned as null segments so
later OCR/classifiers cannot silently inherit guessed boxes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_NAME = "killfeed_segmenter_classical"
MODEL_VERSION = "0.1.0"
TRAINING_DATASET = "none-classical-cv"
EVALUATION_METRICS = {
    "real_labelled_accuracy": None,
    "reason": "segmentation has no labelled real field boxes yet; readiness only",
}

CORE_FIELDS = ("attacker", "weapon", "victim")
OPTIONAL_FIELDS = ("headshot", "assist", "special")
ALL_FIELDS = CORE_FIELDS + OPTIONAL_FIELDS


@dataclass(frozen=True)
class Segment:
    field: str
    box: tuple[float, float, float, float] | None
    confidence: float
    crop_path: str | None = None
    failure_reason: str | None = None

    @property
    def present(self) -> bool:
        return self.box is not None and self.confidence > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "box": [round(v, 4) for v in self.box] if self.box else None,
            "confidence": round(self.confidence, 4),
            "crop_path": self.crop_path,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class KillfeedSegmentation:
    sample_id: str | None
    row_width: int
    row_height: int
    model_name: str
    model_version: str
    training_dataset: str
    confidence: float
    latency_ms: float
    fallback_used: bool
    failure_reason: str | None
    segments: dict[str, Segment]
    evaluation_metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "row_width": self.row_width,
            "row_height": self.row_height,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "training_dataset": self.training_dataset,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 3),
            "fallback_used": self.fallback_used,
            "failure_reason": self.failure_reason,
            "evaluation_metrics": self.evaluation_metrics,
            "segments": {field: segment.as_dict() for field, segment in self.segments.items()},
        }


def _runs_above(signal: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(signal):
        if value > threshold and start is None:
            start = i
        elif value <= threshold and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(signal)))
    return runs


def _merge_runs(runs: list[tuple[int, int]], max_gap: int = 5) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _bbox(mask: np.ndarray, x_range: tuple[int, int] | None = None) -> tuple[int, int, int, int] | None:
    if x_range is not None:
        x0, x1 = x_range
        sub = mask[:, max(0, x0):min(mask.shape[1], x1)]
        offset = max(0, x0)
    else:
        sub = mask
        offset = 0
    ys, xs = np.where(sub > 0)
    if len(xs) < 4 or len(ys) < 2:
        return None
    return int(xs.min() + offset), int(ys.min()), int(xs.max() + offset + 1), int(ys.max() + 1)


def _expand_box(box: tuple[int, int, int, int], width: int, height: int, pad_x: int = 2, pad_y: int = 2) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


def _normalise_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height


def _column_runs(mask: np.ndarray, threshold: float, smooth: int = 3, max_gap: int = 4) -> list[tuple[int, int]]:
    signal = mask.sum(axis=0) / 255.0
    if smooth > 1:
        signal = np.convolve(signal, np.ones(smooth) / smooth, mode="same")
    runs = _merge_runs(_runs_above(signal, threshold), max_gap=max_gap)
    return [(a, b) for a, b in runs if b - a >= 3]


def _range_from_runs(runs: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not runs:
        return None
    return runs[0][0], runs[-1][1]


def _run_fill(mask: np.ndarray, run: tuple[int, int]) -> float:
    a, b = run
    if b <= a:
        return 0.0
    return float(mask[:, a:b].mean() / 255.0)


def _segment_from_box(
    field: str,
    box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
    confidence: float,
    failure_reason: str | None,
) -> Segment:
    if box is None:
        return Segment(field=field, box=None, confidence=0.0, failure_reason=failure_reason)
    return Segment(
        field=field,
        box=_normalise_box(box, width, height),
        confidence=confidence,
        failure_reason=failure_reason,
    )


def crop_segment(image: np.ndarray, segment: Segment) -> np.ndarray | None:
    """Crop a segment from a row image using the segment's normalized box."""
    if segment.box is None:
        return None
    h, w = image.shape[:2]
    x, y, bw, bh = segment.box
    x0 = max(0, min(w, int(round(x * w))))
    y0 = max(0, min(h, int(round(y * h))))
    x1 = max(0, min(w, int(round((x + bw) * w))))
    y1 = max(0, min(h, int(round((y + bh) * h))))
    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1]


class KillfeedSegmenter:
    """Classical row-layout segmenter.

    The detector expects rows in the CDL killfeed layout: attacker text, a bright
    weapon icon cluster, and victim text. It only emits boxes when the row has two
    color-text clusters plus a bright region between them. Otherwise the field is
    null, not guessed.
    """

    name = MODEL_NAME
    model_version = MODEL_VERSION
    training_dataset = TRAINING_DATASET
    evaluation_metrics = EVALUATION_METRICS

    def segment(self, row_image: np.ndarray, sample_id: str | None = None) -> KillfeedSegmentation:
        start = time.perf_counter()
        h, w = row_image.shape[:2]
        segments: dict[str, Segment] = {}
        if h < 8 or w < 20:
            elapsed = (time.perf_counter() - start) * 1000
            segments = {
                field: Segment(field=field, box=None, confidence=0.0, failure_reason="row_too_small")
                for field in ALL_FIELDS
            }
            return KillfeedSegmentation(
                sample_id=sample_id,
                row_width=w,
                row_height=h,
                model_name=self.name,
                model_version=self.model_version,
                training_dataset=self.training_dataset,
                confidence=0.0,
                latency_ms=elapsed,
                fallback_used=False,
                failure_reason="row_too_small",
                segments=segments,
                evaluation_metrics=self.evaluation_metrics,
            )

        hsv = cv2.cvtColor(row_image, cv2.COLOR_BGR2HSV)
        _hue, saturation, value = cv2.split(hsv)
        color_mask = ((saturation > 85) & (value > 85)).astype(np.uint8) * 255
        white_mask = (((saturation < 85) & (value > 150)) | (value > 220)).astype(np.uint8) * 255
        ink_mask = cv2.bitwise_or(color_mask, white_mask)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((3, 11), np.uint8))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((2, 5), np.uint8))
        ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, np.ones((2, 5), np.uint8))

        ink_runs = _column_runs(ink_mask, max(1.0, h * 0.08), smooth=3, max_gap=max(3, w // 80))
        ink_runs = [(a, b) for a, b in ink_runs if b - a >= max(3, int(w * 0.018))]
        color_runs = _column_runs(color_mask, max(1.0, h * 0.08), smooth=5, max_gap=max(3, w // 70))
        color_runs = [(a, b) for a, b in color_runs if b - a >= max(3, int(w * 0.025))]

        weapon_candidates: list[tuple[float, int, tuple[int, int]]] = []
        for idx, run in enumerate(ink_runs[1:-1], start=1):
            width_frac = (run[1] - run[0]) / w
            white_fill = _run_fill(white_mask, run)
            if width_frac < 0.025 or white_fill < 0.035:
                continue
            center = ((run[0] + run[1]) / 2) / w
            icon_size_score = min(1.0, width_frac / 0.12)
            center_penalty = abs(center - 0.42) * 0.18
            score = white_fill + icon_size_score * 0.25 - center_penalty
            weapon_candidates.append((score, idx, run))

        if weapon_candidates:
            _score, weapon_idx, weapon_run = max(weapon_candidates, key=lambda item: item[0])
            before_runs = ink_runs[:weapon_idx]
            after_runs = ink_runs[weapon_idx + 1:]

            attacker_range = _range_from_runs(before_runs)
            attacker_box = _bbox(ink_mask, attacker_range) if attacker_range else None
            attacker_box = _expand_box(attacker_box, w, h) if attacker_box else None

            weapon_box = _bbox(white_mask, weapon_run)
            weapon_box = _expand_box(weapon_box, w, h, pad_x=3, pad_y=2) if weapon_box else None

            headshot_box = None
            victim_runs = after_runs
            if len(after_runs) >= 2:
                possible_headshot = after_runs[0]
                headshot_width = (possible_headshot[1] - possible_headshot[0]) / w
                if headshot_width <= 0.10 and _run_fill(white_mask, possible_headshot) > 0.05:
                    headshot_box = _bbox(white_mask, possible_headshot)
                    headshot_box = _expand_box(headshot_box, w, h, pad_x=2, pad_y=2) if headshot_box else None
                    victim_runs = after_runs[1:]

            victim_box = None
            if victim_runs:
                victim_start = victim_runs[0][0]
                color_after = [run for run in color_runs if run[0] >= victim_start - 2]
                if color_after:
                    victim_range = max(color_after, key=lambda run: run[1] - run[0])
                    victim_box = _bbox(color_mask, victim_range)
                else:
                    victim_range = _range_from_runs(victim_runs)
                    victim_box = _bbox(ink_mask, victim_range) if victim_range else None
                victim_box = _expand_box(victim_box, w, h) if victim_box else None

            segments["attacker"] = _segment_from_box(
                "attacker", attacker_box, w, h, 0.72, None if attacker_box else "attacker_text_not_detected"
            )
            segments["weapon"] = _segment_from_box(
                "weapon", weapon_box, w, h, 0.72, None if weapon_box else "weapon_icon_not_detected"
            )
            segments["victim"] = _segment_from_box(
                "victim", victim_box, w, h, 0.72, None if victim_box else "victim_text_not_detected"
            )
            segments["headshot"] = _segment_from_box(
                "headshot",
                headshot_box,
                w,
                h,
                0.45,
                None if headshot_box else "not_detected_by_classical_stage_b",
            )
        else:
            for field in CORE_FIELDS:
                segments[field] = Segment(
                    field=field,
                    box=None,
                    confidence=0.0,
                    failure_reason="weapon_icon_layout_not_detected",
                )

        optional_reason = "not_detected_by_classical_stage_b"
        for field in OPTIONAL_FIELDS:
            if field in segments:
                continue
            segments[field] = Segment(field=field, box=None, confidence=0.0, failure_reason=optional_reason)

        core = [segments[field] for field in CORE_FIELDS]
        present = [segment for segment in core if segment.present]
        confidence = float(np.mean([segment.confidence for segment in present])) if present else 0.0
        missing = [segment.field for segment in core if not segment.present]
        failure_reason = f"missing_core_segments:{','.join(missing)}" if missing else None
        elapsed = (time.perf_counter() - start) * 1000
        return KillfeedSegmentation(
            sample_id=sample_id,
            row_width=w,
            row_height=h,
            model_name=self.name,
            model_version=self.model_version,
            training_dataset=self.training_dataset,
            confidence=confidence,
            latency_ms=elapsed,
            fallback_used=False,
            failure_reason=failure_reason,
            segments=segments,
            evaluation_metrics=self.evaluation_metrics,
        )


def write_segment_crops(
    row_image: np.ndarray,
    segmentation: KillfeedSegmentation,
    out_dir: Path,
) -> KillfeedSegmentation:
    """Write detected segment crops and return a copy with crop paths filled."""
    out_dir.mkdir(parents=True, exist_ok=True)
    updated: dict[str, Segment] = {}
    sample_id = segmentation.sample_id or "row"
    for field, segment in segmentation.segments.items():
        crop = crop_segment(row_image, segment)
        if crop is None:
            updated[field] = segment
            continue
        path = out_dir / f"{sample_id}_{field}.png"
        cv2.imwrite(str(path), crop)
        updated[field] = Segment(
            field=segment.field,
            box=segment.box,
            confidence=segment.confidence,
            crop_path=path.as_posix(),
            failure_reason=segment.failure_reason,
        )
    return KillfeedSegmentation(
        sample_id=segmentation.sample_id,
        row_width=segmentation.row_width,
        row_height=segmentation.row_height,
        model_name=segmentation.model_name,
        model_version=segmentation.model_version,
        training_dataset=segmentation.training_dataset,
        confidence=segmentation.confidence,
        latency_ms=segmentation.latency_ms,
        fallback_used=segmentation.fallback_used,
        failure_reason=segmentation.failure_reason,
        segments=updated,
        evaluation_metrics=segmentation.evaluation_metrics,
    )
