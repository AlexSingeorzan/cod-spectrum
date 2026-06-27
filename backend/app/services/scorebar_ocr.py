"""CDL broadcast scorebar OCR: evaluated CPU baseline.

This is Phase 3's honest scorebar OCR baseline. It is data-driven, CPU-only, and
intentionally limited:

* labels come from human-verified scorebar crops;
* digits are segmented with classical CV;
* a k-NN digit gallery reads candidate score hypotheses;
* chronological reads use Hardpoint monotonicity as a temporal prior;
* confidence is capped by the latest leave-one-crop-out evaluation result.

It is not the default OCR engine. Select it with ``--ocr-engine cdl`` after
building/evaluating the dataset. The stub remains available for deterministic
offline demos.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import get_settings
from .ocr import OcrResult

MODEL_NAME = "cdl_scorebar_knn"
MODEL_VERSION = "0.1.0-knn"

# Sub-regions inside a wide CDL scorebar crop, tuned to the verified LAT/VAN
# evidence crops. The default HUD profile's scorebar crop is narrower; production
# use should calibrate the HUD region so the full bar geometry matches this shape.
LEFT_X = (0.255, 0.385)
RIGHT_X = (0.610, 0.745)
Y_BAND = (0.430, 0.840)

GLYPH = 28
INNER = 20
DEFAULT_CONFIDENCE_CEILING = 0.50


@dataclass(frozen=True)
class ScorebarSample:
    sample_id: str
    timestamp_seconds: float
    image_path: Path
    score_a: int
    score_b: int
    source_url: str
    label_source: str
    labeled_by: str


@dataclass(frozen=True)
class ScoreCandidate:
    value: int
    text: str
    confidence: float
    digit_confidences: list[float]


@dataclass(frozen=True)
class ScorePairCandidate:
    score_a: int
    score_b: int
    text: str
    confidence: float
    left: ScoreCandidate
    right: ScoreCandidate


def default_dataset_dir() -> Path:
    return get_settings().data_dir / "fixtures" / "scorebar_ocr" / "lat_van_hp"


def region(img: np.ndarray, side: str) -> np.ndarray:
    h, w = img.shape[:2]
    xr = LEFT_X if side == "left" else RIGHT_X
    x0, x1 = int(xr[0] * w), int(xr[1] * w)
    y0, y1 = int(Y_BAND[0] * h), int(Y_BAND[1] * h)
    return img[y0:y1, x0:x1]


def white_mask(roi: np.ndarray) -> np.ndarray:
    """Binary mask of white/grey digit ink, dropping saturated team logos."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _h, saturation, value = cv2.split(hsv)
    threshold = max(130, int(np.percentile(value, 85)) - 20)
    mask = ((saturation < 75) & (value > threshold)).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def digit_extent(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x0, x1, y0, y1) around tall digit ink, ignoring small specks."""
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    comps = [stats[i] for i in range(1, n)]
    max_h = max(int(c[cv2.CC_STAT_HEIGHT]) for c in comps)
    tall = [
        c for c in comps
        if int(c[cv2.CC_STAT_HEIGHT]) > 0.5 * max_h and int(c[cv2.CC_STAT_AREA]) > 8
    ]
    if not tall:
        return None
    x0 = min(int(c[cv2.CC_STAT_LEFT]) for c in tall)
    x1 = max(int(c[cv2.CC_STAT_LEFT] + c[cv2.CC_STAT_WIDTH]) for c in tall)
    y0 = min(int(c[cv2.CC_STAT_TOP]) for c in tall)
    y1 = max(int(c[cv2.CC_STAT_TOP] + c[cv2.CC_STAT_HEIGHT]) for c in tall)
    return x0, x1, y0, y1


def normalise_glyph(cell_mask: np.ndarray) -> np.ndarray | None:
    """Tighten to glyph ink, scale into INNER, centre in a GLYPH square."""
    ys, xs = np.where(cell_mask > 0)
    if len(xs) < 6:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    glyph = cell_mask[y0:y1, x0:x1]
    h, w = glyph.shape
    scale = INNER / max(h, w)
    resized = cv2.resize(
        glyph,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((GLYPH, GLYPH), np.float32)
    rh, rw = resized.shape
    oy, ox = (GLYPH - rh) // 2, (GLYPH - rw) // 2
    canvas[oy:oy + rh, ox:ox + rw] = (resized > 0).astype(np.float32)
    return canvas


def slice_digits(mask: np.ndarray, n: int) -> list[np.ndarray]:
    """Slice the score block into n cells and normalise each cell's glyph."""
    extent = digit_extent(mask)
    if extent is None or n < 1:
        return []
    x0, x1, y0, y1 = extent
    block = mask[y0:y1, x0:x1]
    width = block.shape[1]
    glyphs: list[np.ndarray] = []
    for i in range(n):
        a, b = i * width // n, (i + 1) * width // n
        glyph = normalise_glyph(block[:, a:b])
        if glyph is not None:
            glyphs.append(glyph)
    return glyphs


def side_present(roi: np.ndarray) -> bool:
    mask = white_mask(roi)
    if mask.mean() / 255 < 0.015:
        return False
    return digit_extent(mask) is not None


def scorebar_present(img: np.ndarray) -> bool:
    return side_present(region(img, "left")) and side_present(region(img, "right"))


@dataclass
class DigitClassifier:
    """1-NN gallery over normalised digit glyphs."""

    vectors: np.ndarray
    labels: np.ndarray

    @classmethod
    def from_templates(cls, templates: list[tuple[int, np.ndarray]]) -> "DigitClassifier":
        if not templates:
            return cls(np.zeros((0, GLYPH * GLYPH), np.float32), np.zeros((0,), dtype=int))
        vectors = np.stack([glyph.reshape(-1) for _label, glyph in templates])
        labels = np.array([label for label, _glyph in templates], dtype=int)
        return cls(vectors, labels)

    def classify(self, glyph: np.ndarray) -> tuple[int, float]:
        if len(self.vectors) == 0:
            return 0, 0.0
        v = glyph.reshape(-1)
        distances = np.linalg.norm(self.vectors - v, axis=1)
        order = np.argsort(distances)
        best = int(self.labels[order[0]])
        diff = distances[self.labels != best]
        nearest_other = float(diff.min()) if len(diff) else float(distances[order[0]] + 1.0)
        best_distance = float(distances[order[0]])
        margin = nearest_other / (best_distance + nearest_other + 1e-6)
        return best, max(0.0, min(1.0, (margin - 0.5) * 2.0))


def load_manifest(dataset_dir: Path | None = None) -> tuple[dict[str, Any], list[ScorebarSample]]:
    dataset_dir = dataset_dir or default_dataset_dir()
    path = dataset_dir / "manifest.json"
    payload = json.loads(path.read_text())
    samples: list[ScorebarSample] = []
    for row in payload.get("samples", []):
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = dataset_dir / image_path
        samples.append(
            ScorebarSample(
                sample_id=str(row["sample_id"]),
                timestamp_seconds=float(row["timestamp_seconds"]),
                image_path=image_path,
                score_a=int(row["score_a"]),
                score_b=int(row["score_b"]),
                source_url=str(row["source_url"]),
                label_source=str(row["label_source"]),
                labeled_by=str(row["labeled_by"]),
            )
        )
    return payload, samples


def load_templates(dataset_dir: Path | None = None, exclude_sample_ids: set[str] | None = None) -> list[tuple[int, np.ndarray]]:
    dataset_dir = dataset_dir or default_dataset_dir()
    manifest = dataset_dir / "digits.jsonl"
    templates: list[tuple[int, np.ndarray]] = []
    if not manifest.exists():
        return templates
    exclude_sample_ids = exclude_sample_ids or set()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("sample_id") in exclude_sample_ids:
            continue
        image_path = dataset_dir / row["path"]
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates.append((int(row["label"]), (img > 127).astype(np.float32)))
    return templates


def load_confidence_ceiling(dataset_dir: Path | None = None) -> float:
    dataset_dir = dataset_dir or default_dataset_dir()
    result_path = dataset_dir / "eval_results.json"
    if not result_path.exists():
        return DEFAULT_CONFIDENCE_CEILING
    try:
        payload = json.loads(result_path.read_text())
        accuracy = float(payload["metrics"]["leave_one_out"]["score_exact_accuracy"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return DEFAULT_CONFIDENCE_CEILING
    return max(0.05, min(1.0, accuracy))


def side_candidates(classifier: DigitClassifier, roi: np.ndarray, target: int = 250) -> list[ScoreCandidate]:
    mask = white_mask(roi)
    candidates: list[ScoreCandidate] = []
    for n in (1, 2, 3):
        glyphs = slice_digits(mask, n)
        if len(glyphs) != n:
            continue
        digits: list[str] = []
        confidences: list[float] = []
        for glyph in glyphs:
            digit, confidence = classifier.classify(glyph)
            digits.append(str(digit))
            confidences.append(confidence)
        value = int("".join(digits))
        if value <= target:
            candidates.append(
                ScoreCandidate(
                    value=value,
                    text="".join(digits),
                    confidence=float(np.mean(confidences)) if confidences else 0.0,
                    digit_confidences=confidences,
                )
            )
    return sorted(candidates, key=lambda c: c.confidence, reverse=True)


def score_pair_candidates(
    classifier: DigitClassifier,
    img: np.ndarray,
    target: int = 250,
    limit: int = 20,
) -> list[ScorePairCandidate]:
    if not scorebar_present(img):
        return []
    left = side_candidates(classifier, region(img, "left"), target)
    right = side_candidates(classifier, region(img, "right"), target)
    candidates: list[ScorePairCandidate] = []
    for left_candidate in left:
        for right_candidate in right:
            confidence = (left_candidate.confidence + right_candidate.confidence) / 2.0
            candidates.append(
                ScorePairCandidate(
                    score_a=left_candidate.value,
                    score_b=right_candidate.value,
                    text=f"{left_candidate.text} {right_candidate.text}",
                    confidence=confidence,
                    left=left_candidate,
                    right=right_candidate,
                )
            )
    return sorted(candidates, key=lambda c: c.confidence, reverse=True)[:limit]


class CdlScorebarOcrEngine:
    """Stateful scorebar OCR engine for chronological Hardpoint reads."""

    name = MODEL_NAME
    model_version = MODEL_VERSION

    def __init__(self, dataset_dir: Path | None = None, temporal: bool = True):
        self.dataset_dir = dataset_dir or default_dataset_dir()
        self.templates = load_templates(self.dataset_dir)
        self.classifier = DigitClassifier.from_templates(self.templates)
        self.confidence_ceiling = load_confidence_ceiling(self.dataset_dir)
        self.temporal = temporal
        self.previous_score: tuple[int, int] | None = None

    def reset(self) -> None:
        self.previous_score = None

    def _choose(self, candidates: list[ScorePairCandidate], target: int) -> ScorePairCandidate | None:
        if not candidates:
            return None
        if not self.temporal or self.previous_score is None:
            return candidates[0]

        prev_a, prev_b = self.previous_score
        valid: list[tuple[float, ScorePairCandidate]] = []
        for candidate in candidates:
            da = candidate.score_a - prev_a
            db = candidate.score_b - prev_b
            if da < 0 or db < 0:
                continue
            if candidate.score_a > target or candidate.score_b > target:
                continue
            if da > 45 or db > 45 or da + db > 55:
                continue
            temporal_score = 1.0 - min(1.0, abs((da + db) - 16) / 45.0) * 0.10
            valid.append((candidate.confidence * temporal_score, candidate))
        if valid:
            return max(valid, key=lambda item: item[0])[1]
        return candidates[0]

    def read(self, image: np.ndarray, hints: dict | None = None) -> OcrResult:
        hints = hints or {}
        target = int(hints.get("target", get_settings().hardpoint_target))
        if not self.templates:
            return OcrResult(text="", confidence=0.0, is_placeholder=True)

        candidates = score_pair_candidates(self.classifier, image, target=target)
        chosen = self._choose(candidates, target)
        boxes = [
            {
                "model_name": self.name,
                "model_version": self.model_version,
                "candidate_count": len(candidates),
                "confidence_ceiling": round(self.confidence_ceiling, 4),
                "temporal_previous_score": list(self.previous_score) if self.previous_score else None,
                "top_candidates": [
                    {
                        "text": candidate.text,
                        "score_a": candidate.score_a,
                        "score_b": candidate.score_b,
                        "confidence": round(candidate.confidence, 4),
                    }
                    for candidate in candidates[:5]
                ],
            }
        ]
        if chosen is None:
            return OcrResult(text="", confidence=0.0, boxes=boxes, is_placeholder=True)

        self.previous_score = (chosen.score_a, chosen.score_b)
        confidence = round(min(chosen.confidence, self.confidence_ceiling), 4)
        return OcrResult(
            text=f"{chosen.score_a} {chosen.score_b}",
            confidence=confidence,
            boxes=boxes,
            is_placeholder=False,
        )
