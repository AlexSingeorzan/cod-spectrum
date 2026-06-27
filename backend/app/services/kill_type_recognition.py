"""Coarse kill-type recognition from killfeed icon crops.

This module deliberately classifies broad kill causes instead of exact weapons.
That is the right abstraction for the current annotation budget and for downstream
analytics: the stable field is ``kill_type``. A later exact-weapon classifier can
fill optional ``weapon`` metadata without changing the event schema.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from ..events import Evidence, GameEvent, KillType, Provenance, SourceKind, WeaponEvent

MODEL_VERSION = "0.2.0"
IMAGE_SIZE = (48, 32)  # width, height
DEFAULT_CONFIDENCE_THRESHOLD = 0.62
KILL_TYPE_CATEGORIES: tuple[KillType, ...] = (
    "gun",
    "grenade",
    "melee",
    "fall_damage",
    "suicide",
    "environment",
    "objective",
    "killstreak",
    "unknown",
)
Approach = Literal["template", "histogram"]

MODEL_NAMES: dict[Approach, str] = {
    "template": "kill_type_icon_template_nn",
    "histogram": "kill_type_icon_histogram_nn",
}
EVALUATION_METRICS = {
    "real_labelled_accuracy": None,
    "reason": "real kill-type icon crops are not labelled yet; eval reports readiness only",
}


@dataclass(frozen=True)
class KillTypeLabel:
    sample_id: str
    kill_type: KillType
    image_path: Path
    video_timestamp_seconds: float
    source_row_image: str | None
    segment_box: list[float] | None
    segment_confidence: float | None
    source_url: str | None
    label_source: str
    labeled_by: str
    exact_weapon: str | None = None


@dataclass(frozen=True)
class KillTypePrediction:
    sample_id: str | None
    kill_type: KillType | None
    confidence: float
    latency_ms: float
    model_name: str
    model_version: str
    training_dataset: str
    failure_reason: str | None
    fallback_used: bool
    evaluation_metrics: dict[str, Any]
    evidence_crop_path: str
    evidence_box: list[float] | None
    video_timestamp_seconds: float
    source_url: str | None = None
    exact_weapon: str | None = None
    top_candidates: list[dict[str, Any]] | None = None

    @property
    def accepted(self) -> bool:
        return self.kill_type is not None and self.failure_reason is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "kill_type": self.kill_type,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 3),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "training_dataset": self.training_dataset,
            "failure_reason": self.failure_reason,
            "fallback_used": self.fallback_used,
            "evaluation_metrics": self.evaluation_metrics,
            "evidence_crop_path": self.evidence_crop_path,
            "evidence_box": self.evidence_box,
            "video_timestamp_seconds": self.video_timestamp_seconds,
            "source_url": self.source_url,
            "exact_weapon": self.exact_weapon,
            "top_candidates": self.top_candidates or [],
        }


def default_dataset_dir() -> Path:
    return Path("data/kill_type_dataset")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(path)


def load_kill_type_rows(dataset_dir: Path | str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = Path(dataset_dir)
    manifest = json.loads((dataset / "manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (dataset / "annotations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return manifest, rows


def labeled_kill_type_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labelled = []
    for row in rows:
        label = row.get("label", {})
        kill_type = label.get("kill_type")
        if not kill_type:
            continue
        if kill_type not in KILL_TYPE_CATEGORIES:
            continue
        if label.get("valid_kill_type") is not True or label.get("unclear") is True:
            continue
        if row.get("label_source") == "unlabeled" or not row.get("labeled_by"):
            continue
        labelled.append(row)
    return labelled


def kill_type_label_from_row(dataset_dir: Path, row: dict[str, Any]) -> KillTypeLabel:
    label = row["label"]
    image_path = Path(row["icon_image"])
    if not image_path.is_absolute():
        image_path = dataset_dir / image_path
    return KillTypeLabel(
        sample_id=str(row["id"]),
        kill_type=label["kill_type"],
        image_path=image_path,
        video_timestamp_seconds=float(row.get("video_timestamp_seconds", 0.0)),
        source_row_image=row.get("source_row_image"),
        segment_box=row.get("segment_box"),
        segment_confidence=row.get("segment_confidence"),
        source_url=row.get("source_url"),
        label_source=str(row.get("label_source") or "manual_label"),
        labeled_by=str(row["labeled_by"]),
        exact_weapon=label.get("exact_weapon"),
    )


def load_kill_type_labels(
    dataset_dir: Path | str,
    exclude_sample_ids: set[str] | None = None,
) -> list[KillTypeLabel]:
    dataset = Path(dataset_dir)
    _manifest, rows = load_kill_type_rows(dataset)
    exclude_sample_ids = exclude_sample_ids or set()
    labels = []
    for row in labeled_kill_type_rows(rows):
        if row["id"] in exclude_sample_ids:
            continue
        labels.append(kill_type_label_from_row(dataset, row))
    return labels


def _template_feature(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((IMAGE_SIZE[0] * IMAGE_SIZE[1],), dtype=np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    if resized.max() > resized.min():
        resized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
    return (resized.astype(np.float32) / 255.0).reshape(-1)


def _histogram_feature(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((8 + 4 + 4,), dtype=np.float32)
    resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).reshape(-1)
    s_hist = cv2.calcHist([hsv], [1], None, [4], [0, 256]).reshape(-1)
    v_hist = cv2.calcHist([hsv], [2], None, [4], [0, 256]).reshape(-1)
    feature = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    return feature / norm if norm else feature


def _feature(image: np.ndarray, approach: Approach) -> np.ndarray:
    if approach == "histogram":
        return _histogram_feature(image)
    return _template_feature(image)


def _confidence(distances: np.ndarray, labels: np.ndarray, best_index: int) -> float:
    if len(distances) == 0:
        return 0.0
    best = float(distances[best_index])
    different = distances[labels != labels[best_index]]
    if len(different) == 0:
        if best <= 1e-6:
            return 1.0
        return max(0.0, min(1.0, 1.0 / (1.0 + best)))
    nearest_other = float(different.min())
    if nearest_other <= 1e-6:
        return 0.0
    return round(max(0.0, min(1.0, (nearest_other - best) / nearest_other)), 4)


class KillTypeRecognizer:
    """Nearest-neighbour baseline for broad kill-type icon classification."""

    model_version = MODEL_VERSION

    def __init__(
        self,
        dataset_dir: Path | str = default_dataset_dir(),
        *,
        approach: Approach = "template",
        exclude_sample_ids: set[str] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.approach: Approach = approach
        self.name = MODEL_NAMES[approach]
        self.confidence_threshold = confidence_threshold
        self.manifest, _rows = load_kill_type_rows(self.dataset_dir)
        self.training_dataset = str(self.manifest.get("dataset_id") or self.dataset_dir)
        self.evaluation_metrics = self.manifest.get("evaluation_metrics") or EVALUATION_METRICS
        self.labels = load_kill_type_labels(self.dataset_dir, exclude_sample_ids=exclude_sample_ids)

        vectors: list[np.ndarray] = []
        kept: list[KillTypeLabel] = []
        for label in self.labels:
            image = cv2.imread(str(label.image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            vectors.append(_feature(image, self.approach))
            kept.append(label)
        self.labels = kept
        shape = _feature(np.zeros((1, 1, 3), np.uint8), self.approach).shape[0]
        self.vectors = np.stack(vectors) if vectors else np.zeros((0, shape), dtype=np.float32)
        self.label_values = np.array([label.kill_type for label in self.labels])

    @property
    def label_count(self) -> int:
        return len(self.labels)

    @property
    def classes(self) -> list[str]:
        return sorted(set(str(label.kill_type) for label in self.labels))

    def read_image(
        self,
        image: np.ndarray,
        *,
        sample_id: str | None = None,
        video_timestamp_seconds: float = 0.0,
        evidence_crop_path: str = "",
        evidence_box: list[float] | None = None,
        source_url: str | None = None,
    ) -> KillTypePrediction:
        start = time.perf_counter()
        if self.label_count == 0:
            return self._prediction(
                start, sample_id, None, 0.0, "no_labeled_kill_type_templates",
                video_timestamp_seconds, evidence_crop_path, evidence_box, source_url, []
            )
        if image is None or image.size == 0:
            return self._prediction(
                start, sample_id, None, 0.0, "kill_type_crop_unreadable",
                video_timestamp_seconds, evidence_crop_path, evidence_box, source_url, []
            )

        vector = _feature(image, self.approach)
        distances = np.linalg.norm(self.vectors - vector, axis=1)
        order = np.argsort(distances)
        best_index = int(order[0])
        kill_type = str(self.label_values[best_index])
        confidence = _confidence(distances, self.label_values, best_index)
        top = [
            {
                "kill_type": str(self.label_values[int(idx)]),
                "distance": round(float(distances[int(idx)]), 4),
            }
            for idx in order[:5]
        ]
        if confidence < self.confidence_threshold:
            return self._prediction(
                start, sample_id, None, confidence, "below_confidence_threshold",
                video_timestamp_seconds, evidence_crop_path, evidence_box, source_url, top
            )
        return self._prediction(
            start, sample_id, kill_type, confidence, None,
            video_timestamp_seconds, evidence_crop_path, evidence_box, source_url, top
        )

    def read_row(self, row: dict[str, Any]) -> KillTypePrediction:
        image_path = Path(row["icon_image"])
        if not image_path.is_absolute():
            image_path = self.dataset_dir / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        return self.read_image(
            image,
            sample_id=row["id"],
            video_timestamp_seconds=float(row.get("video_timestamp_seconds", 0.0)),
            evidence_crop_path=_display_path(image_path),
            evidence_box=row.get("segment_box"),
            source_url=row.get("source_url"),
        )

    def _prediction(
        self,
        start: float,
        sample_id: str | None,
        kill_type: KillType | None,
        confidence: float,
        failure_reason: str | None,
        video_timestamp_seconds: float,
        evidence_crop_path: str,
        evidence_box: list[float] | None,
        source_url: str | None,
        top_candidates: list[dict[str, Any]],
    ) -> KillTypePrediction:
        return KillTypePrediction(
            sample_id=sample_id,
            kill_type=kill_type,
            confidence=round(confidence, 4),
            latency_ms=(time.perf_counter() - start) * 1000,
            model_name=self.name,
            model_version=self.model_version,
            training_dataset=self.training_dataset,
            failure_reason=failure_reason,
            fallback_used=False,
            evaluation_metrics=self.evaluation_metrics,
            evidence_crop_path=evidence_crop_path,
            evidence_box=evidence_box,
            video_timestamp_seconds=video_timestamp_seconds,
            source_url=source_url,
            top_candidates=top_candidates,
        )


def kill_type_event_from_prediction(
    prediction: KillTypePrediction,
    *,
    player: str | None = None,
    team: str | None = None,
    action: Literal["pickup", "swap", "use"] = "use",
) -> GameEvent | None:
    """Create a WeaponEvent carrying the stable kill_type field.

    ``WeaponEvent`` is retained as the existing combat-icon event type, but
    analytics should read ``payload.kill_type``. ``payload.weapon`` is optional
    future metadata for exact weapon classifiers.
    """
    if not prediction.accepted:
        return None
    return GameEvent(
        video_timestamp_seconds=prediction.video_timestamp_seconds,
        confidence=prediction.confidence,
        provenance=Provenance(
            source=SourceKind.MODEL,
            model_name=prediction.model_name,
            model_version=prediction.model_version,
            producer="kill-type-recognizer",
            note="coarse killfeed icon classifier; exact weapon remains optional metadata",
        ),
        evidence=Evidence(
            video_timestamp_seconds=prediction.video_timestamp_seconds,
            crop_path=prediction.evidence_crop_path,
            source_url=prediction.source_url,
        ),
        payload=WeaponEvent(
            player=player,
            team=team,
            kill_type=prediction.kill_type,
            weapon=prediction.exact_weapon,
            action=action,
            attributes={
                "source": "kill_type_recognition",
                "sample_id": prediction.sample_id,
                "evidence_box": prediction.evidence_box,
                "training_dataset": prediction.training_dataset,
                "top_candidates": prediction.top_candidates or [],
            },
        ),
        tags=["kill_type_recognition", "killfeed"],
    )
