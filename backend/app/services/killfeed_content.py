"""Killfeed content reader — Phase 4, deliverable 2.

The panel counter is the kill spine: it owns kill count and player totals. This
module is the enrichment layer for the killfeed row itself: attacker/victim text,
weapon icon class, headshot marker, and trade relationships.

The real LAT/VAN killfeed scaffold is currently unlabelled. That is intentional
and enforced here: no label means no read. Once ``data/killfeed_dataset/
annotations.jsonl`` contains human labels, this module can train a tiny
nearest-neighbour baseline and evaluate it honestly. Until then it only emits
events from labelled rows or deterministic synthetic fixtures.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..events import (
    DeathEvent,
    Evidence,
    GameEvent,
    KillEvent,
    Provenance,
    SourceKind,
    TradeEvent,
    WeaponEvent,
)

MODEL_NAME = "killfeed_content_knn"
MODEL_VERSION = "0.1.0"
IMAGE_SIZE = (192, 32)  # width, height
DEFAULT_CONFIDENCE_THRESHOLD = 0.60


@dataclass(frozen=True)
class KillfeedContentLabel:
    sample_id: str
    video_timestamp_seconds: float
    row_image: Path
    attacker: str
    attacker_team: str | None
    victim: str
    victim_team: str | None
    weapon: str | None
    headshot: bool | None
    is_trade: bool | None
    source_url: str | None
    label_source: str
    labeled_by: str


@dataclass(frozen=True)
class KillfeedContentRead:
    sample_id: str | None
    video_timestamp_seconds: float
    attacker: str | None
    attacker_team: str | None
    victim: str | None
    victim_team: str | None
    weapon: str | None
    headshot: bool | None
    is_trade: bool | None
    confidence: float
    source: str
    evidence_crop_path: str
    source_url: str | None = None
    labeled_by: str | None = None
    model_name: str | None = MODEL_NAME
    model_version: str | None = MODEL_VERSION

    @property
    def has_identity(self) -> bool:
        return bool(self.attacker and self.victim)

    @property
    def has_weapon(self) -> bool:
        return bool(self.weapon)


def default_dataset_dir() -> Path:
    return Path("data/killfeed_dataset")


def load_annotation_rows(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (dataset_dir / "annotations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return manifest, rows


def labeled_content_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows that are valid kills and contain enough human content to train/read."""
    out = []
    for row in rows:
        label = row.get("label", {})
        if label.get("valid_kill") is not True:
            continue
        if not (label.get("attacker") and label.get("victim")):
            continue
        if row.get("label_source") == "unlabeled" or not row.get("labeled_by"):
            continue
        out.append(row)
    return out


def label_from_row(dataset_dir: Path, row: dict[str, Any]) -> KillfeedContentLabel:
    label = row["label"]
    row_image = Path(row["row_image"])
    if not row_image.is_absolute():
        row_image = dataset_dir / row_image
    return KillfeedContentLabel(
        sample_id=str(row["id"]),
        video_timestamp_seconds=float(row["video_timestamp_seconds"]),
        row_image=row_image,
        attacker=str(label["attacker"]),
        attacker_team=label.get("attacker_team"),
        victim=str(label["victim"]),
        victim_team=label.get("victim_team"),
        weapon=label.get("weapon"),
        headshot=label.get("headshot"),
        is_trade=label.get("is_trade"),
        source_url=row.get("source_url"),
        label_source=str(row.get("label_source") or "manual_label"),
        labeled_by=str(row["labeled_by"]),
    )


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(path)


def load_labeled_content(dataset_dir: Path, exclude_sample_ids: set[str] | None = None) -> list[KillfeedContentLabel]:
    _manifest, rows = load_annotation_rows(dataset_dir)
    exclude_sample_ids = exclude_sample_ids or set()
    labels = []
    for row in labeled_content_rows(rows):
        if row["id"] in exclude_sample_ids:
            continue
        labels.append(label_from_row(dataset_dir, row))
    return labels


def _row_vector(image: np.ndarray) -> np.ndarray:
    """Small colour-aware row embedding for the baseline nearest-neighbour reader."""
    if image is None or image.size == 0:
        return np.zeros((IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3,), dtype=np.float32)
    resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] /= 179.0
    hsv[:, :, 1:] /= 255.0
    return hsv.reshape(-1)


def _confidence_from_distances(distances: np.ndarray, best_index: int) -> float:
    if len(distances) == 0:
        return 0.0
    best = float(distances[best_index])
    if len(distances) == 1:
        return 1.0
    others = np.delete(distances, best_index)
    nearest_other = float(others.min()) if len(others) else best + 1.0
    margin = (nearest_other - best) / max(nearest_other, 1e-6)
    return round(max(0.0, min(1.0, margin)), 4)


class KillfeedContentReader:
    """Nearest-neighbour baseline trained from labelled killfeed row crops.

    It intentionally abstains when there are no labels. The baseline is useful as
    a contract and regression harness before a real OCR/icon model replaces it.
    """

    name = MODEL_NAME
    model_version = MODEL_VERSION

    def __init__(
        self,
        dataset_dir: Path | str = default_dataset_dir(),
        *,
        exclude_sample_ids: set[str] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.confidence_threshold = confidence_threshold
        self.labels = load_labeled_content(self.dataset_dir, exclude_sample_ids=exclude_sample_ids)
        vectors = []
        kept_labels = []
        for label in self.labels:
            image = cv2.imread(str(label.row_image), cv2.IMREAD_COLOR)
            if image is None:
                continue
            vectors.append(_row_vector(image))
            kept_labels.append(label)
        self.labels = kept_labels
        self.vectors = np.stack(vectors) if vectors else np.zeros((0, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3), dtype=np.float32)

    @property
    def label_count(self) -> int:
        return len(self.labels)

    def read_image(
        self,
        image: np.ndarray,
        *,
        sample_id: str | None = None,
        video_timestamp_seconds: float = 0.0,
        evidence_crop_path: str = "",
        source_url: str | None = None,
    ) -> KillfeedContentRead | None:
        if self.label_count == 0:
            return None
        vector = _row_vector(image)
        distances = np.linalg.norm(self.vectors - vector, axis=1)
        best_index = int(np.argmin(distances))
        confidence = _confidence_from_distances(distances, best_index)
        if confidence < self.confidence_threshold:
            return None
        label = self.labels[best_index]
        return KillfeedContentRead(
            sample_id=sample_id,
            video_timestamp_seconds=video_timestamp_seconds,
            attacker=label.attacker,
            attacker_team=label.attacker_team,
            victim=label.victim,
            victim_team=label.victim_team,
            weapon=label.weapon,
            headshot=label.headshot,
            is_trade=label.is_trade,
            confidence=confidence,
            source="model",
            evidence_crop_path=evidence_crop_path,
            source_url=source_url,
            model_name=self.name,
            model_version=self.model_version,
        )

    def read_annotation(self, row: dict[str, Any]) -> KillfeedContentRead | None:
        image_path = Path(row["row_image"])
        if not image_path.is_absolute():
            image_path = self.dataset_dir / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        return self.read_image(
            image,
            sample_id=row["id"],
            video_timestamp_seconds=float(row["video_timestamp_seconds"]),
            evidence_crop_path=_display_path(image_path),
            source_url=row.get("source_url"),
        )


def manual_read_from_row(dataset_dir: Path, row: dict[str, Any]) -> KillfeedContentRead | None:
    """Convert a human-labelled row into a read without invoking a model."""
    if row not in labeled_content_rows([row]):
        return None
    label = label_from_row(dataset_dir, row)
    return KillfeedContentRead(
        sample_id=label.sample_id,
        video_timestamp_seconds=label.video_timestamp_seconds,
        attacker=label.attacker,
        attacker_team=label.attacker_team,
        victim=label.victim,
        victim_team=label.victim_team,
        weapon=label.weapon,
        headshot=label.headshot,
        is_trade=label.is_trade,
        confidence=1.0,
        source="manual_label",
        evidence_crop_path=_display_path(label.row_image),
        source_url=label.source_url,
        labeled_by=label.labeled_by,
        model_name=None,
        model_version=None,
    )


def _provenance_for_read(read: KillfeedContentRead, note: str) -> Provenance:
    if read.source == "manual_label":
        return Provenance(
            source=SourceKind.MANUAL_LABEL,
            producer="killfeed-content-reader",
            labeled_by=read.labeled_by,
            note=note,
        )
    return Provenance(
        source=SourceKind.MODEL,
        model_name=read.model_name or MODEL_NAME,
        model_version=read.model_version or MODEL_VERSION,
        producer="killfeed-content-reader",
        note=note,
    )


def _evidence_for_read(read: KillfeedContentRead) -> Evidence:
    return Evidence(
        video_timestamp_seconds=read.video_timestamp_seconds,
        crop_path=read.evidence_crop_path,
        source_url=read.source_url,
    )


def events_from_content_reads(reads: list[KillfeedContentRead], *, trade_window_seconds: float = 5.0) -> list[GameEvent]:
    """Turn chronological content reads into combat facts.

    TradeEvent is emitted only when the current kill clearly trades a recent
    teammate death: current attacker is on the previous victim's team and kills
    the previous attacker inside ``trade_window_seconds``.
    """
    events: list[GameEvent] = []
    recent_kills: list[GameEvent] = []

    for read in sorted((r for r in reads if r.has_identity), key=lambda r: r.video_timestamp_seconds):
        evidence = _evidence_for_read(read)
        provenance = _provenance_for_read(
            read,
            "killfeed row content read: attacker/victim/weapon labels from row crop",
        )
        kill = GameEvent(
            video_timestamp_seconds=read.video_timestamp_seconds,
            confidence=read.confidence,
            provenance=provenance,
            evidence=evidence,
            payload=KillEvent(
                attacker=read.attacker,
                attacker_team=read.attacker_team,
                victim=read.victim,
                victim_team=read.victim_team,
                weapon=read.weapon,
                headshot=read.headshot,
                is_trade=read.is_trade,
                attributes={"source": "killfeed_content", "sample_id": read.sample_id},
            ),
            tags=["killfeed", "content_read"],
        )
        events.append(kill)

        events.append(
            GameEvent(
                video_timestamp_seconds=read.video_timestamp_seconds,
                confidence=read.confidence,
                provenance=provenance,
                evidence=evidence,
                payload=DeathEvent(
                    player=read.victim or "",
                    team=read.victim_team,
                    killer=read.attacker,
                    weapon=read.weapon,
                    attributes={"source": "killfeed_content", "sample_id": read.sample_id},
                ),
                tags=["killfeed", "content_read"],
            )
        )

        if read.has_weapon:
            events.append(
                GameEvent(
                    video_timestamp_seconds=read.video_timestamp_seconds,
                    confidence=read.confidence,
                    provenance=provenance,
                    evidence=evidence,
                    payload=WeaponEvent(
                        player=read.attacker or "",
                        team=read.attacker_team,
                        weapon=read.weapon or "",
                        action="use",
                        attributes={"source": "killfeed_content", "sample_id": read.sample_id},
                    ),
                    tags=["killfeed", "weapon_read"],
                )
            )

        if read.is_trade:
            trade = _trade_event_for_read(read, kill, recent_kills, provenance, evidence, trade_window_seconds)
            if trade is not None:
                events.append(trade)

        recent_kills.append(kill)
        recent_kills = [
            event
            for event in recent_kills
            if read.video_timestamp_seconds - event.video_timestamp_seconds <= trade_window_seconds
        ]

    return events


def _trade_event_for_read(
    read: KillfeedContentRead,
    kill: GameEvent,
    recent_kills: list[GameEvent],
    provenance: Provenance,
    evidence: Evidence,
    trade_window_seconds: float,
) -> GameEvent | None:
    for previous in reversed(recent_kills):
        payload = previous.payload
        if not isinstance(payload, KillEvent):
            continue
        if payload.attacker == read.victim and payload.victim_team == read.attacker_team:
            dt = read.video_timestamp_seconds - previous.video_timestamp_seconds
            if 0 <= dt <= trade_window_seconds:
                return GameEvent(
                    video_timestamp_seconds=read.video_timestamp_seconds,
                    confidence=read.confidence,
                    provenance=provenance,
                    evidence=evidence,
                    payload=TradeEvent(
                        dead_player=payload.victim or "",
                        dead_team=payload.victim_team,
                        trading_player=read.attacker or "",
                        trading_team=read.attacker_team,
                        trade_window_seconds=round(dt, 3),
                        original_kill_event_id=previous.event_id,
                        trade_kill_event_id=kill.event_id,
                        attributes={"source": "killfeed_content", "sample_id": read.sample_id},
                    ),
                    tags=["killfeed", "trade_read"],
                )
    return None
