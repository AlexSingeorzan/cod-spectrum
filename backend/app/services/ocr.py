from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..config import get_settings


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float
    boxes: list[dict[str, Any]] = field(default_factory=list)
    is_placeholder: bool = False


@runtime_checkable
class OcrEngine(Protocol):
    """Swappable OCR engine used by region extractors."""

    def read(self, image: np.ndarray, hints: dict | None = None) -> OcrResult: ...


class StubOcrEngine:
    """Deterministic fixture-backed OCR for offline tests and the sample flow."""

    def __init__(self, fixture_path: Path | None = None):
        path = fixture_path or get_settings().data_dir / "fixtures" / "sample_scores.json"
        self.readings = json.loads(path.read_text())["readings"] if path.exists() else []

    def read(self, image: np.ndarray, hints: dict | None = None) -> OcrResult:
        # TODO(model): Replace fixture lookup with an OCR model trained/calibrated on scorebar digits.
        hints = hints or {}
        index = int(hints.get("sample_index", 0))
        if self.readings:
            reading = self.readings[min(index, len(self.readings) - 1)]
            return OcrResult(
                text=f"{reading['score_a']} {reading['score_b']}",
                confidence=float(reading.get("confidence", 0.96)),
                is_placeholder=True,
            )
        return OcrResult(text="STUB 0 0", confidence=0.25, is_placeholder=True)


class TesseractOcrEngine:
    """Optional CPU OCR backend. Requires the tesseract binary and pytesseract extra."""

    def read(self, image: np.ndarray, hints: dict | None = None) -> OcrResult:
        # TODO(model): Calibrate preprocessing and confidence thresholds against labeled CDL scorebars.
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError("Tesseract backend requires: pip install pytesseract and the tesseract binary") from exc
        config = "--psm 7"
        whitelist = (hints or {}).get("whitelist")
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        words = [word for word in data["text"] if word.strip()]
        confidences = [float(value) / 100 for value in data["conf"] if float(value) >= 0]
        return OcrResult(text=" ".join(words), confidence=sum(confidences) / len(confidences) if confidences else 0.0)


def parse_score(result: OcrResult) -> tuple[int, int] | None:
    values = [int(value) for value in re.findall(r"\d+", result.text)]
    if len(values) < 2 or values[0] > 250 or values[1] > 250:
        return None
    return values[0], values[1]


def build_ocr_engine(name: str) -> OcrEngine:
    if name == "stub":
        return StubOcrEngine()
    if name == "tesseract":
        return TesseractOcrEngine()
    raise ValueError(f"unknown OCR engine: {name}")
