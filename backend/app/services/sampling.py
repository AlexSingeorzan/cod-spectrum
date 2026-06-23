from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .hud import HudProfile, crop_region


@dataclass(frozen=True)
class FrameSample:
    index: int
    timestamp_seconds: float
    frame: np.ndarray


def iter_video_samples(path: Path, sample_fps: float = 1.0) -> Iterator[FrameSample]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps
    index = 0
    timestamp = 0.0
    try:
        while timestamp < duration:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                break
            yield FrameSample(index=index, timestamp_seconds=timestamp, frame=frame)
            index += 1
            timestamp = index / sample_fps
    finally:
        capture.release()


def crop_change_score(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None or previous.shape != current.shape:
        return 255.0
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(previous_gray, current_gray)))


def dump_debug_crops(sample: FrameSample, profile: HudProfile, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, region in profile.regions.items():
        path = output_dir / f"sample_{sample.index:05d}_{name}.jpg"
        cv2.imwrite(str(path), crop_region(sample.frame, region))
        paths.append(path)
    return paths

