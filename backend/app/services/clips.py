from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def generate_clip(video_path: Path, output_path: Path, timestamp_seconds: float, before: float = 3.0, after: float = 4.0) -> tuple[float, float]:
    """Generate a CPU-friendly MP4 around a moment and return the actual bounds."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to generate clips")
    start = max(0.0, timestamp_seconds - before)
    end = max(start + 0.5, timestamp_seconds + after)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{end - start:.3f}",
        "-an", "-c:v", "mpeg4", "-q:v", "5", "-movflags", "+faststart", str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg clip generation failed: {completed.stderr.strip()}")
    return start, end

