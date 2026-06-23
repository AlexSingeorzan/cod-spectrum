from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yt_dlp


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveredVideo:
    platform: str
    video_id: str
    title: str
    url: str
    local_path: str | None = None


class SourceDiscoverer(Protocol):
    """Discovers recent VOD metadata without requiring a video download."""

    def discover(self, source: dict[str, Any]) -> list[DiscoveredVideo]: ...


class YtDlpDiscoverer:
    def discover(self, source: dict[str, Any]) -> list[DiscoveredVideo]:
        if source["platform"] == "local" or source["type"] == "local":
            path = Path(source["url"]).resolve()
            if not path.exists():
                logger.warning("local source missing: %s", path)
                return []
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            video_id = digest.hexdigest()
            return [DiscoveredVideo("local", video_id, path.stem.replace("_", " ").title(), str(path), str(path))]
        options = {
            "quiet": True, "no_warnings": True, "extract_flat": "in_playlist",
            "playlistend": 10, "skip_download": True, "sleep_interval_requests": 1,
        }
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(source["url"], download=False)
        entries = info.get("entries") or [info]
        return [
            DiscoveredVideo(
                platform=source["platform"], video_id=str(entry["id"]), title=entry.get("title") or str(entry["id"]),
                url=entry.get("webpage_url") or entry.get("url") or source["url"],
            )
            for entry in entries if entry and entry.get("id")
        ]
