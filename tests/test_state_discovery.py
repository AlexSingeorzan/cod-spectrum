from __future__ import annotations

import pytest
from sqlalchemy import func, select

from backend.app.models import Broadcast, BroadcastStatus, Source
from backend.app.sources.discovery import DiscoveredVideo
from backend.app.state import transition_broadcast
from backend.app.workers.scheduler import register_discovered_video


def test_legal_and_illegal_status_transitions():
    broadcast = Broadcast(platform="local", video_id="x", title="x", status=BroadcastStatus.discovered)
    transition_broadcast(broadcast, BroadcastStatus.processing)
    assert broadcast.status == BroadcastStatus.processing
    transition_broadcast(broadcast, BroadcastStatus.processed)
    with pytest.raises(ValueError):
        transition_broadcast(broadcast, BroadcastStatus.processing)


def test_source_discovery_deduplicates_platform_and_video_id(session):
    source = Source(name="fixture", platform="local", source_type="local", url="fixture.mp4")
    session.add(source)
    session.flush()
    video = DiscoveredVideo("local", "abc123", "Fixture", "fixture.mp4", "fixture.mp4")
    first, first_created = register_discovered_video(session, source, video)
    second, second_created = register_discovered_video(session, source, video)
    assert first.id == second.id
    assert first_created is True
    assert second_created is False
    assert session.scalar(select(func.count(Broadcast.id))) == 1

