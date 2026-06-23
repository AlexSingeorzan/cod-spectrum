from __future__ import annotations

from backend.app.models import Broadcast, BroadcastStatus, ProcessingJob, Report
from backend.app.ui import render_dashboard, render_featured_coach_report, render_featured_coach_report_markdown


def test_dashboard_renders_live_metrics_and_escapes_titles():
    broadcast = Broadcast(
        id=7, platform="local", video_id="sample", title="A < B", status=BroadcastStatus.processed,
    )
    report = Report(id=3, broadcast_id=7, json_path="x", markdown_path="x", html_path="x", data_confidence=0.9)
    job = ProcessingJob(broadcast_id=7, stage="report", status="completed")

    page = render_dashboard([broadcast], [report], [job], event_count=12, clip_count=3)

    assert "Spectrum · Match intelligence" in page
    assert "A &lt; B" in page
    assert "A < B" not in page
    assert "Evidence events</span><strong>12" in page
    assert "/reports/3" in page


def test_featured_coach_report_states_sources_and_limits():
    page = render_featured_coach_report()
    markdown = render_featured_coach_report_markdown()

    assert "LAT controlled the series" in page
    assert "High confidence on outcomes" in page
    assert "fieldlevelmedia.com" in page
    assert "Tactical causes: 58%" in markdown
    assert "Hacienda Search & Destroy" in markdown
