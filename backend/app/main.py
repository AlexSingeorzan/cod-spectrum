from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db import get_db, init_db
from .models import Broadcast, Clip, ProcessingJob, Report
from .routes.api import router
from .routes.sim import router as sim_router
from .services.event_store import count_game_events
from .ui import (
    COACH_REPORT_SLUG,
    render_dashboard,
    render_featured_coach_report,
    render_featured_coach_report_markdown,
    render_lab,
    render_match_report,
)


app = FastAPI(title="cod-spectrum", version="0.1.0", description="Evidence-backed CoD broadcast analytics")
app.include_router(router)
app.include_router(sim_router)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def dashboard(session: Session = Depends(get_db)) -> str:
    broadcasts = session.scalars(select(Broadcast).order_by(Broadcast.created_at.desc())).all()
    reports = session.scalars(select(Report).order_by(Report.created_at.desc())).all()
    jobs = session.scalars(select(ProcessingJob).order_by(ProcessingJob.id)).all()
    event_count = count_game_events(session)
    clip_count = session.scalar(select(func.count(Clip.id))) or 0
    return render_dashboard(broadcasts, reports, jobs, event_count=event_count, clip_count=clip_count)


@app.get("/lab", response_class=HTMLResponse)
def whatif_lab() -> str:
    return render_lab()


@app.get("/match/lat-van-hp", response_class=HTMLResponse)
def lat_van_match_report() -> str:
    return render_match_report()


@app.get(f"/coach-reports/{COACH_REPORT_SLUG}", response_class=HTMLResponse)
def featured_coach_report() -> str:
    return render_featured_coach_report()


@app.get(f"/coach-reports/{COACH_REPORT_SLUG}/download", response_class=PlainTextResponse)
def download_featured_coach_report() -> PlainTextResponse:
    return PlainTextResponse(
        render_featured_coach_report_markdown(),
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{COACH_REPORT_SLUG}-coach-brief.md"'},
    )
