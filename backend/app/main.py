from __future__ import annotations

import html

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import get_db, init_db
from .models import Broadcast, ProcessingJob, Report
from .routes.api import router


app = FastAPI(title="cod-spectrum", version="0.1.0", description="Evidence-backed CoD broadcast analytics")
app.include_router(router)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def dashboard(session: Session = Depends(get_db)) -> str:
    broadcasts = session.scalars(select(Broadcast).order_by(Broadcast.created_at.desc())).all()
    rows: list[str] = []
    for broadcast in broadcasts:
        report = session.scalar(select(Report).where(Report.broadcast_id == broadcast.id))
        jobs = session.scalars(select(ProcessingJob).where(ProcessingJob.broadcast_id == broadcast.id)).all()
        job_status = ", ".join(f"{job.stage}:{job.status}" for job in jobs) or "not queued"
        report_link = f'<a href="/reports/{report.id}">view report ({report.data_confidence:.1%})</a>' if report else "—"
        rows.append(
            f"<tr><td><a href='/broadcasts/{broadcast.id}'>{broadcast.id}</a></td><td>{html.escape(broadcast.title)}</td>"
            f"<td><span class='status'>{broadcast.status.value}</span></td><td>{html.escape(job_status)}</td><td>{report_link}</td></tr>"
        )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>cod-spectrum</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem;background:#f7f8fa;color:#17202a}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:.75rem;border-bottom:1px solid #ddd;text-align:left}}.status{{font-family:monospace}}a{{color:#075bb5}}</style></head>
<body><h1>cod-spectrum</h1><p>Always-on, evidence-backed broadcast analytics.</p>
<table><thead><tr><th>ID</th><th>Broadcast</th><th>Status</th><th>Jobs</th><th>Report</th></tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>"""
