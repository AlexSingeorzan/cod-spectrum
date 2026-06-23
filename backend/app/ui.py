from __future__ import annotations

import html
import json
from collections.abc import Iterable
from typing import Any

from .models import Broadcast, ProcessingJob, Report
from .services import simulation as sim


COACH_REPORT_SLUG = "lat-vancouver-2026-06-20"


def _icon(name: str, size: int = 18) -> str:
    paths = {
        "grid": '<rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/>',
        "report": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h6"/>',
        "film": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="m10 9 5 3-5 3Z"/>',
        "pulse": '<path d="M3 12h4l2-7 4 14 2-7h6"/>',
        "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
        "check": '<path d="m5 12 4 4L19 6"/>',
        "download": '<path d="M12 3v12m0 0 4-4m-4 4-4-4M5 21h14"/>',
        "back": '<path d="m15 18-6-6 6-6"/>',
        "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
        "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-4"/>',
        "flask": '<path d="M9 3h6M10 3v6l-5.5 9A2 2 0 0 0 6.2 21h11.6a2 2 0 0 0 1.7-3L14 9V3"/><path d="M7.5 15h9"/>',
        "sliders": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
        "swap": '<path d="M7 4 3 8l4 4M3 8h13M17 20l4-4-4-4M21 16H8"/>',
    }
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true">{paths[name]}</svg>'
    )


def _shell_css() -> str:
    return """
    :root {
      color-scheme: dark;
      --bg: #090b0d;
      --panel: #111417;
      --panel-2: #171b1f;
      --line: rgba(255,255,255,.08);
      --muted: #8f989f;
      --text: #f5f7f7;
      --acid: #d6ff3f;
      --acid-soft: rgba(214,255,63,.11);
      --coral: #ff725e;
      --blue: #71b7ff;
      --radius: 18px;
      --shadow: 0 24px 80px rgba(0,0,0,.28);
    }
    * { box-sizing: border-box; }
    html { background: var(--bg); scroll-behavior: smooth; }
    body {
      margin: 0; min-height: 100vh; color: var(--text); background:
      radial-gradient(circle at 78% 5%, rgba(214,255,63,.07), transparent 26rem),
      radial-gradient(circle at 10% 85%, rgba(113,183,255,.05), transparent 28rem), var(--bg);
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      -webkit-font-smoothing: antialiased;
    }
    a { color: inherit; text-decoration: none; }
    button, input { font: inherit; }
    .app-shell { display: grid; grid-template-columns: 232px minmax(0,1fr); min-height: 100vh; }
    .sidebar {
      position: sticky; top: 0; height: 100vh; padding: 28px 18px; border-right: 1px solid var(--line);
      background: rgba(9,11,13,.84); backdrop-filter: blur(18px); z-index: 20;
    }
    .brand { display: flex; align-items: center; gap: 11px; padding: 0 9px 28px; }
    .brand-mark { width: 31px; height: 31px; display: grid; place-items: center; color: #090b0d; background: var(--acid); border-radius: 9px; font-size: 13px; font-weight: 900; letter-spacing: -.06em; }
    .brand-word { font-size: 13px; font-weight: 780; letter-spacing: .16em; }
    .brand-word span { display: block; color: var(--muted); font-size: 9px; font-weight: 600; letter-spacing: .24em; margin-top: 3px; }
    .nav-label { color: #666f76; font-size: 10px; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; margin: 13px 10px 8px; }
    .nav-link { display: flex; align-items: center; gap: 11px; padding: 10px 11px; margin: 3px 0; border-radius: 10px; color: #8e979e; font-size: 13px; transition: .18s ease; }
    .nav-link:hover { color: var(--text); background: rgba(255,255,255,.04); }
    .nav-link.active { color: var(--text); background: var(--panel-2); box-shadow: inset 0 0 0 1px rgba(255,255,255,.05); }
    .nav-link.active svg { color: var(--acid); }
    .side-status { position: absolute; bottom: 24px; left: 18px; right: 18px; padding: 13px; border: 1px solid var(--line); border-radius: 12px; background: var(--panel); }
    .side-status-head { display: flex; align-items: center; gap: 8px; font-size: 11px; font-weight: 700; }
    .live-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--acid); box-shadow: 0 0 0 4px rgba(214,255,63,.08); }
    .side-status p { color: var(--muted); font-size: 10px; line-height: 1.45; margin: 7px 0 0; }
    main { min-width: 0; padding: 28px 38px 64px; }
    .topbar { display: flex; justify-content: space-between; align-items: center; gap: 18px; max-width: 1320px; margin: 0 auto 38px; }
    .eyebrow { color: var(--muted); font-size: 10px; font-weight: 740; letter-spacing: .16em; text-transform: uppercase; }
    .top-title { margin-top: 5px; font-size: 18px; font-weight: 650; letter-spacing: -.025em; }
    .top-actions { display: flex; align-items: center; gap: 10px; }
    .status-pill { display: flex; align-items: center; gap: 8px; min-height: 36px; padding: 0 12px; border: 1px solid var(--line); border-radius: 999px; color: #afb6bb; background: rgba(17,20,23,.72); font-size: 11px; }
    .button { display: inline-flex; align-items: center; justify-content: center; gap: 8px; min-height: 39px; padding: 0 15px; border-radius: 10px; border: 1px solid var(--line); background: var(--panel-2); color: var(--text); font-size: 12px; font-weight: 700; transition: .18s ease; }
    .button:hover { transform: translateY(-1px); border-color: rgba(255,255,255,.16); }
    .button.primary { color: #0c0e0e; background: var(--acid); border-color: var(--acid); }
    .button.ghost { background: transparent; }
    .content { max-width: 1320px; margin: 0 auto; }
    .section-head { display: flex; align-items: end; justify-content: space-between; gap: 18px; margin: 38px 0 15px; }
    .section-head h2 { font-size: 18px; letter-spacing: -.03em; margin: 0; }
    .section-head p { color: var(--muted); font-size: 11px; margin: 5px 0 0; }
    .section-link { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 11px; }
    .card { border: 1px solid var(--line); border-radius: var(--radius); background: rgba(17,20,23,.84); box-shadow: var(--shadow); overflow: hidden; }
    .muted { color: var(--muted); }
    .mobile-nav { display: none; }
    @media (max-width: 860px) {
      .app-shell { display: block; }
      .sidebar { display: none; }
      main { padding: 20px 16px 88px; }
      .topbar { margin-bottom: 24px; }
      .top-actions .status-pill { display: none; }
      .mobile-nav { display: grid; grid-template-columns: repeat(3,1fr); position: fixed; z-index: 50; bottom: 10px; left: 12px; right: 12px; padding: 7px; border: 1px solid var(--line); border-radius: 16px; background: rgba(17,20,23,.94); backdrop-filter: blur(20px); box-shadow: var(--shadow); }
      .mobile-nav a { display: flex; flex-direction: column; align-items: center; gap: 4px; padding: 5px; color: var(--muted); font-size: 9px; }
      .mobile-nav a:first-child { color: var(--acid); }
    }
    """


def _nav(active: str = "overview") -> str:
    links = [
        ("overview", "grid", "/", "Overview"),
        ("lab", "flask", "/lab", "What-If Lab"),
        ("reports", "report", "#reports", "Reports"),
        ("film", "film", "#film-room", "Film room"),
        ("pipeline", "pulse", "#pipeline", "Pipeline"),
    ]
    items = "".join(
        f'<a class="nav-link {"active" if key == active else ""}" href="{href}">{_icon(icon)}<span>{label}</span></a>'
        for key, icon, href, label in links
    )
    return f"""
    <aside class="sidebar">
      <a class="brand" href="/" aria-label="Spectrum home">
        <span class="brand-mark">SP</span><span class="brand-word">SPECTRUM<span>MATCH INTELLIGENCE</span></span>
      </a>
      <div class="nav-label">Workspace</div>{items}
      <div class="nav-label">System</div>
      <a class="nav-link" href="/docs">{_icon('target')}<span>API workspace</span></a>
      <div class="side-status"><div class="side-status-head"><span class="live-dot"></span> Analysis engine online</div><p>Evidence-backed processing on this Mac mini.</p></div>
    </aside>
    """


def _mobile_nav() -> str:
    return f"""<nav class="mobile-nav" aria-label="Mobile navigation">
      <a href="/">{_icon('grid', 17)}Overview</a><a href="#reports">{_icon('report', 17)}Reports</a><a href="#pipeline">{_icon('pulse', 17)}Pipeline</a>
    </nav>"""


def render_dashboard(
    broadcasts: Iterable[Broadcast],
    reports: Iterable[Report],
    jobs: Iterable[ProcessingJob],
    *,
    event_count: int,
    clip_count: int,
) -> str:
    broadcast_list = list(broadcasts)
    report_list = list(reports)
    job_list = list(jobs)
    report_by_broadcast = {report.broadcast_id: report for report in report_list}
    jobs_by_broadcast: dict[int, list[ProcessingJob]] = {}
    for job in job_list:
        jobs_by_broadcast.setdefault(job.broadcast_id, []).append(job)

    broadcast_rows: list[str] = []
    for broadcast in broadcast_list:
        report = report_by_broadcast.get(broadcast.id)
        report_action = (
            f'<a class="row-action" href="/reports/{report.id}">Open report {_icon("arrow", 14)}</a>'
            if report else '<span class="muted">Pending report</span>'
        )
        stages = jobs_by_broadcast.get(broadcast.id, [])
        complete = sum(job.status == "completed" for job in stages)
        progress = round(100 * complete / len(stages)) if stages else 0
        broadcast_rows.append(f"""
          <article class="analysis-row" data-search="{html.escape(broadcast.title.lower())}">
            <div class="analysis-id">{broadcast.id:02d}</div>
            <div class="analysis-main"><strong>{html.escape(broadcast.title)}</strong><span>{html.escape(broadcast.platform.upper())} · {len(stages)} pipeline stages</span></div>
            <div class="progress-wrap"><span>{progress}%</span><div class="progress"><i style="width:{progress}%"></i></div></div>
            <span class="state state-{html.escape(broadcast.status.value)}">{html.escape(broadcast.status.value)}</span>
            <div>{report_action}</div>
          </article>""")
    if not broadcast_rows:
        broadcast_rows.append('<div class="empty-state">No broadcasts yet. Register a source through the API to begin.</div>')

    completed_jobs = sum(job.status == "completed" for job in job_list)
    processed = sum(b.status.value == "processed" for b in broadcast_list)
    pipeline_pct = round(100 * completed_jobs / len(job_list)) if job_list else 0

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#090b0d"><title>Spectrum · Match intelligence</title>
<style>{_shell_css()}
  .hero {{ display: grid; grid-template-columns: minmax(0,1.35fr) minmax(290px,.65fr); min-height: 362px; }}
  .hero-copy {{ position: relative; padding: 42px; display: flex; flex-direction: column; justify-content: space-between; overflow: hidden; }}
  .hero-copy:after {{ content:""; position:absolute; width:360px; height:360px; border:1px solid rgba(214,255,63,.10); border-radius:50%; right:-140px; top:-170px; box-shadow:0 0 0 55px rgba(214,255,63,.018),0 0 0 110px rgba(214,255,63,.012); }}
  .hero-kicker {{ display:flex; align-items:center; gap:8px; color:var(--acid); font-size:10px; font-weight:760; letter-spacing:.15em; text-transform:uppercase; }}
  .hero h1 {{ max-width:760px; margin:18px 0 15px; font-size:clamp(35px,5vw,66px); line-height:.98; letter-spacing:-.065em; font-weight:690; }}
  .hero-sub {{ max-width:600px; color:#a1a9ae; font-size:13px; line-height:1.65; }}
  .hero-foot {{ display:flex; align-items:center; gap:16px; margin-top:30px; }}
  .hero-note {{ color:var(--muted); font-size:10px; line-height:1.35; }}
  .featured {{ padding:26px; background:linear-gradient(155deg,#1b1f21,#101315 75%); border-left:1px solid var(--line); display:flex; flex-direction:column; }}
  .featured-label {{ display:flex; justify-content:space-between; color:var(--muted); font-size:9px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }}
  .matchup {{ display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:15px; margin:auto 0; }}
  .team {{ text-align:center; }} .team-mark {{ display:grid; place-items:center; width:66px; height:66px; margin:0 auto 12px; border:1px solid var(--line); border-radius:17px; background:#0c0e10; font-size:19px; font-weight:820; letter-spacing:-.04em; }}
  .team.lat .team-mark {{ color:#fff; box-shadow:inset 0 -3px 0 var(--coral); }} .team.van .team-mark {{ color:var(--blue); box-shadow:inset 0 -3px 0 var(--blue); }}
  .team strong {{ font-size:13px; }} .team span {{ display:block; color:var(--muted); font-size:9px; margin-top:4px; }}
  .series-score {{ text-align:center; }} .series-score strong {{ font-size:38px; letter-spacing:-.07em; }} .series-score span {{ display:block; color:var(--muted); font-size:9px; text-transform:uppercase; letter-spacing:.1em; }}
  .map-strip {{ display:grid; grid-template-columns:repeat(3,1fr); gap:7px; }} .map-chip {{ padding:10px 7px; border:1px solid var(--line); border-radius:9px; text-align:center; background:#0e1113; }} .map-chip strong {{ display:block; font-size:11px; }} .map-chip span {{ display:block; color:var(--muted); font-size:8px; margin-top:3px; }}
  .stats-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }} .stat-card {{ padding:21px 22px; border:1px solid var(--line); border-radius:14px; background:rgba(17,20,23,.72); }}
  .stat-card span {{ color:var(--muted); font-size:9px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }} .stat-card strong {{ display:block; margin-top:11px; font-size:26px; letter-spacing:-.045em; }} .stat-card small {{ display:block; color:#707980; font-size:9px; margin-top:5px; }}
  .analysis-list {{ border:1px solid var(--line); border-radius:14px; overflow:hidden; }} .analysis-row {{ display:grid; grid-template-columns:44px minmax(180px,1.5fr) minmax(110px,.7fr) 90px 120px; gap:16px; align-items:center; min-height:72px; padding:0 19px; border-bottom:1px solid var(--line); background:rgba(17,20,23,.68); font-size:11px; }} .analysis-row:last-child {{ border-bottom:0; }}
  .analysis-id {{ color:#5e676e; font:10px ui-monospace,SFMono-Regular,Menlo,monospace; }} .analysis-main strong {{ display:block; font-size:12px; }} .analysis-main span {{ display:block; color:var(--muted); font-size:9px; margin-top:5px; }}
  .progress-wrap {{ display:flex; align-items:center; gap:8px; color:var(--muted); font-size:9px; }} .progress {{ width:72px; height:3px; background:#24292d; border-radius:9px; overflow:hidden; }} .progress i {{ display:block; height:100%; background:var(--acid); }}
  .state {{ justify-self:start; padding:5px 8px; border-radius:999px; color:#aab2b7; background:rgba(255,255,255,.05); font-size:8px; font-weight:750; letter-spacing:.07em; text-transform:uppercase; }} .state-processed {{ color:var(--acid); background:var(--acid-soft); }} .state-failed {{ color:#ff8b7d; background:rgba(255,114,94,.1); }}
  .row-action {{ display:flex; align-items:center; justify-content:flex-end; gap:5px; font-size:10px; font-weight:700; }} .row-action:hover {{ color:var(--acid); }}
  .pipeline-card {{ display:grid; grid-template-columns:.8fr 1.2fr; }} .pipeline-summary {{ padding:28px; border-right:1px solid var(--line); }} .pipeline-score {{ display:flex; align-items:end; gap:6px; margin:23px 0 8px; }} .pipeline-score strong {{ font-size:56px; letter-spacing:-.07em; line-height:.8; }} .pipeline-score span {{ color:var(--muted); font-size:12px; }}
  .pipeline-summary p {{ color:var(--muted); font-size:11px; line-height:1.6; }} .pipeline-stages {{ padding:24px 28px; display:grid; grid-template-columns:repeat(4,1fr); gap:9px; align-content:center; }} .stage {{ padding:14px 12px; border:1px solid var(--line); border-radius:11px; background:#0d1012; }} .stage b {{ display:flex; align-items:center; gap:6px; color:#d8dddf; font-size:10px; }} .stage b svg {{ color:var(--acid); }} .stage span {{ display:block; color:var(--muted); font-size:8px; margin-top:8px; }}
  .empty-state {{ padding:28px; color:var(--muted); font-size:11px; background:var(--panel); }}
  @media(max-width:1050px) {{ .hero {{ grid-template-columns:1fr; }} .featured {{ min-height:300px; border-left:0; border-top:1px solid var(--line); }} .matchup {{ margin:30px 0; }} .stats-grid {{ grid-template-columns:repeat(2,1fr); }} }}
  @media(max-width:700px) {{ .hero-copy {{ padding:28px 22px; }} .hero h1 {{ font-size:41px; }} .featured {{ padding:22px; }} .stats-grid {{ gap:8px; }} .stat-card {{ padding:17px; }} .analysis-row {{ grid-template-columns:34px 1fr auto; padding:13px 14px; gap:10px; }} .analysis-row .progress-wrap,.analysis-row>div:last-child {{ display:none; }} .pipeline-card {{ grid-template-columns:1fr; }} .pipeline-summary {{ border-right:0; border-bottom:1px solid var(--line); }} .pipeline-stages {{ grid-template-columns:repeat(2,1fr); padding:18px; }} }}
</style></head><body><div class="app-shell">{_nav()}
<main><header class="topbar"><div><div class="eyebrow">Operations desk</div><div class="top-title">Match intelligence</div></div><div class="top-actions"><div class="status-pill"><span class="live-dot"></span>System healthy</div><a class="button" href="/docs">API</a></div></header>
<div class="content">
  <section class="card hero">
    <div class="hero-copy"><div><div class="hero-kicker"><span class="live-dot"></span> Evidence before opinion</div><h1>See the match<br>behind the score.</h1><p class="hero-sub">Broadcast-to-brief analysis for coaching teams. Every stored event carries confidence, timestamp and visual evidence—so the conversation starts with what happened.</p></div><div class="hero-foot"><a class="button primary" href="/coach-reports/{COACH_REPORT_SLUG}">Open latest brief {_icon('arrow',15)}</a><span class="hero-note">Latest · Major IV qualifiers<br>Updated 23 Jun 2026</span></div></div>
    <div class="featured"><div class="featured-label"><span>Featured matchup</span><span>Final · BO5</span></div><div class="matchup"><div class="team van"><div class="team-mark">VAN</div><strong>Vancouver</strong><span>Surge</span></div><div class="series-score"><strong>0–3</strong><span>Series</span></div><div class="team lat"><div class="team-mark">LAT</div><strong>Los Angeles</strong><span>Thieves</span></div></div><div class="map-strip"><div class="map-chip"><strong>156–250</strong><span>Hacienda · HP</span></div><div class="map-chip"><strong>5–6</strong><span>Hacienda · S&amp;D</span></div><div class="map-chip"><strong>0–8</strong><span>Den · OVL</span></div></div></div>
  </section>
  <div class="stats-grid" style="margin-top:12px"><div class="stat-card"><span>Broadcasts</span><strong>{len(broadcast_list):02d}</strong><small>{processed} fully processed</small></div><div class="stat-card"><span>Evidence events</span><strong>{event_count}</strong><small>Timestamped observations</small></div><div class="stat-card"><span>Review clips</span><strong>{clip_count}</strong><small>Ready for film room</small></div><div class="stat-card"><span>Pipeline health</span><strong>{pipeline_pct}%</strong><small>{completed_jobs} stages complete</small></div></div>
  <section id="reports"><div class="section-head"><div><h2>Analysis queue</h2><p>Recent broadcasts and processing state</p></div><a class="section-link" href="/reports">All reports {_icon('arrow',13)}</a></div><div class="analysis-list">{''.join(broadcast_rows)}</div></section>
  <section id="pipeline"><div class="section-head"><div><h2>Processing pipeline</h2><p>From source capture to coach-ready evidence</p></div></div><div class="card pipeline-card"><div class="pipeline-summary"><div class="eyebrow">System completion</div><div class="pipeline-score"><strong>{pipeline_pct}</strong><span>%</span></div><p>Deterministic stages, auditable retries and confidence carried through to every output.</p></div><div class="pipeline-stages"><div class="stage"><b>{_icon('check',14)} Ingest</b><span>Source registered</span></div><div class="stage"><b>{_icon('check',14)} Extract</b><span>HUD sampled</span></div><div class="stage"><b>{_icon('check',14)} Analyse</b><span>Moments scored</span></div><div class="stage"><b>{_icon('check',14)} Report</b><span>Brief generated</span></div></div></div></section>
</div></main></div>{_mobile_nav()}</body></html>"""


def render_featured_coach_report() -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta name="theme-color" content="#090b0d"><title>LAT 3–0 VAN · Coach brief</title>
<style>{_shell_css()}
  .report-page {{ max-width:1120px; margin:0 auto; }} .report-top {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:34px; }} .back-link {{ display:flex; align-items:center; gap:7px; color:var(--muted); font-size:11px; }} .back-link:hover {{ color:var(--text); }}
  .report-hero {{ position:relative; padding:46px; overflow:hidden; }} .report-hero:after {{ content:"03"; position:absolute; right:26px; bottom:-42px; color:rgba(255,255,255,.025); font-size:240px; font-weight:850; letter-spacing:-.12em; pointer-events:none; }} .report-meta {{ display:flex; align-items:center; gap:9px; color:var(--muted); font-size:9px; text-transform:uppercase; letter-spacing:.12em; }} .report-meta i {{ width:3px; height:3px; border-radius:50%; background:var(--acid); }}
  .report-hero h1 {{ max-width:760px; margin:18px 0 12px; font-size:clamp(38px,6vw,72px); line-height:.96; letter-spacing:-.065em; }} .report-deck {{ max-width:680px; color:#9ca5aa; font-size:13px; line-height:1.65; }} .score-banner {{ display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:30px; margin-top:38px; padding-top:28px; border-top:1px solid var(--line); }} .score-team strong {{ display:block; font-size:16px; }} .score-team span {{ display:block; color:var(--muted); font-size:9px; margin-top:5px; }} .score-team:last-child {{ text-align:right; }} .score-big {{ font-size:42px; font-weight:770; letter-spacing:-.07em; }}
  .brief-grid {{ display:grid; grid-template-columns:minmax(0,1.5fr) minmax(270px,.5fr); gap:14px; margin-top:14px; }} .brief-main {{ padding:34px; }} .brief-main h2,.brief-side h2 {{ margin:0 0 19px; font-size:17px; letter-spacing:-.03em; }} .lead {{ color:#d9dddf; font-size:15px; line-height:1.7; margin:0 0 25px; }} .callout {{ padding:16px 18px; border-left:2px solid var(--acid); background:var(--acid-soft); color:#dbe4bd; font-size:11px; line-height:1.6; }}
  .brief-side {{ padding:28px; }} .signal {{ padding:15px 0; border-bottom:1px solid var(--line); }} .signal:last-child {{ border:0; }} .signal span {{ display:block; color:var(--muted); font-size:8px; text-transform:uppercase; letter-spacing:.1em; }} .signal strong {{ display:block; margin-top:7px; font-size:20px; letter-spacing:-.04em; }} .signal small {{ display:block; color:#747e84; font-size:9px; line-height:1.45; margin-top:5px; }}
  .map-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }} .map-card {{ padding:25px; }} .map-no {{ color:var(--acid); font:9px ui-monospace,monospace; }} .map-card h3 {{ margin:13px 0 5px; font-size:16px; }} .map-score {{ font-size:28px; font-weight:760; letter-spacing:-.05em; }} .map-card p {{ color:var(--muted); font-size:10px; line-height:1.6; min-height:48px; }} .map-tag {{ display:inline-block; margin-top:9px; padding:5px 7px; border-radius:6px; color:#b8c0c4; background:rgba(255,255,255,.05); font-size:8px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }}
  .plan-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }} .plan {{ padding:28px; }} .team-line {{ display:flex; align-items:center; gap:10px; margin-bottom:22px; }} .mini-mark {{ display:grid; place-items:center; width:37px; height:37px; border-radius:10px; background:#0b0e10; border:1px solid var(--line); font-size:10px; font-weight:800; }} .team-line h3 {{ margin:0; font-size:14px; }} .team-line span {{ display:block; color:var(--muted); font-size:8px; margin-top:3px; }} .action {{ display:grid; grid-template-columns:24px 1fr; gap:10px; padding:13px 0; border-top:1px solid var(--line); }} .action:first-of-type {{ border-top:0; }} .action-num {{ color:var(--acid); font:9px ui-monospace,monospace; padding-top:2px; }} .action strong {{ display:block; font-size:11px; }} .action p {{ color:var(--muted); font-size:9px; line-height:1.55; margin:5px 0 0; }}
  .confidence {{ display:grid; grid-template-columns:.8fr 1.2fr; }} .confidence-copy {{ padding:28px; border-right:1px solid var(--line); }} .confidence-copy h2 {{ margin:0 0 10px; font-size:17px; }} .confidence-copy p {{ color:var(--muted); font-size:10px; line-height:1.6; }} .confidence-bars {{ padding:26px 30px; }} .conf-row {{ display:grid; grid-template-columns:125px 1fr 38px; gap:10px; align-items:center; margin:12px 0; font-size:9px; }} .conf-track {{ height:5px; border-radius:9px; background:#252a2e; overflow:hidden; }} .conf-track i {{ display:block; height:100%; background:var(--acid); }}
  .sources {{ padding:25px 28px; }} .sources h2 {{ margin:0 0 13px; font-size:14px; }} .source-list {{ display:grid; grid-template-columns:repeat(2,1fr); gap:8px; }} .source-link {{ display:flex; justify-content:space-between; align-items:center; gap:10px; padding:12px; border:1px solid var(--line); border-radius:10px; color:var(--muted); font-size:9px; }} .source-link:hover {{ color:var(--text); border-color:rgba(255,255,255,.15); }}
  @media(max-width:800px) {{ .report-hero {{ padding:28px 22px; }} .brief-grid,.confidence {{ grid-template-columns:1fr; }} .map-grid,.plan-grid {{ grid-template-columns:1fr; }} .brief-main {{ padding:24px 21px; }} .confidence-copy {{ border-right:0; border-bottom:1px solid var(--line); }} .source-list {{ grid-template-columns:1fr; }} .score-banner {{ gap:12px; }} .score-team strong {{ font-size:12px; }} .score-big {{ font-size:32px; }} }}
</style></head><body><div class="app-shell">{_nav('reports')}<main><div class="report-page"><div class="report-top"><a class="back-link" href="/">{_icon('back',15)} Back to overview</a><a class="button ghost" href="/coach-reports/{COACH_REPORT_SLUG}/download">{_icon('download',15)} Download brief</a></div>
<section class="card report-hero"><div class="report-meta"><span>Coach report</span><i></i><span>Major IV qualifiers</span><i></i><span>20 June 2026</span></div><h1>LAT controlled the series.<br>Vancouver only bent Search.</h1><p class="report-deck">A 3–0 result built on decisive respawn separation: a 94-point Hardpoint margin and an 8–0 Overload shutout. Hacienda Search reached round eleven—the single mode where Vancouver kept the match structurally competitive.</p><div class="score-banner"><div class="score-team"><strong>Vancouver Surge</strong><span>Craze · Lunarz · Nero · Mamba</span></div><div class="score-big">0–3</div><div class="score-team"><strong>Los Angeles Thieves</strong><span>aBeZy · Nium · Scrap · HyDra</span></div></div></section>
<div class="brief-grid"><section class="card brief-main"><div class="eyebrow">Executive read</div><h2 style="margin-top:9px">The scoreboard identifies the mode split. It does not yet explain every cause.</h2><p class="lead">LAT produced overwhelming separation in both respawns while HyDra set the lobby ceiling at 77–49 (1.57 K/D). Scrap’s 64–48 (1.33) added a second sustained positive differential. No Vancouver player finished the series positive, which makes isolated map tactics an incomplete explanation: the review has to test spacing, trade distance and repeated loss of map control alongside individual engagements.</p><div class="callout"><strong>Coach’s bottom line:</strong> Vancouver’s usable tape starts with the 11-round Search. LAT’s useful tape is the opposite: identify which habits remain repeatable against a stronger respawn opponent, rather than treating the sweep itself as proof.</div></section><aside class="card brief-side"><h2>Series signals</h2><div class="signal"><span>LAT respawn margin</span><strong>+102</strong><small>+94 HP points and +8 Overload score differential.</small></div><div class="signal"><span>Lobby leader</span><strong>HyDra · 1.57</strong><small>77 kills, 49 deaths, +28 differential.</small></div><div class="signal"><span>Competitive window</span><strong>Round 11</strong><small>Hacienda S&amp;D was the only one-score map.</small></div></aside></div>
<div class="section-head"><div><h2>Map diagnosis</h2><p>Verified outcomes with bounded coaching interpretation</p></div></div><div class="map-grid"><article class="card map-card"><div class="map-no">MAP 01 · HARDPOINT</div><h3>Hacienda</h3><div class="map-score">156–250</div><p>A 94-point deficit is large enough to demand a full rotation audit: first-wave trades, spawn retention and scrap-time exit timing.</p><span class="map-tag">LAT +37.6% of target</span></article><article class="card map-card"><div class="map-no">MAP 02 · SEARCH</div><h3>Hacienda</h3><div class="map-score">5–6</div><p>Vancouver created the only late-map leverage of the series. Pull rounds 9–11 first and grade opening plan, man-advantage conversion and information discipline.</p><span class="map-tag">Decided R11</span></article><article class="card map-card"><div class="map-no">MAP 03 · OVERLOAD</div><h3>Den</h3><div class="map-score">0–8</div><p>The shutout warrants a point-by-point reset: opening lanes, carrier isolation, recovery routes and whether deaths arrived in tradable pairs.</p><span class="map-tag">Shutout</span></article></div>
<div class="section-head"><div><h2>Coach action plan</h2><p>What each room should take into the next review</p></div></div><div class="plan-grid"><section class="card plan"><div class="team-line"><div class="mini-mark" style="color:var(--blue)">VAN</div><div><h3>Vancouver review order</h3><span>Repair the repeatable system first</span></div></div><div class="action"><span class="action-num">01</span><div><strong>Start with Den, not the close Search.</strong><p>Tag all eight scores by opening setup, first death location and surviving trade partner. Look for the first repeated structural failure.</p></div></div><div class="action"><span class="action-num">02</span><div><strong>Grade Hacienda HP in 60-second blocks.</strong><p>Separate hill-time loss from rotation loss. A 94-point gap can hide multiple different problems.</p></div></div><div class="action"><span class="action-num">03</span><div><strong>Preserve the Search framework.</strong><p>Rounds 9–11 are the best evidence of a competitive plan. Identify what produced parity and what failed at the close.</p></div></div></section><section class="card plan"><div class="team-line"><div class="mini-mark" style="color:var(--coral)">LAT</div><div><h3>LAT review order</h3><span>Pressure-test the winning shape</span></div></div><div class="action"><span class="action-num">01</span><div><strong>Audit wins that depended on slaying.</strong><p>HyDra and Scrap created +44 combined differential. Mark situations where the setup would fail without the gunfight edge.</p></div></div><div class="action"><span class="action-num">02</span><div><strong>Pull the five lost Search rounds.</strong><p>The one close map carries more learning value than the two respawn blowouts. Classify plan loss versus execution loss.</p></div></div><div class="action"><span class="action-num">03</span><div><strong>Bank Den opening patterns.</strong><p>Keep the repeatable lane timings and trade spacing; discard opponent-specific reads that will not transfer.</p></div></div></section></div>
<div class="section-head"><div><h2>Evidence confidence</h2><p>Transparent limits on what this brief can claim</p></div></div><section class="card confidence"><div class="confidence-copy"><h2>High confidence on outcomes.<br>Medium confidence on causes.</h2><p>Map scores, series score, rosters and published player totals are corroborated. Tactical recommendations are disciplined review hypotheses because full event telemetry and timestamped VOD coding were not available for this brief.</p></div><div class="confidence-bars"><div class="conf-row"><span>Match outcome</span><div class="conf-track"><i style="width:99%"></i></div><strong>99%</strong></div><div class="conf-row"><span>Player totals</span><div class="conf-track"><i style="width:95%"></i></div><strong>95%</strong></div><div class="conf-row"><span>Mode diagnosis</span><div class="conf-track"><i style="width:78%"></i></div><strong>78%</strong></div><div class="conf-row"><span>Tactical causes</span><div class="conf-track"><i style="width:58%"></i></div><strong>58%</strong></div></div></section>
<section class="card sources" style="margin-top:14px"><h2>Source ledger</h2><div class="source-list"><a class="source-link" href="https://callofdutyleague.com/en-us/schedule/" target="_blank" rel="noreferrer">Official CDL schedule {_icon('arrow',13)}</a><a class="source-link" href="https://fieldlevelmedia.com/esports/la-thieves-unblemished-at-cdl-major-4-qualifying/" target="_blank" rel="noreferrer">Published match recap {_icon('arrow',13)}</a><a class="source-link" href="https://www.reddit.com/r/CoDCompetitive/comments/1ub3tcj/" target="_blank" rel="noreferrer">Official discussion score ledger {_icon('arrow',13)}</a><a class="source-link" href="https://vercel.breakingpoint.gg/en/match/214996/Vancouver-Surge-vs-Los-Angeles-Thieves-at-CDL-Major-4-Qualifier-2026" target="_blank" rel="noreferrer">Breaking Point matchup context {_icon('arrow',13)}</a></div></section>
</div></main></div>{_mobile_nav()}</body></html>"""


def render_featured_coach_report_markdown() -> str:
    return """# Coach brief: Los Angeles Thieves 3–0 Vancouver Surge

**Competition:** CDL 2026 Major IV Qualifiers  
**Date:** 20 June 2026  
**Maps:** Hacienda Hardpoint 250–156 LAT; Hacienda Search & Destroy 6–5 LAT; Den Overload 8–0 LAT

## Executive read

LAT produced overwhelming separation in both respawns. HyDra set the lobby ceiling at 77–49 (1.57 K/D, +28); Scrap added 64–48 (1.33, +16). No Vancouver player finished positive. Hacienda Search reached round eleven and was the only map where Vancouver kept the series structurally competitive.

The result is conclusive; the tactical causes remain review hypotheses until the VOD is coded. Vancouver should test spacing, trade distance, spawn retention and repeated loss of map control instead of reducing the series to individual gunfights.

## Map diagnosis

### 1. Hacienda Hardpoint — LAT 250–156

- The 94-point margin requires a full rotation audit.
- Grade first-wave trades, spawn retention and scrap-time exit timing.
- Split the VOD into 60-second blocks to distinguish hill-time loss from rotation loss.

### 2. Hacienda Search & Destroy — LAT 6–5

- This was Vancouver’s only late-map leverage.
- Pull rounds 9–11 first.
- Grade opening plan, man-advantage conversion and information discipline.
- LAT should prioritise its five lost rounds; they carry more learning value than the respawn blowouts.

### 3. Den Overload — LAT 8–0

- Tag all eight scores by opening setup, first-death location and surviving trade partner.
- Audit carrier isolation, recovery routes and whether deaths arrived in tradable pairs.
- Identify the first repeated structural failure before discussing isolated mistakes.

## Vancouver coaching plan

1. Start with Den, not the close Search; repair the repeatable system first.
2. Grade Hacienda HP by phase and separate rotation losses from hill-time losses.
3. Preserve the Search framework that produced round-eleven parity, then isolate the closing failure.

## LAT coaching plan

1. Audit wins that depended on the +44 combined HyDra/Scrap differential.
2. Classify every lost Search round as plan loss or execution loss.
3. Bank repeatable Den lane timings and trade spacing; discard opponent-specific reads.

## Evidence confidence

- Match outcome and map scores: 99%
- Published player totals: 95%
- Mode-level diagnosis: 78%
- Tactical causes: 58% (review hypotheses; no timestamped event telemetry)

## Sources

- Official CDL schedule: https://callofdutyleague.com/en-us/schedule/
- Field Level Media recap: https://fieldlevelmedia.com/esports/la-thieves-unblemished-at-cdl-major-4-qualifying/
- Official discussion score ledger: https://www.reddit.com/r/CoDCompetitive/comments/1ub3tcj/
- Breaking Point matchup context: https://vercel.breakingpoint.gg/en/match/214996/Vancouver-Surge-vs-Los-Angeles-Thieves-at-CDL-Major-4-Qualifier-2026

Prepared by Spectrum Match Intelligence on 23 June 2026.
"""


_LAB_CSS = """
  .lab-banner { display:flex; gap:13px; align-items:flex-start; padding:15px 18px; margin-bottom:16px;
    border:1px solid rgba(255,114,94,.32); border-radius:14px; background:rgba(255,114,94,.07); }
  .lab-banner svg { color:var(--coral); flex:none; margin-top:1px; }
  .lab-banner strong { color:#ffb7ab; font-size:11px; letter-spacing:.04em; }
  .lab-banner p { margin:5px 0 0; color:#c9a7a0; font-size:10.5px; line-height:1.55; }
  .lab-grid { display:grid; grid-template-columns:minmax(0,1.55fr) minmax(300px,.62fr); gap:14px; align-items:start; }
  .outcome { display:grid; grid-template-columns:1fr auto 1fr auto; gap:22px; align-items:center; padding:26px 30px; margin-bottom:14px; }
  .outcome .team-block { display:flex; flex-direction:column; gap:6px; }
  .outcome .team-block.b { text-align:right; }
  .outcome .tname { font-size:12px; font-weight:760; letter-spacing:.04em; }
  .outcome .tname.a { color:var(--acid); } .outcome .tname.b { color:var(--blue); }
  .outcome .tscore { font-size:54px; font-weight:780; letter-spacing:-.06em; line-height:.9; }
  .outcome .delta-chip { font-size:10px; font-weight:700; color:var(--muted); }
  .outcome .delta-chip.up { color:var(--acid); } .outcome .delta-chip.down { color:var(--coral); }
  .outcome .vs { display:flex; flex-direction:column; align-items:center; gap:7px; color:var(--muted); }
  .outcome .vs .winner-tag { padding:5px 11px; border-radius:999px; font-size:9px; font-weight:780; letter-spacing:.09em; text-transform:uppercase; background:var(--acid-soft); color:var(--acid); }
  .outcome .vs .winner-tag.flip { background:rgba(255,114,94,.14); color:var(--coral); animation:flipPulse 1.1s ease infinite; }
  @keyframes flipPulse { 0%,100%{ box-shadow:0 0 0 0 rgba(255,114,94,.0);} 50%{ box-shadow:0 0 0 6px rgba(255,114,94,.10);} }
  .scenario-readout { grid-column:1/-1; padding-top:16px; margin-top:4px; border-top:1px solid var(--line); color:var(--muted); font-size:10.5px; line-height:1.5; }
  .scenario-readout b { color:var(--text); font-weight:680; }
  .chart-card { padding:20px 22px; margin-bottom:14px; }
  .chart-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px; }
  .chart-head h3 { margin:0; font-size:13px; letter-spacing:-.02em; }
  .chart-head .legend { display:flex; gap:13px; color:var(--muted); font-size:9.5px; }
  .chart-head .legend i { display:inline-block; width:9px; height:3px; border-radius:2px; margin-right:5px; vertical-align:middle; }
  .chart-card svg { width:100%; height:auto; display:block; }
  .control-band + .control-band { margin-top:9px; }
  .control-band .band-label { color:var(--muted); font-size:9px; letter-spacing:.08em; text-transform:uppercase; margin-bottom:5px; }
  .side-card { padding:20px; margin-bottom:14px; }
  .side-card h3 { margin:0 0 4px; font-size:13px; letter-spacing:-.02em; }
  .side-card .sub { color:var(--muted); font-size:10px; margin:0 0 15px; line-height:1.5; }
  .preset-row { display:flex; flex-wrap:wrap; gap:7px; margin-bottom:6px; }
  .preset { padding:8px 11px; border:1px solid var(--line); border-radius:9px; background:var(--panel-2); color:#cfd5d8; font-size:10.5px; font-weight:650; cursor:pointer; transition:.16s ease; }
  .preset:hover { border-color:rgba(214,255,63,.4); color:var(--text); transform:translateY(-1px); }
  .preset.danger:hover { border-color:rgba(255,114,94,.4); }
  .swing-list { display:flex; flex-direction:column; gap:3px; max-height:430px; overflow-y:auto; margin:0 -6px; padding:0 6px; }
  .swing-row { display:grid; grid-template-columns:20px 1fr 70px; gap:9px; align-items:center; padding:7px 8px; border-radius:8px; cursor:pointer; transition:.13s ease; border:1px solid transparent; }
  .swing-row:hover { background:rgba(255,255,255,.035); }
  .swing-row.on { background:var(--acid-soft); border-color:rgba(214,255,63,.28); }
  .swing-row .box { width:15px; height:15px; border-radius:4px; border:1.6px solid #3a4148; display:grid; place-items:center; }
  .swing-row.on .box { background:var(--acid); border-color:var(--acid); color:#0c0e0e; }
  .swing-row .box svg { width:10px; height:10px; opacity:0; } .swing-row.on .box svg { opacity:1; }
  .swing-row .who { min-width:0; } .swing-row .who b { font-size:11px; font-weight:640; display:block; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .swing-row .who span { color:var(--muted); font-size:8.5px; }
  .swing-row .who .star { color:var(--coral); }
  .swing-bar { position:relative; height:18px; }
  .swing-bar .mid { position:absolute; left:50%; top:0; bottom:0; width:1px; background:#2c3239; }
  .swing-bar i { position:absolute; top:4px; height:10px; border-radius:2px; }
  .swing-bar i.pos { left:50%; background:var(--acid); } .swing-bar i.neg { right:50%; background:var(--blue); }
  .swing-bar b { position:absolute; top:1px; font-size:9px; font-weight:700; color:#aeb5ba; }
  .swing-bar b.pos { left:calc(50% + 4px); } .swing-bar b.neg { right:calc(50% + 4px); }
  .swap-chip { margin-top:6px; display:inline-flex; align-items:center; gap:5px; padding:3px 7px; border-radius:7px; border:1px solid var(--line); background:#0d1012; color:var(--muted); font-size:8.5px; font-weight:700; cursor:pointer; }
  .swap-chip.on { color:var(--blue); border-color:rgba(113,183,255,.4); }
  details.assume { border-top:1px solid var(--line); margin-top:6px; padding-top:12px; }
  details.assume summary { cursor:pointer; color:var(--muted); font-size:10.5px; font-weight:650; list-style:none; }
  details.assume summary::-webkit-details-marker { display:none; }
  details.assume p { color:#828b91; font-size:9.5px; line-height:1.6; margin:10px 0 0; }
  details.assume code { color:#b9c2a6; font-size:9px; }
  @media(max-width:1040px){ .lab-grid{ grid-template-columns:1fr; } }
  @media(max-width:640px){ .outcome{ grid-template-columns:1fr auto 1fr; gap:12px; padding:20px; } .outcome .tscore{ font-size:38px; } .outcome .scenario-readout{ grid-column:1/-1; } }
"""


_LAB_JS = r"""
(function(){
  const D = window.__SPECTRUM__;
  const M = D.match, BASE = D.result, EVENTS = D.events, IMPACTS = D.impacts || [];
  const DUR = BASE.duration, TARGET = M.target, A = M.team_a, B = M.team_b;
  const C = { a:'#d6ff3f', b:'#71b7ff', cf:'#ff725e', contested:'#363d45', grid:'#22282d', axis:'#5a636a' };
  const byId = Object.fromEntries(EVENTS.map(e => [e.id, e]));
  const sel = new Map();           // event_id -> kind
  const FLIPS = new Set(IMPACTS.filter(i => i.flips_winner).map(i => i.id));

  const $ = s => document.querySelector(s);
  const fmt = n => (n>0?'+':'') + n;

  function interventions(){
    return [...sel.entries()].map(([event_id, kind]) => ({ kind, event_id, delta:0 }));
  }

  // ---- charts ---------------------------------------------------------------
  function pathFrom(points, fx, fy){
    return points.map((p,i) => (i? 'L':'M') + fx(p[0]).toFixed(1) + ',' + fy(p[1]).toFixed(1)).join(' ');
  }

  function drawWinProb(cf){
    const W=920, H=250, pl=46, pr=14, pt=16, pb=24, w=W-pl-pr, h=H-pt-pb;
    const fx = t => pl + (Math.min(t,DUR)/DUR)*w;
    const fy = p => pt + (1-p)*h;
    let g='';
    [0,0.25,0.5,0.75,1].forEach(v=>{
      const y=fy(v);
      g += `<line x1="${pl}" y1="${y}" x2="${W-pr}" y2="${y}" stroke="${v===0.5?'#2e353c':C.grid}" stroke-width="1" ${v===0.5?'stroke-dasharray="3 4"':''}/>`;
      g += `<text x="${pl-8}" y="${y+3}" text-anchor="end" fill="${C.axis}" font-size="9">${Math.round(v*100)}%</text>`;
    });
    const basePts = BASE.timeline.map(p=>[p.t, p.prob_a]).concat([[DUR, BASE.timeline[BASE.timeline.length-1].prob_a]]);
    const baseLine = pathFrom(basePts, fx, fy);
    const area = `M${fx(0)},${fy(0)} ` + basePts.map(p=>'L'+fx(p[0]).toFixed(1)+','+fy(p[1]).toFixed(1)).join(' ') + ` L${fx(DUR)},${fy(0)} Z`;
    let cfLine='';
    if(cf){
      const cfPts = cf.timeline.map(p=>[p.t, p.prob_a]).concat([[cf.duration, cf.timeline[cf.timeline.length-1].prob_a]]);
      cfLine = `<path d="${pathFrom(cfPts, fx, fy)}" fill="none" stroke="${C.cf}" stroke-width="2.4" stroke-dasharray="6 4" stroke-linejoin="round"/>`;
    }
    $('#chart-winprob').innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Win probability over time">
      ${g}
      <path d="${area}" fill="rgba(214,255,63,.08)"/>
      <path d="${baseLine}" fill="none" stroke="${C.a}" stroke-width="${cf?1.6:2.6}" stroke-opacity="${cf?0.55:1}" stroke-linejoin="round"/>
      ${cfLine}
      <text x="${pl}" y="11" fill="${C.axis}" font-size="9">P(${A} win)</text>
      <text x="${W-pr}" y="11" text-anchor="end" fill="${C.axis}" font-size="9">map time →</text>
    </svg>`;
  }

  function drawRace(cf){
    const W=920, H=230, pl=40, pr=14, pt=14, pb=22, w=W-pl-pr, h=H-pt-pb;
    const fx = t => pl + (Math.min(t,DUR)/DUR)*w;
    const fy = s => pt + (1-Math.min(s,TARGET)/TARGET)*h;
    let g='';
    [0,0.5,1].forEach(v=>{ const y=pt+(1-v)*h; g+=`<line x1="${pl}" y1="${y}" x2="${W-pr}" y2="${y}" stroke="${C.grid}"/>`; g+=`<text x="${pl-7}" y="${y+3}" text-anchor="end" fill="${C.axis}" font-size="9">${Math.round(v*TARGET)}</text>`; });
    g += `<line x1="${pl}" y1="${fy(TARGET)}" x2="${W-pr}" y2="${fy(TARGET)}" stroke="${C.cf}" stroke-width="1" stroke-dasharray="2 4" opacity=".6"/>`;
    const aPts = BASE.timeline.map(p=>[p.t,p.a]), bPts = BASE.timeline.map(p=>[p.t,p.b]);
    let extra='';
    if(cf){
      extra = `<path d="${pathFrom(cf.timeline.map(p=>[p.t,p.a]),fx,fy)}" fill="none" stroke="${C.a}" stroke-width="1.4" stroke-dasharray="5 4" opacity=".9"/>`
            + `<path d="${pathFrom(cf.timeline.map(p=>[p.t,p.b]),fx,fy)}" fill="none" stroke="${C.b}" stroke-width="1.4" stroke-dasharray="5 4" opacity=".9"/>`;
    }
    $('#chart-race').innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      ${g}
      <path d="${pathFrom(aPts,fx,fy)}" fill="none" stroke="${C.a}" stroke-width="${cf?1.7:2.4}" stroke-opacity="${cf?.5:1}"/>
      <path d="${pathFrom(bPts,fx,fy)}" fill="none" stroke="${C.b}" stroke-width="${cf?1.7:2.4}" stroke-opacity="${cf?.5:1}"/>
      ${extra}
    </svg>`;
  }

  function drawBand(el, segments){
    const W=920, H=30, pl=4, pr=4, w=W-pl-pr;
    const fx = t => pl + (t/DUR)*w;
    let r='';
    segments.forEach(s=>{
      const col = s.holder===A?C.a : s.holder===B?C.b : C.contested;
      const op = (s.holder===A||s.holder===B)?0.92:0.5;
      r += `<rect x="${fx(s.start).toFixed(1)}" y="6" width="${Math.max(0.4,(fx(s.end)-fx(s.start))).toFixed(1)}" height="18" fill="${col}" opacity="${op}"/>`;
    });
    for(let t=M.hill_seconds; t<DUR; t+=M.hill_seconds){ r += `<line x1="${fx(t)}" y1="3" x2="${fx(t)}" y2="27" stroke="#0a0c0e" stroke-width="1.4"/>`; }
    EVENTS.filter(e=>e.kind==='spawn_flip').forEach(e=>{
      const x=fx(e.t); r += `<line x1="${x}" y1="2" x2="${x}" y2="28" stroke="#fff" stroke-width="1.3"/><circle cx="${x}" cy="3" r="2.6" fill="${C.cf}"/>`;
    });
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${r}</svg>`;
  }

  // ---- swing list -----------------------------------------------------------
  function buildSwingList(){
    const top = IMPACTS.slice(0, 34);
    const maxAbs = Math.max(1, ...top.map(i=>Math.abs(i.swing_points)));
    const host = $('#swing-list'); host.innerHTML='';
    top.forEach(it=>{
      const row = document.createElement('div');
      row.className='swing-row'; row.dataset.id=it.id;
      const who = it.kind==='spawn_flip' ? `${it.team} spawn flip` : `${it.killer} › ${it.victim}`;
      const sub = it.kind==='spawn_flip' ? 'systemic · prevent' : `${it.zone||''} · t=${Math.round(it.t)}s`;
      const star = FLIPS.has(it.id) ? ' <span class="star" title="removing this flips the map">★</span>' : '';
      const pct = (Math.abs(it.swing_points)/maxAbs)*48;
      const pos = it.swing_points>=0;
      row.innerHTML = `<span class="box">${ICON_CHECK}</span>
        <span class="who"><b>${who}${star}</b><span>${sub}</span></span>
        <span class="swing-bar"><span class="mid"></span>
          <i class="${pos?'pos':'neg'}" style="width:${pct.toFixed(1)}%"></i>
          <b class="${pos?'pos':'neg'}">${fmt(it.swing_points)}</b></span>`;
      row.addEventListener('click', ()=> toggle(it.id));
      host.appendChild(row);
    });
  }

  function toggle(id){
    if(sel.has(id)) sel.delete(id);
    else sel.set(id, byId[id] && byId[id].kind==='spawn_flip' ? 'prevent_flip' : 'remove_event');
    syncRows(); recompute();
  }
  function syncRows(){
    document.querySelectorAll('.swing-row').forEach(r=> r.classList.toggle('on', sel.has(r.dataset.id)));
  }

  // ---- outcome + recompute --------------------------------------------------
  function renderOutcome(res, delta){
    $('#sc-a').textContent = res.final_a; $('#sc-b').textContent = res.final_b;
    const dA=$('#delta-a'), dB=$('#delta-b');
    dA.textContent = delta.team_a_change? fmt(delta.team_a_change)+' pts' : '';
    dB.textContent = delta.team_b_change? fmt(delta.team_b_change)+' pts' : '';
    dA.className = 'delta-chip ' + (delta.team_a_change>0?'up':delta.team_a_change<0?'down':'');
    dB.className = 'delta-chip ' + (delta.team_b_change>0?'up':delta.team_b_change<0?'down':'');
    const tag=$('#winner-tag');
    tag.textContent = delta.winner_flipped ? (delta.winner+' steal the map') : (res.winner+' win');
    tag.className = 'winner-tag' + (delta.winner_flipped?' flip':'');
  }

  function scenarioText(){
    if(sel.size===0) return `Baseline simulation — <b>no edits applied.</b> Toggle events on the right or pick a scenario to see the counterfactual.`;
    const parts = [...sel.entries()].map(([id,kind])=>{
      const e = byId[id]; if(!e) return id;
      if(kind==='prevent_flip') return `prevent ${e.team}'s spawn flip`;
      if(kind==='swap_kill') return `${e.victim} wins the duel vs ${e.killer}`;
      return e.kind==='spawn_flip'? `remove ${e.team} flip` : `erase ${e.killer}'s kill on ${e.victim}`;
    });
    return `<b>${sel.size} edit${sel.size>1?'s':''}:</b> ` + parts.join('; ') + '.';
  }

  let pending=0;
  async function recompute(){
    $('#scenario-readout').innerHTML = scenarioText();
    if(sel.size===0){ paintBaseline(); return; }
    const my = ++pending;
    try{
      const r = await fetch('/api/sim/counterfactual', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ interventions: interventions() })
      });
      const j = await r.json();
      if(my!==pending) return;
      drawWinProb(j.counterfactual); drawRace(j.counterfactual);
      $('#cf-band-wrap').style.display='block';
      drawBand($('#band-cf'), j.counterfactual.control_segments);
      renderOutcome(j.counterfactual, j.delta);
    }catch(err){ console.error(err); }
  }
  function paintBaseline(){
    drawWinProb(null); drawRace(null);
    $('#cf-band-wrap').style.display='none';
    drawBand($('#band-base'), BASE.control_segments);
    renderOutcome(BASE, { team_a_change:0, team_b_change:0, winner_flipped:false, winner:BASE.winner });
  }

  // ---- presets --------------------------------------------------------------
  function setSelection(entries){ sel.clear(); entries.forEach(([id,k])=>sel.set(id,k)); syncRows(); recompute(); }
  window.__lab = {
    preset(name){
      if(name==='reset') return setSelection([]);
      if(name==='flip'){ const f=EVENTS.find(e=>e.kind==='spawn_flip'); return setSelection(f?[[f.id,'prevent_flip']]:[]); }
      if(name==='triple') return setSelection(['H1','H2','H3'].filter(id=>byId[id]).map(id=>[id,'remove_event']));
      if(name==='vanbest'){
        const picks=[]; const f=EVENTS.find(e=>e.kind==='spawn_flip'); if(f) picks.push([f.id,'prevent_flip']);
        IMPACTS.filter(i=>i.kind==='kill' && i.swing_points>0).slice(0,6).forEach(i=>picks.push([i.id,'remove_event']));
        return setSelection(picks);
      }
    }
  };

  const ICON_CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12 4 4L19 6"/></svg>';
  buildSwingList(); paintBaseline();
  const h = decodeURIComponent(location.hash.slice(1));
  if(h && window.__lab) window.__lab.preset(h);
  window.addEventListener('hashchange', ()=>{ const x=decodeURIComponent(location.hash.slice(1)); if(x) window.__lab.preset(x); });
})();
"""


def render_lab() -> str:
    match = sim.get_active_match()
    payload = sim.to_payload(match)
    data_json = json.dumps(payload, separators=(",", ":"))
    result = payload["result"]
    note = html.escape(match.source_note)
    synthetic = match.synthetic
    banner = (
        f"""<div class="lab-banner">{_icon('shield', 17)}<div>
          <strong>SYNTHETIC DEMO DATA — not the real {html.escape(match.team_a)}–{html.escape(match.team_b)} match.</strong>
          <p>{note} Every kill, spawn flip and timestamp is invented to exercise the engine. Counterfactual scores are
          <b>model estimates</b>: the signal is the <b>delta</b> and the mechanism, not the absolute number. Drop a coded
          match into <code>data/fixtures/demo_synthetic_match.json</code> (with <code>synthetic:false</code>) to analyse a real VOD.</p>
        </div></div>"""
        if synthetic
        else f"""<div class="lab-banner" style="border-color:rgba(214,255,63,.3);background:var(--acid-soft)">{_icon('check', 17)}<div>
          <strong>CODED MATCH — {html.escape(match.map_name)}.</strong>
          <p>{note} Counterfactual scores remain <b>model estimates</b> from the documented control rule; treat deltas as directional.</p></div></div>"""
    )

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta name="theme-color" content="#090b0d">
<title>What-If Lab · Spectrum</title>
<style>{_shell_css()}{_LAB_CSS}</style></head>
<body><div class="app-shell">{_nav('lab')}
<main><header class="topbar"><div><div class="eyebrow">Counterfactual engine</div><div class="top-title">What-If Lab · {html.escape(match.map_name)}</div></div>
<div class="top-actions"><div class="status-pill">{_icon('sliders',14)} {len(payload['events'])} events modelled</div><a class="button" href="/coach-reports/lat-vancouver-2026-06-20">Coach brief</a></div></header>
<div class="content">
{banner}
<section class="card outcome">
  <div class="team-block a"><span class="tname a">{html.escape(match.team_a)}</span><span class="tscore" id="sc-a">{result['final_a']}</span><span class="delta-chip" id="delta-a"></span></div>
  <div class="vs"><span class="winner-tag" id="winner-tag">{html.escape(result['winner'])} win</span><span style="font-size:9px;color:var(--muted);letter-spacing:.1em">RACE TO {match.target}</span></div>
  <div class="team-block b"><span class="tname b">{html.escape(match.team_b)}</span><span class="tscore" id="sc-b">{result['final_b']}</span><span class="delta-chip" id="delta-b"></span></div>
  <div></div>
  <div class="scenario-readout" id="scenario-readout"></div>
</section>
<div class="lab-grid">
  <div>
    <section class="card chart-card">
      <div class="chart-head"><h3>Win probability — model xMWP</h3><div class="legend"><span><i style="background:#d6ff3f"></i>baseline</span><span><i style="background:#ff725e"></i>what-if</span></div></div>
      <div id="chart-winprob"></div>
    </section>
    <section class="card chart-card">
      <div class="chart-head"><h3>Race to {match.target}</h3><div class="legend"><span><i style="background:#d6ff3f"></i>{html.escape(match.team_a)}</span><span><i style="background:#71b7ff"></i>{html.escape(match.team_b)}</span><span>dashed = what-if</span></div></div>
      <div id="chart-race"></div>
    </section>
    <section class="card chart-card">
      <div class="chart-head"><h3>Hill control &amp; spawns</h3><div class="legend"><span><i style="background:#d6ff3f"></i>{html.escape(match.team_a)}</span><span><i style="background:#71b7ff"></i>{html.escape(match.team_b)}</span><span><i style="background:#363d45"></i>contested</span><span>● flip</span></div></div>
      <div class="control-band"><div class="band-label">Baseline</div><div id="band-base"></div></div>
      <div class="control-band" id="cf-band-wrap" style="display:none"><div class="band-label" style="color:#ff9c8f">What-if</div><div id="band-cf"></div></div>
    </section>
  </div>
  <div>
    <section class="card side-card">
      <h3>{_icon('flask',15)} Scenarios</h3>
      <p class="sub">One-click counterfactuals. Stack them to see how far the map can move.</p>
      <div class="preset-row">
        <button class="preset" onclick="__lab.preset('flip')">Spawns hold (no flip)</button>
        <button class="preset" onclick="__lab.preset('triple')">Erase HyDra's triple</button>
        <button class="preset" onclick="__lab.preset('vanbest')">{html.escape(match.team_b)} best case</button>
      </div>
      <div class="preset-row"><button class="preset danger" onclick="__lab.preset('reset')">{_icon('back',13)} Reset to actual</button></div>
      <details class="assume"><summary>How this is computed</summary>
        <p>The active hill ticks for whichever team has more <b>available</b> players (alive and returned from spawn).
        A kill removes the victim for <code>{int(match.respawn_delay)}s</code> + spawn-to-hill travel, so removing a kill or
        preventing a spawn flip changes who holds. Even counts are a contested standoff that scores for no one.
        Win probability is the uncalibrated <code>HeuristicV0</code>. Numbers are <b>model estimates</b> — read the deltas.</p>
      </details>
    </section>
    <section class="card side-card">
      <h3>{_icon('swap',15)} Biggest swings</h3>
      <p class="sub">Leave-one-out: net points each event added to {html.escape(match.team_a)}'s margin. Click to toggle. ★ = removing it flips the map.</p>
      <div class="swing-list" id="swing-list"></div>
    </section>
  </div>
</div>
</div></main></div>{_mobile_nav()}
<script>window.__SPECTRUM__ = {data_json};</script>
<script>{_LAB_JS}</script>
</body></html>"""
