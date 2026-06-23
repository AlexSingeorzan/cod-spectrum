from __future__ import annotations

import html
import json
from pathlib import Path

from ..schemas import ReportDocument


def write_reports(document: ReportDocument, output_dir: Path, stem: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / f"{stem}.json",
        "markdown": output_dir / f"{stem}.md",
        "html": output_dir / f"{stem}.html",
    }
    paths["json"].write_text(document.model_dump_json(indent=2))
    markdown = render_markdown(document)
    paths["markdown"].write_text(markdown)
    paths["html"].write_text(render_html(document))
    return paths


def render_markdown(document: ReportDocument) -> str:
    final_metric = document.hardpoint_summary.get("final_score")
    final = final_metric.value if final_metric else {}
    lines = [
        f"# cod-spectrum report — {document.broadcast.get('title', 'Broadcast')}",
        "",
        f"**Mode:** {document.map.get('mode')}  ",
        f"**Final score:** {final.get('team_a', 'n/a')}–{final.get('team_b', 'n/a')}  ",
        f"**Data confidence:** {document.data_confidence:.1%}",
        "",
        "## Key moments",
        "",
    ]
    for event in document.key_moments:
        lines.append(f"- {event['timestamp_seconds']:.1f}s — `{event['event_type']}` at {event.get('score_a')}–{event.get('score_b')} (confidence {event['confidence']:.1%}; evidence: `{event['evidence_frame_path']}`)")
    lines.extend(["", "## Possible breaks / retakes", ""])
    for event in document.possible_breaks_retakes:
        lines.append(f"- {event['timestamp_seconds']:.1f}s — `{event['event_type']}`: {event.get('raw_text')} ({event['confidence']:.1%})")
    lines.extend(["", "## Recommended clips", ""])
    for clip in document.recommended_clips:
        lines.append(f"- [{clip.title}]({clip.url}) ({clip.start_seconds:.1f}s–{clip.end_seconds:.1f}s; confidence {clip.confidence:.1%}; evidence: `{clip.evidence_frame_path}`)")
    lines.extend(["", "## xMWP timeline", "", "| Time | Score | P(Team A) | Confidence |", "|---:|---:|---:|---:|"])
    for point in document.xmwp_timeline:
        lines.append(f"| {point.timestamp_seconds:.1f}s | {point.score_a}–{point.score_b} | {point.probability_a:.1%} | {point.confidence:.1%} |")
    lines.extend(["", "## Full timeline", ""])
    for event in document.timeline:
        lines.append(f"- {event['timestamp_seconds']:.1f}s `{event['event_type']}` {event.get('score_a')}–{event.get('score_b')} — confidence {event['confidence']:.1%}")
    lines.extend(["", "## Known limitations", ""])
    lines.extend(f"- {limitation}" for limitation in document.known_limitations)
    return "\n".join(lines) + "\n"


def render_html(document: ReportDocument) -> str:
    timeline_rows = "".join(
        f"<tr><td>{event['timestamp_seconds']:.1f}s</td><td>{html.escape(event['event_type'])}</td><td>{event.get('score_a')}–{event.get('score_b')}</td><td>{event['confidence']:.1%}</td><td><code>{html.escape(event['evidence_frame_path'])}</code></td></tr>"
        for event in document.timeline
    )
    clip_items = "".join(
        f"<li><a href=\"{html.escape(clip.url)}\">{html.escape(clip.title)}</a> ({clip.start_seconds:.1f}s–{clip.end_seconds:.1f}s; confidence {clip.confidence:.1%}; evidence <code>{html.escape(clip.evidence_frame_path)}</code>)</li>"
        for clip in document.recommended_clips
    )
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in document.known_limitations)
    payload = html.escape(json.dumps(document.model_dump(mode="json")))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>cod-spectrum report</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#18202a}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}}.confidence{{font-size:2rem}}code{{font-size:.8rem}}</style></head>
<body><h1>{html.escape(document.broadcast.get('title', 'Broadcast'))}</h1>
<p class="confidence">Data confidence: {document.data_confidence:.1%}</p>
<h2>Recommended clips</h2><ul>{clip_items}</ul>
<h2>Timeline</h2><table><thead><tr><th>Time</th><th>Event</th><th>Score</th><th>Confidence</th><th>Evidence</th></tr></thead><tbody>{timeline_rows}</tbody></table>
<h2>Known limitations</h2><ul>{limitations}</ul>
<script type="application/json" id="report-data">{payload}</script></body></html>"""
