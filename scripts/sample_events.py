"""Phase 1 sample output + schema evaluation.

Emits a realistic ``GameEvent`` stream to ``data/fixtures/sample_events.jsonl`` and
prints an evaluation report. The stream has two clearly-separated groups:

  * VERIFIED   - real facts + one real insight from the LA Thieves vs Vancouver
                 Surge VOD (human-read scorebar, with the actual evidence crops).
  * ILLUSTRATIVE - flagged synthetic examples of future-module payloads
                 (kill / communication / position) to show the taxonomy. They are
                 marked is_placeholder + source=synthetic and use a non-real evidence
                 marker, so they can never be mistaken for real telemetry.

Run: .venv/bin/python scripts/sample_events.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `backend...` imports

from backend.app.events import (
    CommunicationEvent,
    Evidence,
    EventKind,
    GameEvent,
    InsightEvent,
    KillEvent,
    LeadChangeEvent,
    MapEndEvent,
    PositionEvent,
    Provenance,
    ScoreUpdateEvent,
    SourceKind,
    to_jsonl,
)
from backend.app.services.real_match import (
    MAP_NAME,
    SOURCE_URL,
    TEAM_A,
    TEAM_B,
    VERIFIED,
)

ROOT = Path(__file__).resolve().parents[1]
CROP_DIR = ROOT / "data/crops/lat_van_hp"
OUT_PATH = ROOT / "data/fixtures/sample_events.jsonl"
ILLUSTRATIVE_FRAME = "illustrative://schema-demo"  # deliberately not a real path

_AVAILABLE_CROPS = sorted(CROP_DIR.glob("sb_*.png"))


def crop_rel(seconds: int) -> str:
    """Relative path to the evidence crop for a timestamp, falling back to the
    nearest available crop (the final read at 685s is evidenced by sb_0678)."""
    exact = CROP_DIR / f"sb_{seconds:04d}.png"
    chosen = exact if exact.exists() else min(
        _AVAILABLE_CROPS, key=lambda p: abs(int(p.stem.split("_")[1]) - seconds)
    )
    return str(chosen.relative_to(ROOT))


def verified(seconds: int) -> Provenance:
    return Provenance(source=SourceKind.HUMAN_VERIFIED, labeled_by="alex",
                      note="read frame-by-frame from the official VOD")


def evidence_at(seconds: int) -> Evidence:
    return Evidence(video_timestamp_seconds=float(seconds), crop_path=crop_rel(seconds), source_url=SOURCE_URL)


def score_at(seconds: int) -> tuple[int, int]:
    return next((a, b) for t, a, b in VERIFIED if t == seconds)


def build_stream() -> list[GameEvent]:
    events: list[GameEvent] = []

    # --- VERIFIED facts: score timeline at four real, evidenced points ----------
    score_ids: dict[int, str] = {}
    for seconds in (90, 146, 258, 650):
        a, b = score_at(seconds)
        event = GameEvent(
            broadcast_id=1, match_id=1, map_id=1, video_timestamp_seconds=float(seconds),
            confidence=0.97, provenance=verified(seconds), evidence=evidence_at(seconds),
            payload=ScoreUpdateEvent(team_a=TEAM_A, team_b=TEAM_B, score_a=a, score_b=b),
            tags=["verified"],
        )
        events.append(event)
        score_ids[seconds] = event.event_id

    # --- VERIFIED facts: the two real lead changes -----------------------------
    lead_b = GameEvent(  # VAN takes the lead at 2:26
        broadcast_id=1, match_id=1, map_id=1, video_timestamp_seconds=146.0, confidence=0.97,
        provenance=verified(146), evidence=evidence_at(146),
        payload=LeadChangeEvent(team_a=TEAM_A, team_b=TEAM_B, score_a=34, score_b=38, new_leader_side="b"),
        tags=["verified"],
    )
    lead_a = GameEvent(  # LAT retakes the lead by 4:18
        broadcast_id=1, match_id=1, map_id=1, video_timestamp_seconds=258.0, confidence=0.97,
        provenance=verified(258), evidence=evidence_at(258),
        payload=LeadChangeEvent(team_a=TEAM_A, team_b=TEAM_B, score_a=85, score_b=63, new_leader_side="a"),
        tags=["verified"],
    )
    events += [lead_b, lead_a]

    # --- VERIFIED fact: map end -------------------------------------------------
    events.append(GameEvent(
        broadcast_id=1, match_id=1, map_id=1, video_timestamp_seconds=685.0, confidence=0.99,
        provenance=verified(685), evidence=evidence_at(685),
        payload=MapEndEvent(team_a=TEAM_A, team_b=TEAM_B, score_a=250, score_b=156, winner_side="a",
                            map_name=MAP_NAME),
        tags=["verified"],
    ))

    # --- VERIFIED insight: the real finding, citing the real facts --------------
    events.append(GameEvent(
        broadcast_id=1, match_id=1, map_id=1, video_timestamp_seconds=258.0, confidence=0.85,
        provenance=Provenance(source=SourceKind.DERIVED, producer="coach-view@0.1",
                              note="lead-duration over the verified score timeline"),
        evidence=Evidence(video_timestamp_seconds=258.0, source_url=SOURCE_URL),
        derived_from=[score_ids[146], lead_b.event_id, score_ids[258], lead_a.event_id],
        payload=InsightEvent(
            headline="VAN's early lead never threatened the result",
            explanation=("VAN led from 2:26 to ~4:18 (t146-258), but it cost LAT only a "
                         "temporary deficit before the retake. The 250-156 box score hides "
                         "that swing entirely."),
            metric="opponent_lead_duration_seconds", value=112.0, subject="VAN",
        ),
        tags=["verified"],
    ))

    # --- ILLUSTRATIVE: future-module payloads, flagged synthetic ----------------
    demo_provenance = Provenance(source=SourceKind.SYNTHETIC, producer="schema-demo",
                                 note="illustrative only - this detector is not built yet")
    demo_evidence = Evidence(video_timestamp_seconds=258.0, frame_path=ILLUSTRATIVE_FRAME)
    demo = [
        KillEvent(attacker="Envoy", attacker_team=TEAM_A, victim="Pred", victim_team=TEAM_B,
                  weapon="SMG", headshot=False),
        CommunicationEvent(transcript="they flipped us, rotate P4 now", speaker="Envoy", team=TEAM_A,
                           callout_type="rotate", targets=["P4"], audio_source="player"),
        PositionEvent(x=0.42, y=0.71, player="Envoy", team=TEAM_A, observed_team=TEAM_A, detector="minimap"),
    ]
    for payload in demo:
        events.append(GameEvent(
            broadcast_id=1, match_id=1, map_id=1, video_timestamp_seconds=258.0, confidence=0.50,
            provenance=demo_provenance, evidence=demo_evidence, payload=payload,
            is_placeholder=True, tags=["illustrative", "synthetic"],
        ))

    return events


def evaluate(events: list[GameEvent]) -> bool:
    ids = {e.event_id for e in events}
    facts = [e for e in events if e.kind == EventKind.FACT]
    insights = [e for e in events if e.kind == EventKind.INSIGHT]
    verified_events = [e for e in events if not e.is_placeholder]
    illustrative = [e for e in events if e.is_placeholder]
    facts_with_visual = [e for e in facts if e.evidence.has_visual()]
    model_events = [e for e in events if e.provenance.source == SourceKind.MODEL]
    model_versioned = [e for e in model_events if e.provenance.model_version]
    dangling = [(e.event_id, ref) for e in insights for ref in e.derived_from if ref not in ids]

    print("\n=== sample event stream ===")
    for e in events:
        flag = "  " if not e.is_placeholder else "~ "
        print(f"{flag}{e.video_timestamp_seconds:6.1f}s  {e.kind.value:<7} {e.event_type:<13} "
              f"conf={e.confidence:.2f}  src={e.provenance.source.value:<14} [{','.join(e.tags)}]")

    print("\n=== evaluation ===")
    print(f"total events ................. {len(events)}")
    print(f"  facts / insights .......... {len(facts)} / {len(insights)}")
    print(f"  verified / illustrative ... {len(verified_events)} / {len(illustrative)}")
    fact_pct = 100.0 * len(facts_with_visual) / len(facts) if facts else 100.0
    print(f"facts with visual evidence .. {len(facts_with_visual)}/{len(facts)} ({fact_pct:.0f}%)")
    mv = f"{len(model_versioned)}/{len(model_events)}" if model_events else "n/a (no model-sourced events)"
    print(f"model events with version ... {mv}")
    print(f"insight citation integrity .. {'OK' if not dangling else f'BROKEN {dangling}'}")
    distinct_ids = len(ids) == len(events)
    print(f"event ids distinct .......... {'OK' if distinct_ids else 'COLLISION'}")

    ok = (fact_pct == 100.0) and not dangling and distinct_ids
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}  ->  {OUT_PATH.relative_to(ROOT)}")
    return ok


def main() -> int:
    events = build_stream()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(to_jsonl(events) + "\n")
    return 0 if evaluate(events) else 1


if __name__ == "__main__":
    sys.exit(main())
