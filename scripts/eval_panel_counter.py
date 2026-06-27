"""Evaluate the scoreboard kill counter against verified ground truth — Phase 4.

Runs the counting logic over the cached panel readings (``run_panel_counter.py``
produced ``readings.jsonl``) and scores it against the **human-verified post-game
card** (``PLAYER_MAP_STATS`` / ``TEAM_KILL_CHECKPOINTS`` in ``hardpoint_breakdown``):

  * per-player final kills vs the post-game card,
  * team totals vs the card (LAT 106 / VAN 79) and the mid-map 505s checkpoint,
  * killfeed reconciliation — using the panel counter as the kill ground truth to
    estimate the killfeed detector's precision/recall.

Offline + reproducible: it reads the committed ``readings.jsonl``, no VOD or OCR.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.events import Evidence  # noqa: E402
from backend.app.services.hardpoint_breakdown import PLAYER_MAP_STATS, TEAM_KILL_CHECKPOINTS  # noqa: E402
from backend.app.services.panel_counter import PanelKillCounter, reconcile_with_killfeed  # noqa: E402
from backend.app.services.real_match import TEAM_A, TEAM_B  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "panel_counter"
KILLFEED_ANNOTATIONS = ROOT / "data" / "killfeed_dataset" / "annotations.jsonl"
CHECKPOINT_T = 505


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _ground_truth_by_slot(names: dict[str, str]) -> dict[str, int]:
    stats = {TEAM_A: PLAYER_MAP_STATS.get("LAT", {}), TEAM_B: PLAYER_MAP_STATS.get("VAN", {})}
    out: dict[str, int] = {}
    for slot, name in names.items():
        team = TEAM_A if slot[0] == "a" else TEAM_B
        kd = stats.get(team, {}).get(name)
        if kd is not None:
            out[slot] = kd[0]
    return out


def _killfeed_onset_times() -> list[float]:
    if not KILLFEED_ANNOTATIONS.exists():
        return []
    return [json.loads(line)["video_timestamp_seconds"]
            for line in KILLFEED_ANNOTATIONS.read_text().splitlines() if line.strip()]


def evaluate(dataset_dir: Path) -> dict[str, Any]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    names = manifest["names"]
    readings = [json.loads(line) for line in (dataset_dir / "readings.jsonl").read_text().splitlines() if line.strip()]

    counter = PanelKillCounter(team_a=TEAM_A, team_b=TEAM_B, names=names)
    kill_times: list[float] = []
    checkpoint_team: dict[str, int] | None = None
    for row in readings:
        t = row["t"]
        for e in counter.update(t, {"a": row["a"], "b": row["b"]},
                                Evidence(video_timestamp_seconds=t, crop_path="x")):
            if e.event_type == "kill":
                kill_times.append(t)
        if checkpoint_team is None and t >= CHECKPOINT_T:
            checkpoint_team = dict(counter.team_kills())

    # --- per-player + team totals vs the post-game card ---------------------------
    gt = _ground_truth_by_slot(names)
    got = counter.kills_by_slot()
    per_player = []
    abs_errors = []
    for slot in sorted(gt):
        predicted, truth = got.get(slot, 0), gt[slot]
        per_player.append({"slot": slot, "player": names.get(slot, slot),
                           "predicted_kills": predicted, "verified_kills": truth,
                           "error": predicted - truth})
        abs_errors.append(abs(predicted - truth))

    counter_team = counter.team_kills()
    pred_a, pred_b = counter_team["a"], counter_team["b"]
    gt_team = {TEAM_A: sum(k for s, k in gt.items() if s[0] == "a"),
               TEAM_B: sum(k for s, k in gt.items() if s[0] == "b")}

    checkpoint = None
    if checkpoint_team and CHECKPOINT_T in TEAM_KILL_CHECKPOINTS:
        verified_cp = TEAM_KILL_CHECKPOINTS[CHECKPOINT_T]
        checkpoint = {"t": CHECKPOINT_T,
                      "predicted": {TEAM_A: checkpoint_team["a"], TEAM_B: checkpoint_team["b"]},
                      "verified": verified_cp}

    rec = reconcile_with_killfeed(kill_times, _killfeed_onset_times())

    n = len(abs_errors)
    return {
        "dataset": {"path": _rel(dataset_dir), "detector": manifest.get("detector"),
                    "frames": len(readings), "source_url": manifest.get("source_url")},
        "metrics": {
            "per_player_kills": per_player,
            "mean_abs_error_kills": round(sum(abs_errors) / n, 3) if n else None,
            "exact_player_matches": f"{sum(e == 0 for e in abs_errors)}/{n}",
            "team_totals": {
                "predicted": {TEAM_A: pred_a, TEAM_B: pred_b},
                "verified": gt_team,
                "error": {TEAM_A: pred_a - gt_team[TEAM_A], TEAM_B: pred_b - gt_team[TEAM_B]},
            },
            "checkpoint_505s": checkpoint,
            "kill_events_emitted": len(kill_times),
            "killfeed_reconciliation": rec.as_dict(),
        },
        "notes": [
            "Per-player/team kills are scored against the human-verified post-game card.",
            "The counter only counts increments it observes from the start of the window; "
            "it never invents pre-watch kills.",
            "killfeed_reconciliation treats the panel counter as kill ground truth to estimate "
            "the killfeed detector's precision (confirmed / killfeed onsets).",
        ],
    }


def print_report(result: dict[str, Any]) -> None:
    d, m = result["dataset"], result["metrics"]
    print("# Panel Kill-Counter Evaluation")
    print()
    print(f"- dataset: {d['path']}  | frames: {d['frames']}  | detector: {d['detector']}")
    print(f"- mean abs error (kills/player): {m['mean_abs_error_kills']}  | exact: {m['exact_player_matches']}")
    print()
    print("## per-player final kills vs verified post-game card")
    for p in m["per_player_kills"]:
        flag = "ok" if p["error"] == 0 else f"{p['error']:+d}"
        print(f"  {p['player']:<8} predicted {p['predicted_kills']:>3}  verified {p['verified_kills']:>3}  [{flag}]")
    tt = m["team_totals"]
    print(f"\nteam totals: predicted {tt['predicted']}  verified {tt['verified']}  error {tt['error']}")
    if m["checkpoint_505s"]:
        cp = m["checkpoint_505s"]
        print(f"505s checkpoint: predicted {cp['predicted']}  verified {cp['verified']}")
    print(f"\nkillfeed reconciliation (panel = ground truth): {m['killfeed_reconciliation']}")
    print("\nNotes:")
    for note in result["notes"]:
        print(f"- {note}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--write-json", type=Path)
    args = ap.parse_args(argv)
    result = evaluate(args.dataset)
    print_report(result)
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
