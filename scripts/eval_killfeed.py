"""Evaluate the killfeed detector against human labels — Phase 4, deliverable 1.

Honesty first: with an unlabelled dataset this prints "no accuracy claim" and exits
cleanly. It only reports precision/recall once a person has labelled
``annotations.jsonl``:

  * mark each detected candidate ``valid_kill`` true (a real kill) or false (a false
    positive), and
  * add any kills the detector MISSED as rows with ``detector="manual_added"`` and
    ``valid_kill=true`` (these are the false negatives that make recall measurable).

Two separate metrics, never conflated:

  * detection — onset-level kill-detection precision / recall / F1. This is what the
    classical detector is judged on.
  * content_readiness — how many rows carry attacker/victim/kill_type labels, i.e. how
    ready the set is for ``scripts/eval_killfeed_content.py`` and the Phase-4
    content reader (which unlocks DeathEvent / WeaponEvent / TradeEvent). This
    reports readiness, not content accuracy — we never fabricate a number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.killfeed import MODEL_NAME, MODEL_VERSION  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "killfeed_dataset"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_annotations(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    path = dataset_dir / "annotations.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return manifest, rows


def _detection_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [r for r in rows if r["label"].get("valid_kill") is not None]
    auto = [r for r in rows if r.get("detector") != "manual_added"]
    auto_labeled = [r for r in auto if r["label"].get("valid_kill") is not None]

    tp = sum(1 for r in auto_labeled if r["label"]["valid_kill"] is True)
    fp = sum(1 for r in auto_labeled if r["label"]["valid_kill"] is False)
    # Misses are the kills a human added by hand because the detector never fired.
    fn = sum(1 for r in rows if r.get("detector") == "manual_added"
             and r["label"].get("valid_kill") is True)

    metrics: dict[str, Any] = {
        "auto_candidates": len(auto),
        "labeled": len(labeled),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }
    if not labeled:
        metrics["status"] = "unlabeled — no accuracy claim (label valid_kill + add manual_added misses)"
        return metrics
    metrics["precision"] = round(tp / (tp + fp), 4) if (tp + fp) else None
    metrics["recall"] = round(tp / (tp + fn), 4) if (tp + fn) else None
    p, r = metrics["precision"], metrics["recall"]
    metrics["f1"] = round(2 * p * r / (p + r), 4) if p and r else None
    metrics["status"] = "labeled"
    return metrics


def _content_readiness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r["label"].get("valid_kill") is True]
    have = lambda key: sum(1 for r in valid if r["label"].get(key))  # noqa: E731
    return {
        "valid_kills": len(valid),
        "with_attacker": have("attacker"),
        "with_victim": have("victim"),
        "with_kill_type": have("kill_type"),
        "with_weapon": have("weapon"),
        "note": ("counts of human-read fields on real kills; these train the future "
                 "content reader. Use eval_killfeed_content.py for reader accuracy once labelled."),
    }


def evaluate(dataset_dir: Path) -> dict[str, Any]:
    manifest, rows = load_annotations(dataset_dir)
    detection = _detection_metrics(rows)
    failures = [
        {"id": r["id"], "video_timestamp_seconds": r["video_timestamp_seconds"],
         "detector_confidence": r.get("detector_confidence")}
        for r in rows
        if r.get("detector") != "manual_added" and r["label"].get("valid_kill") is False
    ][:20]
    return {
        "dataset": {
            "path": _rel(dataset_dir),
            "detector": f"{MODEL_NAME}@{MODEL_VERSION}",
            "manifest_detector": manifest.get("detector"),
            "source_url": manifest.get("source_url"),
            "candidates": len(rows),
            "label_status": manifest.get("label_status"),
        },
        "metrics": {
            "detection": detection,
            "content_readiness": _content_readiness(rows),
        },
        "false_positive_examples": failures,
        "notes": [
            "Detection precision/recall is onset-level kill detection by classical CV; "
            "it is a baseline, not production OCR.",
            "Recall requires the labeller to add missed kills as detector='manual_added'.",
            "content_readiness is not accuracy — run eval_killfeed_content.py once "
            "attacker/victim/kill_type labels exist.",
        ],
    }


def print_report(result: dict[str, Any]) -> None:
    d = result["dataset"]
    det = result["metrics"]["detection"]
    content = result["metrics"]["content_readiness"]
    print("# Killfeed Detection Evaluation")
    print()
    print(f"- dataset: {d['path']}")
    print(f"- detector: {d['detector']}")
    print(f"- candidates: {d['candidates']}  | label_status: {d['label_status']}")
    print()
    print("## detection")
    print(f"- status: {det['status']}")
    print(f"- auto candidates / labeled: {det['auto_candidates']} / {det['labeled']}")
    if det["status"] == "labeled":
        print(f"- TP/FP/FN: {det['true_positives']}/{det['false_positives']}/{det['false_negatives']}")
        print(f"- precision: {det['precision']}  recall: {det['recall']}  F1: {det['f1']}")
    print()
    print("## content readiness (not accuracy)")
    print(f"- valid kills: {content['valid_kills']}  | with attacker/victim/kill_type/weapon: "
          f"{content['with_attacker']}/{content['with_victim']}/"
          f"{content['with_kill_type']}/{content['with_weapon']}")
    print()
    print("Notes:")
    for note in result["notes"]:
        print(f"- {note}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--write-json", type=Path)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate(args.dataset)
    print_report(result)
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
