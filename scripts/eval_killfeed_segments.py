"""Evaluate Phase 4 Stage B killfeed segmentation readiness.

This is not content accuracy. It reports how many detected killfeed rows have
field-level boxes and crops for attacker, weapon, and victim, plus failure
reasons for missing fields.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.killfeed_segmentation import CORE_FIELDS, OPTIONAL_FIELDS  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "killfeed_dataset"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_segments(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "segments.jsonl"
    if not path.exists():
        raise SystemExit(f"segments not found: {_rel(path)}; run scripts/build_killfeed_segments.py first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(dataset_dir: Path) -> dict[str, Any]:
    rows = load_segments(dataset_dir)
    field_counts = {field: 0 for field in CORE_FIELDS + OPTIONAL_FIELDS}
    crop_counts = {field: 0 for field in CORE_FIELDS + OPTIONAL_FIELDS}
    failure_reasons: Counter[str] = Counter()
    complete_core = 0
    confidence_sum = 0.0

    for row in rows:
        segments = row.get("segments", {})
        core_ok = True
        confidence_sum += float(row.get("confidence", 0.0))
        if row.get("failure_reason"):
            failure_reasons[str(row["failure_reason"])] += 1
        for field in CORE_FIELDS + OPTIONAL_FIELDS:
            segment = segments.get(field, {})
            if segment.get("box") is not None:
                field_counts[field] += 1
            if segment.get("crop_path"):
                crop_counts[field] += 1
            if field in CORE_FIELDS and segment.get("box") is None:
                core_ok = False
                reason = segment.get("failure_reason") or f"{field}_missing"
                failure_reasons[str(reason)] += 1
        complete_core += int(core_ok)

    n = len(rows)
    return {
        "dataset": {
            "path": _rel(dataset_dir),
            "segments": _rel(dataset_dir / "segments.jsonl"),
            "rows": n,
        },
        "metrics": {
            "complete_core_segments": complete_core,
            "complete_core_rate": round(complete_core / n, 4) if n else 0.0,
            "mean_segmentation_confidence": round(confidence_sum / n, 4) if n else 0.0,
            "field_box_counts": field_counts,
            "field_crop_counts": crop_counts,
            "failure_reasons": dict(failure_reasons.most_common(20)),
        },
        "notes": [
            "Segmentation readiness is not OCR or weapon-classification accuracy.",
            "Missing or low-confidence fields stay null; no boxes are guessed from fixed layout.",
            "PanelKillCounter remains the kill spine; killfeed segments are enrichment evidence.",
        ],
    }


def print_report(result: dict[str, Any]) -> None:
    dataset = result["dataset"]
    metrics = result["metrics"]
    print("# Killfeed Segmentation Evaluation")
    print()
    print(f"- dataset: {dataset['path']}")
    print(f"- rows: {dataset['rows']}")
    print(f"- complete attacker+weapon+victim boxes: "
          f"{metrics['complete_core_segments']}/{dataset['rows']} "
          f"({metrics['complete_core_rate']:.4f})")
    print(f"- mean segmentation confidence: {metrics['mean_segmentation_confidence']:.4f}")
    print()
    print("## field boxes")
    for field, count in metrics["field_box_counts"].items():
        print(f"- {field}: {count}")
    if metrics["failure_reasons"]:
        print()
        print("## top failure reasons")
        for reason, count in metrics["failure_reasons"].items():
            print(f"- {reason}: {count}")
    print()
    print("Notes:")
    for note in result["notes"]:
        print(f"- {note}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--write-json", type=Path)
    return parser.parse_args(argv)


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
