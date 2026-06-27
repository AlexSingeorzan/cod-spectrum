"""Evaluate the Phase 4 killfeed content reader.

This evaluates attacker/victim/weapon reading from ``annotations.jsonl``. The real
LAT/VAN dataset is currently unlabelled, so this script reports that state without
claiming accuracy. Once rows are human-labelled, it reports both:

* operational gallery: train on all labelled rows and read them back;
* leave-one-out: exclude the evaluated row from the gallery before reading it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.killfeed_content import (  # noqa: E402
    MODEL_NAME,
    MODEL_VERSION,
    KillfeedContentReader,
    default_dataset_dir,
    labeled_content_rows,
    load_annotation_rows,
    manual_read_from_row,
)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _expected(row: dict[str, Any]) -> dict[str, Any]:
    label = row["label"]
    return {
        "attacker": label.get("attacker"),
        "victim": label.get("victim"),
        "weapon": label.get("weapon"),
        "headshot": label.get("headshot"),
        "is_trade": label.get("is_trade"),
    }


def _predicted(read) -> dict[str, Any] | None:
    if read is None:
        return None
    return {
        "attacker": read.attacker,
        "victim": read.victim,
        "weapon": read.weapon,
        "headshot": read.headshot,
        "is_trade": read.is_trade,
    }


def _score_rows(rows: list[dict[str, Any]], reads: list) -> dict[str, Any]:
    n = len(rows)
    fields = ["attacker", "victim", "weapon", "headshot", "is_trade"]
    correct = {field: 0 for field in fields}
    totals = {field: 0 for field in fields}
    exact = 0
    abstentions = 0
    failures = []
    confidence_sum = 0.0

    for row, read in zip(rows, reads):
        expected = _expected(row)
        predicted = _predicted(read)
        if predicted is None:
            abstentions += 1
        else:
            confidence_sum += read.confidence

        row_exact = predicted == expected
        exact += int(row_exact)
        for field in fields:
            if expected[field] is not None:
                totals[field] += 1
                correct[field] += int(predicted is not None and predicted[field] == expected[field])
        if not row_exact:
            failures.append({
                "id": row["id"],
                "timestamp": row["video_timestamp_seconds"],
                "expected": expected,
                "predicted": predicted,
            })

    return {
        "samples": n,
        "exact_matches": exact,
        "exact_accuracy": round(exact / n, 4) if n else 0.0,
        "field_accuracy": {
            field: round(correct[field] / totals[field], 4) if totals[field] else None
            for field in fields
        },
        "abstentions": abstentions,
        "mean_confidence": round(confidence_sum / max(1, n - abstentions), 4),
        "failures": failures[:20],
    }


def _evaluate_gallery(dataset_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    reader = KillfeedContentReader(dataset_dir=dataset_dir, confidence_threshold=0.0)
    reads = [reader.read_annotation(row) for row in rows]
    metrics = _score_rows(rows, reads)
    metrics["training_samples"] = reader.label_count
    return metrics


def _evaluate_leave_one_out(dataset_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    reads = []
    for row in rows:
        reader = KillfeedContentReader(
            dataset_dir=dataset_dir,
            exclude_sample_ids={row["id"]},
        )
        reads.append(reader.read_annotation(row))
    return _score_rows(rows, reads)


def evaluate(dataset_dir: Path) -> dict[str, Any]:
    manifest, rows = load_annotation_rows(dataset_dir)
    labeled = labeled_content_rows(rows)
    label_status = "labeled" if labeled else "unlabeled"
    result: dict[str, Any] = {
        "dataset": {
            "path": _rel(dataset_dir),
            "model": f"{MODEL_NAME}@{MODEL_VERSION}",
            "manifest_detector": manifest.get("detector"),
            "source_url": manifest.get("source_url"),
            "candidates": len(rows),
            "content_labeled_rows": len(labeled),
            "label_status": label_status,
        },
        "metrics": {
            "content_readiness": {
                "valid_kills_with_identity": len(labeled),
                "with_weapon": sum(1 for row in labeled if row["label"].get("weapon")),
                "with_headshot": sum(1 for row in labeled if row["label"].get("headshot") is not None),
                "with_trade": sum(1 for row in labeled if row["label"].get("is_trade") is not None),
            }
        },
        "notes": [
            "No content accuracy is reported until annotations.jsonl has human labels.",
            "Operational gallery is a regression check; leave-one-out is the honest small-data estimate.",
            "Panel counter remains the kill-count spine; killfeed content is the weapon/headshot/trade enrichment layer.",
        ],
    }
    if not labeled:
        result["metrics"]["reader"] = {
            "status": "unlabeled — no content-reader accuracy claim",
            "operational_gallery": None,
            "leave_one_out": None,
        }
        return result

    result["metrics"]["manual_label_events"] = {
        "readable_rows": sum(1 for row in labeled if manual_read_from_row(dataset_dir, row) is not None),
        "source": "manual_label",
    }
    result["metrics"]["reader"] = {
        "status": "labeled",
        "operational_gallery": _evaluate_gallery(dataset_dir, labeled),
        "leave_one_out": _evaluate_leave_one_out(dataset_dir, labeled),
    }
    return result


def print_report(result: dict[str, Any]) -> None:
    d = result["dataset"]
    reader = result["metrics"]["reader"]
    readiness = result["metrics"]["content_readiness"]
    print("# Killfeed Content Reader Evaluation")
    print()
    print(f"- dataset: {d['path']}")
    print(f"- model: {d['model']}")
    print(f"- candidates: {d['candidates']}  | content-labelled rows: {d['content_labeled_rows']}")
    print(f"- label status: {d['label_status']}")
    print()
    print("## readiness")
    print(f"- valid kills with identity: {readiness['valid_kills_with_identity']}")
    print(f"- with weapon/headshot/trade fields: {readiness['with_weapon']}/{readiness['with_headshot']}/{readiness['with_trade']}")
    print()
    print("## reader")
    print(f"- status: {reader['status']}")
    if reader["status"] == "labeled":
        for name in ("operational_gallery", "leave_one_out"):
            m = reader[name]
            print(f"- {name}: exact={m['exact_matches']}/{m['samples']} "
                  f"acc={m['exact_accuracy']} abstain={m['abstentions']} conf={m['mean_confidence']}")
            print(f"  fields={m['field_accuracy']}")
    print()
    print("Notes:")
    for note in result["notes"]:
        print(f"- {note}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=default_dataset_dir())
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
