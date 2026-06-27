"""Evaluate Phase 5 weapon recognition.

The real dataset is currently an unlabelled icon-crop scaffold, so the default
run reports readiness without claiming accuracy. Once labels exist, this compares
the supported baselines on the same split.
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

from backend.app.services.weapon_recognition import (  # noqa: E402
    MODEL_NAMES,
    MODEL_VERSION,
    Approach,
    WeaponRecognizer,
    default_dataset_dir,
    labeled_weapon_rows,
    load_weapon_rows,
)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _expected(row: dict[str, Any]) -> str:
    return str(row["label"]["weapon"])


def _score(rows: list[dict[str, Any]], predictions: list) -> dict[str, Any]:
    n = len(rows)
    expected = [_expected(row) for row in rows]
    predicted = [prediction.weapon if prediction is not None else None for prediction in predictions]
    classes = sorted(set(expected))
    correct = sum(1 for e, p in zip(expected, predicted) if e == p)
    abstentions = sum(1 for p in predicted if p is None)
    confidence_values = [p.confidence for p in predictions if p is not None and p.weapon is not None]
    latency_values = [p.latency_ms for p in predictions if p is not None]

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for klass in classes:
        tp = sum(1 for e, p in zip(expected, predicted) if e == klass and p == klass)
        fp = sum(1 for e, p in zip(expected, predicted) if e != klass and p == klass)
        fn = sum(1 for e, p in zip(expected, predicted) if e == klass and p != klass)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[klass] = {
            "support": sum(1 for e in expected if e == klass),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    confusion: dict[str, dict[str, int]] = {}
    for e, p in zip(expected, predicted):
        confusion.setdefault(e, {})
        confusion[e][str(p or "NULL")] = confusion[e].get(str(p or "NULL"), 0) + 1

    failures = []
    for row, e, p, prediction in zip(rows, expected, predicted, predictions):
        if e != p:
            failures.append({
                "id": row["id"],
                "expected": e,
                "predicted": p,
                "confidence": prediction.confidence if prediction is not None else 0.0,
                "failure_reason": prediction.failure_reason if prediction is not None else "no_prediction",
                "icon_image": row.get("icon_image"),
            })

    return {
        "samples": n,
        "exact_matches": correct,
        "top1_accuracy": round(correct / n, 4) if n else 0.0,
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "abstentions": abstentions,
        "abstention_rate": round(abstentions / n, 4) if n else 0.0,
        "mean_confidence": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else 0.0,
        "mean_latency_ms": round(sum(latency_values) / len(latency_values), 4) if latency_values else 0.0,
        "per_class": per_class,
        "confusion": confusion,
        "failures": failures[:20],
    }


def _gallery(dataset_dir: Path, rows: list[dict[str, Any]], approach: Approach) -> dict[str, Any]:
    recognizer = WeaponRecognizer(dataset_dir, approach=approach, confidence_threshold=0.0)
    return {
        **_score(rows, [recognizer.read_row(row) for row in rows]),
        "training_samples": recognizer.label_count,
        "classes": recognizer.classes,
    }


def _leave_one_out(dataset_dir: Path, rows: list[dict[str, Any]], approach: Approach) -> dict[str, Any]:
    predictions = []
    for row in rows:
        recognizer = WeaponRecognizer(
            dataset_dir,
            approach=approach,
            exclude_sample_ids={row["id"]},
        )
        predictions.append(recognizer.read_row(row))
    return _score(rows, predictions)


def evaluate(
    dataset_dir: Path,
    approaches: list[Approach] | None = None,
) -> dict[str, Any]:
    approaches = approaches or ["template", "histogram"]
    manifest, rows = load_weapon_rows(dataset_dir)
    labelled = labeled_weapon_rows(rows)
    label_counts = Counter(row["label"].get("weapon") for row in labelled)
    unclear = sum(1 for row in rows if row.get("label", {}).get("unclear") is True)
    unknown = sum(1 for row in rows if row.get("label", {}).get("weapon") == "unknown")

    result: dict[str, Any] = {
        "dataset": {
            "path": _rel(dataset_dir),
            "dataset_id": manifest.get("dataset_id"),
            "icon_count": len(rows),
            "labelled_weapon_icons": len(labelled),
            "label_counts": dict(label_counts),
            "unknown_labels": unknown,
            "unclear_labels": unclear,
            "label_status": "labeled" if labelled else "unlabeled",
        },
        "models": {},
        "notes": [
            "Weapon recognition is independent from player-name OCR.",
            "No real accuracy is reported until weapon icons are human-labelled.",
            "Operational gallery is a regression check; leave-one-out is the honest small-data estimate.",
            "Low-confidence predictions return weapon=null.",
        ],
    }

    for approach in approaches:
        model = f"{MODEL_NAMES[approach]}@{MODEL_VERSION}"
        if not labelled:
            result["models"][approach] = {
                "model": model,
                "status": "unlabeled - no weapon-recognition accuracy claim",
                "operational_gallery": None,
                "leave_one_out": None,
            }
            continue
        result["models"][approach] = {
            "model": model,
            "status": "labeled",
            "operational_gallery": _gallery(dataset_dir, labelled, approach),
            "leave_one_out": _leave_one_out(dataset_dir, labelled, approach),
        }
    return result


def print_report(result: dict[str, Any]) -> None:
    d = result["dataset"]
    print("# Weapon Recognition Evaluation")
    print()
    print(f"- dataset: {d['path']}")
    print(f"- icons: {d['icon_count']} | labelled: {d['labelled_weapon_icons']}")
    print(f"- label status: {d['label_status']}")
    if d["label_counts"]:
        print(f"- labels: {d['label_counts']}")
    print()
    for approach, metrics in result["models"].items():
        print(f"## {approach}")
        print(f"- model: {metrics['model']}")
        print(f"- status: {metrics['status']}")
        if metrics["status"] == "labeled":
            for split in ("operational_gallery", "leave_one_out"):
                m = metrics[split]
                print(
                    f"- {split}: top1={m['exact_matches']}/{m['samples']} "
                    f"acc={m['top1_accuracy']} macro_f1={m['macro_f1']} "
                    f"abstain={m['abstentions']}"
                )
        print()
    print("Notes:")
    for note in result["notes"]:
        print(f"- {note}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=default_dataset_dir())
    ap.add_argument("--write-json", type=Path)
    ap.add_argument(
        "--approach",
        choices=sorted(MODEL_NAMES),
        action="append",
        help="Approach to evaluate; repeat for multiple. Defaults to template and histogram.",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate(args.dataset, approaches=args.approach)
    print_report(result)
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
