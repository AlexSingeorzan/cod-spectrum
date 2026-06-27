"""Evaluate the Phase 3 CDL scorebar OCR baseline.

Outputs two separate metrics:

* operational_gallery: trains on all labeled glyphs and reads the known gallery.
  This catches wiring/segmentation regressions.
* leave_one_out: excludes the evaluated crop's glyphs from the gallery. This is
  the honest small-data generalization estimate and is used to cap OCR
  confidence in the runtime engine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.scorebar_ocr import (  # noqa: E402
    DigitClassifier,
    ScorePairCandidate,
    default_dataset_dir,
    load_manifest,
    load_templates,
    score_pair_candidates,
)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _read_candidates(
    sample,
    dataset_dir: Path,
    exclude_sample: bool,
) -> list[ScorePairCandidate]:
    exclude = {sample.sample_id} if exclude_sample else set()
    classifier = DigitClassifier.from_templates(load_templates(dataset_dir, exclude_sample_ids=exclude))
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        return []
    return score_pair_candidates(classifier, image)


def _digit_matches(predicted: tuple[int, int] | None, expected: tuple[int, int]) -> tuple[int, int]:
    expected_digits = f"{expected[0]}{expected[1]}"
    if predicted is None:
        return 0, len(expected_digits)
    predicted_digits = f"{predicted[0]}{predicted[1]}"
    matches = sum(1 for a, b in zip(predicted_digits, expected_digits) if a == b)
    return matches, len(expected_digits)


def _evaluate_mode(dataset_dir: Path, exclude_sample: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _metadata, samples = load_manifest(dataset_dir)
    rows: list[dict[str, Any]] = []
    exact = 0
    side_matches = 0
    digit_matches = 0
    digit_total = 0
    confidence_sum = 0.0
    abstentions = 0

    for sample in samples:
        expected = (sample.score_a, sample.score_b)
        candidates = _read_candidates(sample, dataset_dir, exclude_sample=exclude_sample)
        top = candidates[0] if candidates else None
        predicted = (top.score_a, top.score_b) if top else None
        confidence = float(top.confidence) if top else 0.0
        if top is None:
            abstentions += 1
        exact += int(predicted == expected)
        if predicted is not None:
            side_matches += int(predicted[0] == expected[0])
            side_matches += int(predicted[1] == expected[1])
        dm, dt = _digit_matches(predicted, expected)
        digit_matches += dm
        digit_total += dt
        confidence_sum += confidence
        rows.append({
            "sample_id": sample.sample_id,
            "timestamp_seconds": sample.timestamp_seconds,
            "expected": list(expected),
            "predicted": list(predicted) if predicted is not None else None,
            "raw_text": top.text if top else "",
            "confidence": round(confidence, 4),
            "exact": predicted == expected,
            "top_candidates": [
                {
                    "score_a": candidate.score_a,
                    "score_b": candidate.score_b,
                    "text": candidate.text,
                    "confidence": round(candidate.confidence, 4),
                }
                for candidate in candidates[:5]
            ],
        })

    n = len(samples)
    metrics = {
        "samples": n,
        "exact_matches": exact,
        "score_exact_accuracy": round(exact / n, 4) if n else 0.0,
        "side_accuracy": round(side_matches / (2 * n), 4) if n else 0.0,
        "digit_accuracy": round(digit_matches / digit_total, 4) if digit_total else 0.0,
        "mean_confidence": round(confidence_sum / n, 4) if n else 0.0,
        "abstentions": abstentions,
        "failures": [row for row in rows if not row["exact"]][:20],
    }
    return metrics, rows


def _temporal_decode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = 0
    previous: tuple[int, int] | None = None
    decoded: list[dict[str, Any]] = []
    for row in rows:
        candidates = row["top_candidates"]
        pick = None
        if previous is not None:
            valid = []
            for candidate in candidates:
                score = (candidate["score_a"], candidate["score_b"])
                da = score[0] - previous[0]
                db = score[1] - previous[1]
                if da >= 0 and db >= 0 and da <= 45 and db <= 45 and da + db <= 55:
                    valid.append((candidate["confidence"], candidate))
            if valid:
                pick = max(valid, key=lambda item: item[0])[1]
        if pick is None and candidates:
            pick = candidates[0]
        predicted = [pick["score_a"], pick["score_b"]] if pick else None
        if predicted is not None:
            previous = (predicted[0], predicted[1])
        is_exact = predicted == row["expected"]
        exact += int(is_exact)
        decoded.append({
            "sample_id": row["sample_id"],
            "timestamp_seconds": row["timestamp_seconds"],
            "expected": row["expected"],
            "predicted": predicted,
            "confidence": pick["confidence"] if pick else 0.0,
            "exact": is_exact,
        })
    n = len(rows)
    return {
        "samples": n,
        "exact_matches": exact,
        "score_exact_accuracy": round(exact / n, 4) if n else 0.0,
        "failures": [row for row in decoded if not row["exact"]][:20],
    }


def evaluate(dataset_dir: Path) -> dict[str, Any]:
    metadata, samples = load_manifest(dataset_dir)
    operational, operational_rows = _evaluate_mode(dataset_dir, exclude_sample=False)
    leave_one_out, loo_rows = _evaluate_mode(dataset_dir, exclude_sample=True)
    temporal = _temporal_decode(loo_rows)
    return {
        "dataset": {
            "path": _rel(dataset_dir),
            "samples": len(samples),
            "source_url": metadata.get("source_url"),
            "label_source": metadata.get("label_source"),
            "labeled_by": metadata.get("labeled_by"),
            "excluded": metadata.get("excluded", []),
        },
        "metrics": {
            "operational_gallery": operational,
            "leave_one_out": leave_one_out,
            "leave_one_out_temporal": temporal,
        },
        "rows": {
            "operational_gallery": operational_rows,
            "leave_one_out": loo_rows,
        },
        "notes": [
            "Operational gallery accuracy is a regression check, not generalization.",
            "Leave-one-out accuracy is the confidence ceiling used by the runtime engine.",
            "The dataset is one real Hardpoint map; Phase 3 is not production OCR yet.",
        ],
    }


def print_report(result: dict[str, Any]) -> None:
    print("# Scorebar OCR Evaluation")
    print()
    print(f"- dataset: {result['dataset']['path']}")
    print(f"- samples: {result['dataset']['samples']}")
    print(f"- label source: {result['dataset']['label_source']} by {result['dataset']['labeled_by']}")
    for name, metrics in result["metrics"].items():
        print()
        print(f"## {name}")
        print(f"- score exact accuracy: {metrics['score_exact_accuracy']:.4f}")
        print(f"- exact matches: {metrics['exact_matches']}/{metrics['samples']}")
        if "side_accuracy" in metrics:
            print(f"- side accuracy: {metrics['side_accuracy']:.4f}")
            print(f"- digit accuracy: {metrics['digit_accuracy']:.4f}")
            print(f"- mean confidence: {metrics['mean_confidence']:.4f}")
            print(f"- abstentions: {metrics['abstentions']}")
        if metrics["failures"]:
            print("- first failures:")
            for row in metrics["failures"][:5]:
                print(f"  - {row['sample_id']}: expected {row['expected']}, predicted {row['predicted']}")
    print()
    print("Notes:")
    for note in result["notes"]:
        print(f"- {note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_dataset_dir())
    parser.add_argument("--write-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(args.dataset)
    print_report(result)
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
