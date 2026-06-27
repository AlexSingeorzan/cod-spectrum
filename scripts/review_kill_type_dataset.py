"""Review and validate the Phase 5 kill-type icon dataset.

This is deliberately a small CLI, not a model. It prevents the next phase from
depending on hand-edited JSONL: reviewers can summarize readiness, generate a
contact sheet, list rows, validate schema, and apply one reviewed label at a
time.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.kill_type_recognition import (  # noqa: E402
    KILL_TYPE_CATEGORIES,
    default_dataset_dir,
    labeled_kill_type_rows,
)

REVIEW_STATUSES = {"unreviewed", "reviewed"}
LABEL_SOURCES = {"unlabeled", "manual_label", "human_verified"}
RowStatus = Literal["all", "unreviewed", "reviewed", "labelled", "unclear", "invalid", "missing_crop"]


@dataclass(frozen=True)
class ValidationReport:
    dataset: Path
    rows: list[dict[str, Any]]
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_dataset(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = dataset_dir / "manifest.json"
    annotations_path = dataset_dir / "annotations.jsonl"
    manifest = json.loads(manifest_path.read_text())
    rows = _read_jsonl(annotations_path)
    return manifest, rows


def write_rows(dataset_dir: Path, rows: list[dict[str, Any]]) -> None:
    annotations_path = dataset_dir / "annotations.jsonl"
    annotations_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )


def _bool_or_none(value: Any) -> bool:
    return value is True or value is False or value is None


def _row_icon_path(dataset_dir: Path, row: dict[str, Any]) -> Path:
    image_path = Path(row.get("icon_image", ""))
    if image_path.is_absolute():
        return image_path
    return dataset_dir / image_path


def validate_dataset(dataset_dir: Path) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    manifest, rows = load_dataset(dataset_dir)
    categories = tuple(manifest.get("categories") or KILL_TYPE_CATEGORIES)
    expected_categories = tuple(KILL_TYPE_CATEGORIES)
    if categories != expected_categories:
        errors.append(
            f"manifest categories differ from canonical categories: {categories} != {expected_categories}"
        )

    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id") or f"row_{index}")
        prefix = f"{row_id}: "
        if row_id in seen_ids:
            errors.append(prefix + "duplicate id")
        seen_ids.add(row_id)

        label = row.get("label")
        if not isinstance(label, dict):
            errors.append(prefix + "label must be an object")
            continue

        review_status = row.get("human_review_status")
        label_source = row.get("label_source")
        labeled_by = row.get("labeled_by")
        valid = label.get("valid_kill_type")
        kill_type = label.get("kill_type")
        unclear = label.get("unclear")
        exact_weapon = label.get("exact_weapon")

        if review_status not in REVIEW_STATUSES:
            errors.append(prefix + f"human_review_status must be one of {sorted(REVIEW_STATUSES)}")
        if label_source not in LABEL_SOURCES:
            errors.append(prefix + f"label_source must be one of {sorted(LABEL_SOURCES)}")
        if not _bool_or_none(valid):
            errors.append(prefix + "label.valid_kill_type must be true/false/null")
        if not _bool_or_none(unclear):
            errors.append(prefix + "label.unclear must be true/false/null")
        if kill_type is not None and kill_type not in KILL_TYPE_CATEGORIES:
            errors.append(prefix + f"label.kill_type must be one of {list(KILL_TYPE_CATEGORIES)} or null")
        if exact_weapon is not None and not isinstance(exact_weapon, str):
            errors.append(prefix + "label.exact_weapon must be a string or null")

        if label_source == "unlabeled":
            if labeled_by is not None:
                errors.append(prefix + "unlabeled rows must not have labeled_by")
            if review_status != "unreviewed":
                errors.append(prefix + "unlabeled rows must be human_review_status=unreviewed")
            if any(label.get(field) is not None for field in ("valid_kill_type", "kill_type", "exact_weapon", "unclear")):
                errors.append(prefix + "unlabeled rows must not contain partial labels")
        else:
            if not labeled_by:
                errors.append(prefix + "reviewed rows must include labeled_by")
            if review_status != "reviewed":
                errors.append(prefix + "reviewed labels must set human_review_status=reviewed")

        if valid is True:
            if not kill_type:
                errors.append(prefix + "valid_kill_type=true requires kill_type")
            if unclear is True:
                errors.append(prefix + "valid_kill_type=true cannot also be unclear")
        if valid is False and kill_type is not None:
            errors.append(prefix + "valid_kill_type=false must not set kill_type")
        if unclear is True and valid is True:
            errors.append(prefix + "unclear=true must not be a valid training row")

        icon_path = _row_icon_path(dataset_dir, row)
        if not icon_path.exists():
            errors.append(prefix + f"missing icon crop: {_rel(icon_path)}")
        elif icon_path.stat().st_size == 0:
            errors.append(prefix + f"empty icon crop: {_rel(icon_path)}")

        if row.get("segment_box") is None:
            warnings.append(prefix + "missing segment_box evidence")
        if row.get("segment_confidence") is None:
            warnings.append(prefix + "missing segment_confidence")

    return ValidationReport(dataset=dataset_dir, rows=rows, errors=errors, warnings=warnings)


def row_matches_status(dataset_dir: Path, row: dict[str, Any], status: RowStatus) -> bool:
    if status == "all":
        return True
    label = row.get("label", {})
    if status == "unreviewed":
        return row.get("human_review_status") == "unreviewed"
    if status == "reviewed":
        return row.get("human_review_status") == "reviewed"
    if status == "labelled":
        return row in labeled_kill_type_rows([row])
    if status == "unclear":
        return label.get("unclear") is True
    if status == "missing_crop":
        return not _row_icon_path(dataset_dir, row).exists()
    if status == "invalid":
        single_dir = dataset_dir
        report = validate_rows(single_dir, [row])
        return not report.ok
    raise ValueError(f"unknown status: {status}")


def validate_rows(dataset_dir: Path, rows: list[dict[str, Any]]) -> ValidationReport:
    temp_errors: list[str] = []
    temp_warnings: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        row_id = str(row.get("id") or "")
        if row_id in seen_ids:
            temp_errors.append(f"{row_id}: duplicate id")
        seen_ids.add(row_id)
        label = row.get("label", {})
        if not isinstance(label, dict):
            temp_errors.append(f"{row_id}: label must be an object")
            continue
        review_status = row.get("human_review_status")
        label_source = row.get("label_source")
        labeled_by = row.get("labeled_by")
        valid = label.get("valid_kill_type")
        if review_status not in REVIEW_STATUSES:
            temp_errors.append(f"{row_id}: invalid human_review_status")
        if label_source not in LABEL_SOURCES:
            temp_errors.append(f"{row_id}: invalid label_source")
        if not _bool_or_none(valid):
            temp_errors.append(f"{row_id}: invalid valid_kill_type")
        if not _bool_or_none(label.get("unclear")):
            temp_errors.append(f"{row_id}: invalid unclear")
        kill_type = label.get("kill_type")
        if kill_type is not None and kill_type not in KILL_TYPE_CATEGORIES:
            temp_errors.append(f"{row_id}: invalid kill_type")
        if label_source == "unlabeled":
            if labeled_by is not None:
                temp_errors.append(f"{row_id}: unlabeled row has labeled_by")
            if review_status != "unreviewed":
                temp_errors.append(f"{row_id}: unlabeled row is not unreviewed")
            if any(label.get(field) is not None for field in ("valid_kill_type", "kill_type", "exact_weapon", "unclear")):
                temp_errors.append(f"{row_id}: unlabeled row contains partial labels")
        elif not labeled_by:
            temp_errors.append(f"{row_id}: reviewed row missing labeled_by")
        if valid is True and not kill_type:
            temp_errors.append(f"{row_id}: valid row missing kill_type")
        if valid is False and kill_type is not None:
            temp_errors.append(f"{row_id}: invalid row has kill_type")
        if valid is True and label.get("unclear") is True:
            temp_errors.append(f"{row_id}: valid row is also unclear")
        if not _row_icon_path(dataset_dir, row).exists():
            temp_errors.append(f"{row_id}: missing icon crop")
    return ValidationReport(dataset=dataset_dir, rows=rows, errors=temp_errors, warnings=temp_warnings)


def summarize(dataset_dir: Path) -> dict[str, Any]:
    report = validate_dataset(dataset_dir)
    rows = report.rows
    labelled = labeled_kill_type_rows(rows)
    by_type: dict[str, int] = {}
    for row in labelled:
        kill_type = row["label"]["kill_type"]
        by_type[kill_type] = by_type.get(kill_type, 0) + 1
    unclear = sum(1 for row in rows if row.get("label", {}).get("unclear") is True)
    reviewed = sum(1 for row in rows if row.get("human_review_status") == "reviewed")
    missing_crops = sum(1 for row in rows if not _row_icon_path(dataset_dir, row).exists())
    return {
        "dataset": _rel(dataset_dir),
        "total_rows": len(rows),
        "reviewed_rows": reviewed,
        "unreviewed_rows": len(rows) - reviewed,
        "labelled_training_rows": len(labelled),
        "unclear_rows": unclear,
        "missing_crops": missing_crops,
        "kill_type_counts": dict(sorted(by_type.items())),
        "categories": list(KILL_TYPE_CATEGORIES),
        "validation": {
            "ok": report.ok,
            "errors": report.errors,
            "warnings": report.warnings,
        },
    }


def select_rows(dataset_dir: Path, status: RowStatus, limit: int | None = None) -> list[dict[str, Any]]:
    _manifest, rows = load_dataset(dataset_dir)
    selected = [row for row in rows if row_matches_status(dataset_dir, row, status)]
    if limit is not None:
        return selected[:limit]
    return selected


def apply_label(
    dataset_dir: Path,
    sample_id: str,
    *,
    reviewed_by: str,
    valid: bool,
    kill_type: str | None,
    unclear: bool,
    exact_weapon: str | None,
    label_source: str,
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    if label_source not in {"manual_label", "human_verified"}:
        raise ValueError("set-label requires label_source=manual_label or human_verified")
    if valid and not kill_type:
        raise ValueError("valid=true requires --kill-type")
    if valid and kill_type not in KILL_TYPE_CATEGORIES:
        raise ValueError(f"--kill-type must be one of {list(KILL_TYPE_CATEGORIES)}")
    if not valid and kill_type is not None:
        raise ValueError("valid=false cannot set --kill-type")
    if unclear and valid:
        raise ValueError("--unclear true cannot be combined with valid=true")
    if exact_weapon and not valid:
        raise ValueError("--exact-weapon is only allowed on valid rows")

    _manifest, rows = load_dataset(dataset_dir)
    for row in rows:
        if row.get("id") != sample_id:
            continue
        row["human_review_status"] = "reviewed"
        row["label_source"] = label_source
        row["labeled_by"] = reviewed_by
        row["reviewed_at"] = reviewed_at or datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        row["label"] = {
            "valid_kill_type": valid,
            "kill_type": kill_type if valid else None,
            "exact_weapon": exact_weapon if valid else None,
            "unclear": unclear,
        }
        candidate_report = validate_rows(dataset_dir, [row])
        if not candidate_report.ok:
            raise ValueError("; ".join(candidate_report.errors))
        write_rows(dataset_dir, rows)
        return row
    raise KeyError(f"sample id not found: {sample_id}")


def write_contact_sheet(
    dataset_dir: Path,
    out_path: Path,
    rows: list[dict[str, Any]],
    *,
    cell_w: int = 220,
    cell_h: int = 96,
    icon_w: int = 96,
    icon_h: int = 44,
    columns: int = 4,
) -> Path:
    if not rows:
        raise ValueError("no rows selected for contact sheet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_count = int(np.ceil(len(rows) / columns))
    sheet = np.full((rows_count * cell_h, columns * cell_w, 3), 245, dtype=np.uint8)
    for i, row in enumerate(rows):
        col = i % columns
        r = i // columns
        x = col * cell_w
        y = r * cell_h
        icon = cv2.imread(str(_row_icon_path(dataset_dir, row)), cv2.IMREAD_COLOR)
        if icon is None:
            icon = np.full((icon_h, icon_w, 3), 210, dtype=np.uint8)
            cv2.putText(icon, "missing", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
        else:
            icon = cv2.resize(icon, (icon_w, icon_h), interpolation=cv2.INTER_AREA)
        pad_x = x + 8
        pad_y = y + 8
        sheet[pad_y:pad_y + icon_h, pad_x:pad_x + icon_w] = icon
        label = row.get("label", {})
        review = row.get("human_review_status") or "?"
        text_color = (20, 20, 20)
        cv2.putText(sheet, str(row.get("id")), (x + 8, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)
        cv2.putText(
            sheet,
            f"t={row.get('video_timestamp_seconds')} {review}",
            (x + 8, y + 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            text_color,
            1,
        )
        if label.get("kill_type"):
            cv2.putText(sheet, str(label["kill_type"]), (x + 112, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def print_summary(summary: dict[str, Any]) -> None:
    print("# Kill-Type Dataset Review")
    print(f"- dataset: {summary['dataset']}")
    print(f"- rows: {summary['total_rows']}")
    print(f"- reviewed: {summary['reviewed_rows']} | unreviewed: {summary['unreviewed_rows']}")
    print(f"- labelled training rows: {summary['labelled_training_rows']}")
    print(f"- unclear rows: {summary['unclear_rows']} | missing crops: {summary['missing_crops']}")
    print(f"- validation: {'ok' if summary['validation']['ok'] else 'failed'}")
    if summary["kill_type_counts"]:
        print(f"- kill types: {summary['kill_type_counts']}")
    if summary["validation"]["errors"]:
        print("Errors:")
        for error in summary["validation"]["errors"]:
            print(f"- {error}")


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_dataset_dir())
    sub = parser.add_subparsers(dest="command", required=True)

    summary = sub.add_parser("summary", help="Print review readiness summary")
    summary.add_argument("--write-json", type=Path)

    sub.add_parser("validate", help="Validate annotation schema and crop evidence")

    list_cmd = sub.add_parser("list", help="List selected rows")
    list_cmd.add_argument("--status", choices=["all", "unreviewed", "reviewed", "labelled", "unclear", "invalid", "missing_crop"], default="unreviewed")
    list_cmd.add_argument("--limit", type=int, default=20)
    list_cmd.add_argument("--json", action="store_true")

    set_label = sub.add_parser("set-label", help="Apply one reviewed label")
    set_label.add_argument("--id", required=True)
    set_label.add_argument("--reviewed-by", required=True)
    set_label.add_argument("--valid", type=_parse_bool, required=True)
    set_label.add_argument("--kill-type", choices=list(KILL_TYPE_CATEGORIES))
    set_label.add_argument("--unclear", type=_parse_bool, default=False)
    set_label.add_argument("--exact-weapon")
    set_label.add_argument("--label-source", choices=["manual_label", "human_verified"], default="manual_label")
    set_label.add_argument("--reviewed-at")

    sheet = sub.add_parser("contact-sheet", help="Write a contact sheet for visual review")
    sheet.add_argument("--status", choices=["all", "unreviewed", "reviewed", "labelled", "unclear", "invalid", "missing_crop"], default="unreviewed")
    sheet.add_argument("--limit", type=int)
    sheet.add_argument("--out", type=Path, default=Path("data/kill_type_dataset/review_contact_sheet.png"))
    sheet.add_argument("--columns", type=int, default=4)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = Path(args.dataset)
    if args.command == "summary":
        result = summarize(dataset)
        print_summary(result)
        if args.write_json:
            args.write_json.parent.mkdir(parents=True, exist_ok=True)
            args.write_json.write_text(json.dumps(result, indent=2) + "\n")
        return 0 if result["validation"]["ok"] else 2
    if args.command == "validate":
        report = validate_dataset(dataset)
        if report.ok:
            print(f"validation ok: {_rel(dataset)} ({len(report.rows)} rows)")
            return 0
        print(f"validation failed: {_rel(dataset)}")
        for error in report.errors:
            print(f"- {error}")
        return 2
    if args.command == "list":
        rows = select_rows(dataset, args.status, args.limit)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                label = row.get("label", {})
                print(
                    f"{row.get('id')} t={row.get('video_timestamp_seconds')} "
                    f"status={row.get('human_review_status')} kill_type={label.get('kill_type')} "
                    f"crop={row.get('icon_image')}"
                )
        return 0
    if args.command == "set-label":
        row = apply_label(
            dataset,
            args.id,
            reviewed_by=args.reviewed_by,
            valid=args.valid,
            kill_type=args.kill_type,
            unclear=args.unclear,
            exact_weapon=args.exact_weapon,
            label_source=args.label_source,
            reviewed_at=args.reviewed_at,
        )
        print(json.dumps(row, indent=2, sort_keys=True))
        return 0
    if args.command == "contact-sheet":
        rows = select_rows(dataset, args.status, args.limit)
        out = write_contact_sheet(dataset, args.out, rows, columns=args.columns)
        print(f"wrote {len(rows)} rows to {_rel(out)}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
