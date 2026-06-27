# Weapon Recognition Design

Status: Phase 5 scaffold implemented, no production classifier yet.

## Goal

Classify the weapon icon shown in a killfeed row and emit weapon evidence into:

- `KillEvent.weapon`
- `DeathEvent.weapon`
- `WeaponEvent`

The weapon recognizer must be independent from player-name OCR. OCR failure must
not imply weapon failure, and weapon failure must not imply OCR failure.

## Current Inputs

Phase 4 Stage B now produces weapon-icon crops from killfeed rows:

- source: `data/killfeed_dataset/segments.jsonl`
- crop directory: `data/killfeed_dataset/segments/`
- current readiness: `120/245` rows have attacker+weapon+victim core boxes
- weapon dataset: `data/weapon_dataset/` with `120` copied icon crops
- current labelled weapon classes: `0`

No classifier should be trained or promoted until weapon labels exist.

## Candidate Approaches

| Approach | Pros | Cons | Current fit |
|---|---|---|---|
| Template matching | Fast, deterministic, CPU-only, easy to inspect | Brittle to compression, scale, tint, icon updates; weak confidence calibration | Good baseline only |
| CNN classifier | Fast at inference, trainable on tiny icon crops, easy to version, can calibrate confidence | Needs labelled examples per weapon; data augmentation required | Best first real model after labels |
| Vision Transformer | Strong with enough data and pretraining | Overkill for tiny crops; heavier dependency and annotation needs | Not first choice |
| CLIP embeddings | Can be few-shot and open-vocabulary in theory | Tiny monochrome HUD icons are far from CLIP's natural-image sweet spot; model is heavy | Evaluate as experiment only |
| Few-shot metric learning | Useful when labels/class are scarce; naturally supports new weapons | More moving pieces than a plain CNN; still needs support examples | Useful once a small labelled set exists |

## Recommended Path

1. Build a weapon-icon dataset from Stage B crops. **Done.**
2. Human-label visible weapon classes and mark unclear crops as `unknown`.
3. Evaluate template/histogram nearest-neighbour baselines as the minimum
   transparent baselines. **Wired, blocked by real labels.**
4. Train a lightweight CNN once there are enough labels per weapon class.
5. Evaluate CLIP/few-shot embeddings as optional experiments, not default.
6. Promote only the model with the best accuracy/latency/maintenance tradeoff on
   held-out labelled crops.

The first production candidate should be a **small CNN classifier**, not template
matching, because COD broadcast icons are compressed, anti-aliased, and can shift
between overlay revisions. Template matching remains useful as a transparent
baseline and failure detector.

## Required Model Contract

Every `WeaponRecognizer` output must include:

- `model_name`
- `model_version`
- `training_dataset`
- `confidence`
- `latency_ms`
- `failure_reason`
- `fallback_used`
- `evaluation_metrics`
- `weapon` or `null`
- `evidence.crop_path`
- `evidence.box`

If confidence is below threshold, return `weapon = null`.

## Evaluation Metrics

Minimum metrics before promotion:

- top-1 accuracy
- macro F1
- per-class precision/recall
- abstention rate
- unknown/unclear handling
- mean latency per crop
- confusion matrix
- failure examples with crop paths

## Known Limitations

- Current real dataset has no weapon labels.
- Some killfeed detections are false positives; they must not be used as weapon
  training samples until `valid_kill=true`.
- Stage B complete-core segmentation is `120/245`, so the dataset is currently
  useful for bootstrapping, not production.
- Weapon icons may change by game title, season, broadcast overlay, or CDL
  graphics package.

## Exit Criteria For Phase 5

- `data/weapon_dataset/` exists with icon crops and source metadata. **Done.**
- At least two approaches are evaluated on the same split. **Wired for template
  and histogram baselines; real split is blocked by labels.**
- Low-confidence outputs return `null`.
- `WeaponEvent` sample output exists.
- Tests cover dataset loading, classifier abstention, and event construction.
- Documentation includes metrics, CLI usage, known limitations, and failure
  examples.

The remaining promotion gate is labelled real data, not code structure.
