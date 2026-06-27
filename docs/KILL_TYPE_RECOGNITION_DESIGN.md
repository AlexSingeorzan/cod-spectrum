# Kill-Type Recognition Design

Status: Phase 5 coarse classifier scaffold implemented, no production classifier
yet.

## Goal

Classify each killfeed icon crop into a coarse kill cause:

- `gun`
- `grenade`
- `melee`
- `fall_damage`
- `suicide`
- `environment`
- `objective`
- `killstreak`
- `unknown`

The recognizer returns `kill_type`, `confidence`, evidence crop metadata, and
model version. If confidence is below the configured threshold, it returns
`kill_type=null`. It must not guess.

Downstream analytics consume `kill_type`, not exact weapon names. Exact weapons
remain optional metadata (`weapon`) so a future MCW/C9/AMES-style classifier can
plug in without changing the event schema.

## Current Inputs

Phase 4 Stage B produces killfeed icon crops from row segmentation:

- source: `data/killfeed_dataset/segments.jsonl`
- crop directory: `data/killfeed_dataset/segments/`
- current readiness: `120/245` rows have attacker+weapon+victim core boxes
- kill-type dataset: `data/kill_type_dataset/` with `120` copied icon crops
- current labelled kill-type classes: `0`

No classifier should be trained or promoted until kill-type labels exist.

## Candidate Approaches

| Approach | Pros | Cons | Current fit |
|---|---|---|---|
| Template matching | Fast, deterministic, CPU-only, easy to inspect | Brittle to compression, scale, tint, icon updates; weak confidence calibration | Good baseline only |
| Histogram nearest neighbour | Very cheap, tolerant of tiny shifts | Weak for similar white HUD icons | Secondary baseline only |
| CNN classifier | Fast at inference, trainable on tiny icon crops, easy to version, can calibrate confidence | Needs labelled examples per kill type; data augmentation required | Best first real model after labels |
| CLIP embeddings | Can be few-shot and open-vocabulary in theory | Tiny monochrome HUD icons are far from CLIP's natural-image sweet spot; model is heavy | Evaluate as experiment only |
| Few-shot metric learning | Useful when labels/class are scarce; naturally supports later exact-weapon classes | More moving pieces than a plain CNN; still needs support examples | Useful later |

## Recommended Path

1. Build a kill-type dataset from Stage B icon crops. **Done.**
2. Validate the dataset and create a contact sheet with
   `scripts/review_kill_type_dataset.py`. **Done.**
3. Human-label visible kill types, use `unknown` only for reviewed valid causes
   outside/indistinguishable from the named classes, and mark ambiguous crops
   `unclear=true`.
4. Evaluate template/histogram nearest-neighbour baselines on the same labelled
   split. **Wired, blocked by real labels.**
5. Train a lightweight CNN once there are enough labels per category.
6. Promote only the model with the best accuracy/latency/maintenance tradeoff on
   held-out labelled crops.
7. Add exact weapon classification later as optional metadata only.

## Required Model Contract

Every `KillTypeRecognizer` output must include:

- `model_name`
- `model_version`
- `training_dataset`
- `confidence`
- `latency_ms`
- `failure_reason`
- `fallback_used`
- `evaluation_metrics`
- `kill_type` or `null`
- `evidence_crop_path`
- `evidence_box`
- optional `exact_weapon`

Accepted event payloads carry:

```json
{
  "kill_type": "grenade",
  "confidence": 0.97
}
```

Low-confidence outputs carry:

```json
{
  "kill_type": null
}
```

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

- Current real dataset has no kill-type labels.
- Some killfeed detections are false positives; they must not be used as
  training samples until the crop is human reviewed.
- Stage B complete-core segmentation is `120/245`, so the dataset is currently
  useful for bootstrapping, not production.
- Icons may change by game title, season, broadcast overlay, or CDL graphics
  package.

## Exit Criteria For Phase 5

- `data/kill_type_dataset/` exists with icon crops and source metadata. **Done.**
- The category list includes `killstreak`. **Done.**
- Review tooling can validate JSONL, summarize readiness, list rows, apply one
  reviewed label, and generate a contact sheet. **Done.**
- At least two approaches are evaluated on the same split. **Wired for template
  and histogram baselines; real split is blocked by labels.**
- Low-confidence outputs return `kill_type=null`.
- Synthetic sample output emits a kill-type event with visual evidence.
- Tests cover dataset loading, classifier abstention, event construction,
  evaluation, and killstreak support.

The remaining promotion gate is labelled real data, not code structure.
