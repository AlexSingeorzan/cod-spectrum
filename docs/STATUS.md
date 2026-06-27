# COD Spectrum Status

Last audited: 2026-06-27

This is the current engineering status after Phase 5 kill-type-recognition
scaffolding. It is the checkpoint required before continuing Phase 6+ work.

## Completed Systems

- Universal event architecture: `GameEvent` envelope, fact/insight separation,
  typed payloads, provenance, evidence, confidence bounds, deterministic event
  ids, and tests.
- Event persistence: pipeline outputs are persisted as `game_events`; reports,
  dashboard, and API read the unified event stream.
- Offline vertical slice: fixture processing, score timeline, Hardpoint
  scoring-flow analytics, clips, reports, scheduler scaffolding, API, and UI.
- Score OCR baseline: `CdlScorebarOcrEngine` is implemented, documented, and
  evaluated on labelled LAT/VAN scorebar crops. It is not production-ready and
  is not default.
- Panel kill spine: `PanelKillCounter` reads the clean top-panel K/D counters
  and is exact against the verified LAT/VAN post-game card. This subsystem is
  stable and should not be changed while Phase 4 killfeed content work proceeds.
- Killfeed detection: `KillfeedDetector` localises killfeed rows and collapses
  flicker into candidate onsets with evidence and confidence.
- Killfeed segmentation: `KillfeedSegmenter` splits detected row crops into
  attacker, weapon, victim, and optional indicator segment crops when the layout
  is clear.
- Killfeed content contract: `KillfeedContentReader` can train from labelled
  killfeed rows and emit `KillEvent`, `DeathEvent`, kill-type `WeaponEvent`,
  and `TradeEvent`. It correctly abstains when labels are missing.
- Synthetic killfeed content sample: demonstrates kill_type/headshot/trade event
  emission without claiming real broadcast accuracy.
- Kill-type recognition scaffold: `KillTypeRecognizer` is independent from name
  OCR, compares template and histogram nearest-neighbour baselines, returns
  `kill_type=null` when labels/confidence are insufficient, and can emit
  evidence-backed events from accepted predictions.
- Kill-type icon dataset: `data/kill_type_dataset/` contains `120` Stage B
  killfeed icon crops copied only from manifest-referenced segment metadata.
- Synthetic kill-type recognition sample: demonstrates event emission and
  baseline comparison without claiming real broadcast accuracy.
- Minimap classical baseline: localises some minimap markers and writes
  occupancy heatmaps. It is not player-resolved and not production-grade.
- What-If Lab: synthetic counterfactual hardpoint model exists and is labelled
  as synthetic.

## Partially Complete Systems

- Score OCR: real baseline exists, but leave-one-out exact score accuracy is
  only `10/21` (`0.4762`) and temporal leave-one-out is `11/21` (`0.5238`).
- Killfeed recognition:
  - Stage A detector exists.
  - Stage B segmentation exists and writes field-level boxes/crops, but has no
    manually reviewed real field-box accuracy yet.
  - Stage C OCR is not implemented for real killfeed names.
  - Stage D kill-type recognition has a label-gated baseline/evaluator, but no
    real labelled accuracy yet.
  - Stage E event builder exists for labelled or model-read content only.
- Killfeed real dataset: `data/killfeed_dataset/` has `245` candidate rows, but
  `0` content-labelled rows, so content accuracy cannot be measured yet.
- Kill-type real dataset: `data/kill_type_dataset/` has `120` icon crops, but
  `0` labelled kill-type classes, so real kill-type accuracy cannot be measured
  yet.
- Minimap intelligence: dataset scaffold has six labelled images and a classical
  detector, but no trained player-resolved model, trajectories, velocity,
  heading, or map-control graph.
- Tactical analytics: existing Hardpoint break/retake and coach views are
  scoring-flow heuristics, not full tactical reasoning from kills + positions.
- Autonomous orchestration: scheduler and queue foundations exist, but full CDL
  schedule tracking, live match detection, VOD acquisition, and notification
  workflows are not complete.

## Broken Or Blocked Systems

- Real killfeed name reading is blocked by missing labels. The content reader
  has no right to emit real names until `annotations.jsonl` contains human labels
  or a trained OCR model is evaluated.
- Real kill-type recognition is blocked by missing labels. The independent
  kill-type recognizer exists, but it has no right to emit real kill types until
  `data/kill_type_dataset/annotations.jsonl` contains reviewed labels and
  evaluation metrics.
- Killfeed segmentation readiness is partial: `120/245` rows have complete
  attacker+weapon+victim boxes. The remaining rows stay null.
- Model metadata is inconsistent across older systems. The prompt requires
  every model to expose training dataset, latency, failure reason,
  fallback_used, and evaluation metrics; newer modules expose versions and
  confidence, but not the full contract everywhere.
- Some local generated `.DS_Store` files exist under `data/`; they are ignored
  and should not be committed.

## Technical Debt And Architectural Weaknesses

- Detector output schemas are ad hoc dictionaries in several modules. A shared
  `ModelOutput` / `DetectionResult` contract would make model metadata,
  latency, failure reasons, and fallback handling enforceable.
- Killfeed detection, segmentation, OCR, and content classification are not yet
  cleanly separated as independent stages with separate eval outputs.
- The current killfeed content baseline is nearest-neighbour over whole row
  crops. That is acceptable as a contract test, but not a scalable recognition
  design.
- The current kill-type baseline is nearest-neighbour over icon crops. That is a
  useful transparent baseline, but the design still expects a small CNN once
  enough labelled examples exist.
- Dataset mutation is mostly manual JSONL editing. The project needs a safe
  review tool that validates labels, tracks human review status, and emits
  summary metrics.
- No event fusion graph exists yet. Derived tactical insights cannot currently
  trace causal chains beyond `derived_from` event ids.
- Audio/listen-in intelligence is still greenfield.

## Test Coverage

Fresh verification:

```bash
make scorebar-ocr-eval
make killfeed-content-eval
make kill-type-eval
.venv/bin/python -m pytest -q
```

Current result:

- `134 passed`
- Scorebar OCR eval: operational gallery `21/21`, leave-one-out `10/21`
  (`0.4762`), temporal leave-one-out `11/21` (`0.5238`).
- Killfeed segmentation eval: `120/245` rows with complete attacker+weapon+
  victim boxes (`0.4898`), `120` weapon crops, `23` headshot candidate crops,
  no real accuracy claim.
- Killfeed content eval: `245` candidates, `0` content-labelled rows,
  no content-reader accuracy claim.
- Kill-type eval: `120` icon crops, `0` labelled kill-type icons,
  no kill-type-recognition accuracy claim.

Coverage exists for:

- event schema invariants
- event store and report projection
- score OCR baseline
- killfeed detector/evaluator scaffold
- killfeed content reader and synthetic event expansion
- kill-type dataset builder, recognizer abstention, synthetic baseline
  comparison, killstreak support, and event emission
- panel kill spine
- minimap classical detector
- processor/report/dashboard contracts

Missing coverage:

- real labelled killfeed content accuracy
- manually reviewed killfeed segmentation-box accuracy
- real labelled kill-type classifier metrics
- player-resolved minimap detections
- event fusion graph and tactical reasoning chains
- audio/listen-in subsystem

## Current Blockers

1. Real killfeed content labels do not exist.
2. Kill-type recognition cannot be honestly evaluated until icon labels exist.
3. Stage B segmentation needs human review before production kill-type
   classifier comparisons are meaningful.
4. A shared model-result contract is needed before productionizing multiple
   detectors.

## Suggested Improvements

- Extend the annotation dataset with per-field crop paths and review metadata:
  `review_status`, `reviewed_by`, `reviewed_at`, and `failure_reason`.
- Add an annotation helper CLI that can list unlabeled rows, show crop paths,
  apply labels safely, validate schema, and summarize readiness.
- Label `data/kill_type_dataset/annotations.jsonl` with coarse kill types and
  mark unclear crops rather than forcing classes.
- After labels exist, compare the template/histogram baselines against a small
  CNN on the same split.
- Introduce a shared model output contract for all detectors.

## Next Recommended Phase

Do not touch the panel kill spine.

The next build is **real kill-type icon labelling and classifier evaluation**:

1. Review the `120` icon crops in `data/kill_type_dataset/icons/`.
2. Fill `valid_kill_type`, `kill_type`, and `unclear` in
   `data/kill_type_dataset/annotations.jsonl`. `exact_weapon` is optional
   metadata only.
3. Rerun `make kill-type-eval` to compare template and histogram baselines.
4. Train/evaluate the small CNN candidate from
   `docs/KILL_TYPE_RECOGNITION_DESIGN.md` only after there are enough labels per
   class.

This keeps `PanelKillCounter` as the kill-count truth and uses killfeed content
only as evidence-backed enrichment.
