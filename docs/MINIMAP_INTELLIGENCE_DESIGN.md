# Minimap Intelligence Design

Status: Phase 6 contract scaffold implemented; no player-resolved production
model yet.

## Goal

The minimap subsystem must eventually produce player positions, trajectories,
velocity, heading, map control, spawn pressure, route clusters, crossfire
signals, and future route/spawn predictions.

The first production-safe step is the detector contract. Every minimap detection
must preserve:

- model name and version
- training dataset
- confidence
- latency
- failure reason
- fallback flag
- evaluation metrics
- frame number and timestamp
- frame/crop evidence
- normalized bounding box
- observed-team visibility
- human review status

The broadcast minimap does not expose neutral omniscient truth. It usually shows
the observed team and radar-visible enemies. Hidden opponents must never be
inferred.

## Current Implementation

`ClassicalMinimapDetector` remains the baseline detector. It localizes bright and
red minimap markers with classical CV and now returns a rich
`MinimapFrameResult`:

- `model_name=minimap_classical_marker_detector`
- `model_version=0.2.0`
- `training_dataset=none_classical_cv`
- per-frame latency
- per-detection normalized box and visibility
- visual evidence paths when supplied

Accepted detections can be converted to `PositionEvent` facts via
`position_events_from_minimap_result`. Low-confidence detections are withheld,
and no event is emitted without frame or crop evidence.

The older `detect(frame, hud_profile) -> list[dict]` method is retained for the
YOLO dataset seeder and returns the compact legacy fields.

## Evaluation

The repeatable contract evaluation is:

```bash
make minimap-eval
```

It writes `data/fixtures/minimap_contract/`:

- `frame.png`
- `minimap_crop.png`
- `detections.json`
- `events.jsonl`
- `eval_results.json`

Current synthetic contract result:

- detections: `2`
- position events: `2`
- visibility: `1` observed-team marker, `1` radar-visible enemy marker
- high-threshold events: `0`
- real broadcast accuracy: no claim

## Upgrade Path

1. Generate model-assisted YOLO pre-labels with `scripts/build_minimap_dataset.py`.
2. Correct labels manually; include `observed_player` and `enemy_player` only
   when visible on the broadcast minimap.
3. Train a small detector on labelled minimap crops.
4. Implement `YoloMinimapDetector` behind the same `read_frame`/`detect`
   contract.
5. Add trajectory stitching, velocity, heading, lane occupancy, map-control
   polygons, spawn pressure, and nearest-player graph services downstream of
   `PositionEvent`.

## Limitations

- The classical detector is a contract baseline, not production minimap
  intelligence.
- It is not player-resolved.
- Team classification is low-confidence without a labelled detector.
- No trajectories, velocity, route clustering, spawn prediction, or map-control
  graph exist yet.
