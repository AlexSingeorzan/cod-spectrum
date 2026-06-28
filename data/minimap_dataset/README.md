# Minimap detection dataset (YOLO)

The training path from the classical `ClassicalMinimapDetector` contract baseline
to a real player-marker model.

## Why
The classical detector localises player markers but can't reliably classify team
or reject all noise on a compressed broadcast minimap. A small YOLO model trained
on labelled crops fixes that and unlocks **detected** (not inferred) spawn flips
and per-player positions.

## Workflow

1. **Generate model-assisted pre-labels** (boxes are pre-seeded so you correct,
   not draw):
   ```bash
   .venv/bin/python scripts/build_minimap_dataset.py --vod data/videos/lat_van.mp4 --n 300
   ```
   → `images/*.png` + `labels/*.txt` (YOLO `class cx cy w h`, normalised).

2. **Correct** the boxes in any YOLO tool (Label Studio, Roboflow, labelImg).
   - Classes: `0 observed_player`, `1 enemy_player`.
   - **Never label hidden opponents** — only the observed team + radar-visible
     enemies are real on a broadcast minimap.
   - Add maps/teams/resolutions for generalisation.

3. **Split** `images/` + `labels/` into `train/` and `val/` (≈85/15).

4. **Train**:
   ```bash
   pip install ultralytics
   yolo detect train data=data/minimap_dataset/data.yaml model=yolov8n.pt imgsz=256 epochs=120
   ```

5. **Drop it in**: implement a `YoloMinimapDetector` satisfying the same
   `MinimapDetector` contract (`read_frame(...) -> MinimapFrameResult`, plus the
   compact `detect(frame, hud_profile) -> list[dict]` compatibility method). The
   `PositionEvent` schema, occupancy heatmap, spawn derivation, persistence and
   UI are unchanged — only marker detection swaps from color/shape to the model.

6. **Contract check**:
   ```bash
   make minimap-eval
   ```
   This synthetic fixture verifies metadata, evidence, bounding boxes,
   visibility discipline, and threshold abstention. It is not real broadcast
   accuracy.

## Note
`images/` and `labels/` are gitignored (large / regenerable). Only this README and
`data.yaml` are tracked. Regenerate locally with the command above.
