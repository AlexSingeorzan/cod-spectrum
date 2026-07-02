# Killfeed detection dataset (Phase 4)

Model-assisted annotation scaffold for the broadcast **killfeed** (the
attacker → weapon → victim kill-notification rows). The classical
`KillfeedDetector` pre-labels candidate kill onsets; a human then verifies them.
**No identities are invented here** — every label starts `null`.

## Files

- `manifest.json` — build metadata (source VOD/URL, sample window, detector version, counts, honesty note).
- `annotations.jsonl` — one candidate kill onset per line: timestamp, row-crop path, detector box + confidence, low-confidence team-colour hints, and an **empty `label`** slot.
- `rows/*.png` — the cropped kill-row strips to label (tracked).
- `segments.jsonl` — Stage B field boxes for attacker, weapon, victim, and optional indicators. This is readiness/evidence metadata, not OCR accuracy.
- `segments/*.png` — field-level crops produced from `rows/*.png` (tracked).
- `segmentation_eval_results.json` — current Stage B readiness metrics.
- `content_eval_results.json` — current Stage C/D content-reader readiness metrics.
- `regions/*.png` — full killfeed-region context crops (gitignored; rebuild from the VOD).

The committed set was built from the LA Thieves vs Vancouver Surge Hacienda
Hardpoint (`data/videos/lat_van.mp4`, gitignored). Rebuild:

```bash
make killfeed-dataset        # full Hardpoint window, 2 fps
# or: .venv/bin/python scripts/build_killfeed_dataset.py --vod <vod> --start <s> --end <e>
```

## How to label

Open `annotations.jsonl` and, per row, set the `label` fields:

| field | meaning |
|---|---|
| `valid_kill` | `true` = a real kill, `false` = a false positive. **Required for detection precision/recall.** |
| `attacker`, `victim` | gamertag as shown (feeds the future content reader → `DeathEvent`) |
| `attacker_team`, `victim_team` | team tag, e.g. `LAT` / `VAN` |
| `weapon` | weapon name/icon class (feeds `WeaponEvent`) |
| `headshot` | `true` if a headshot marker is shown |
| `is_trade` | `true` if this kill traded a recent teammate death (feeds `TradeEvent`) |

Also set `label_source` (`manual_label`) and `labeled_by` (your name) — every label
records its source.

**Missed kills (recall):** the detector misses some kills (e.g. faded rows). Add each
missed kill as a new line with `"detector": "manual_added"` and `valid_kill: true`, so
recall is measurable.

Then:

```bash
make killfeed-eval           # precision / recall / F1 once labelled; honest "no claim" until then
make killfeed-segments       # Stage B field crops from row crops
make killfeed-segment-eval   # segmentation readiness, not OCR/classifier accuracy
make killfeed-content-eval   # attacker/victim/weapon reader eval once content-labelled
```

## Stage B segmentation

`KillfeedSegmenter` (`killfeed_segmenter_classical@0.1.0`) splits a detected row into
field-level evidence crops:

- attacker text
- weapon icon
- victim text
- optional headshot marker
- optional assist/special indicators

It returns `null` for fields whose layout is not clear. It does not use fixed-layout
fallbacks and does not classify weapons.

Current committed readiness:

- Rows: `245`
- Complete attacker+weapon+victim segment boxes: `120/245` (`0.4898`)
- Weapon segment crops: `120`
- Headshot candidate crops: `23`
- Real segmentation accuracy: no claim yet, because there are no human-labelled
  field boxes.

## Labels (2026-07-01)

All 245 rows are labelled: every crop was upscaled 4x and read visually (contact
sheets), teams assigned from the verified roster, `is_trade` computed from the
labelled kill sequence (6 s window — rule recorded per row), and re-detections of
a persisting row marked `duplicate_onset_of`. Labeller recorded per row.

- Valid kill rows: `110` (135 false positives) → detector precision **0.449**
- Unique kills (dedup families): `83` — 3 with unreadable attacker names
- Content rows (attacker+victim readable): `107`; headshot rows `12`; trade kills `10`;
  one grenade kill (`kf_0188_559`)
- **Recall vs the verified panel kill spine: 0.3081** (53/172 panel kills have an
  identity-matching labelled row within ±10 s) — `eval_killfeed.py --panel-events`

Measured content-reader baseline (`killfeed_content_knn@0.2.0`, leave-one-out):

- At the 0.6 deployment threshold the reader **abstains on all 107 rows** — the raw
  HSV row embedding is background-dominated, so unseen-row identity reads are not
  trustworthy and the gate correctly holds.
- Forced-read diagnostic (threshold 0): attacker `0.36`, victim `0.39`,
  kill_type `0.98`, headshot `0.88`, is_trade `0.89`.
- The labels themselves (via `manual_read_from_row`) are the production path for
  this match's `KillEvent`/`DeathEvent`/`WeaponEvent`/`TradeEvent` enrichment; a
  text/icon-crop OCR model is the upgrade path for auto-reading new VODs.

The panel counter remains the kill-count spine; the killfeed layer contributes
identity corroboration, kill-type, headshot, and trade structure.
