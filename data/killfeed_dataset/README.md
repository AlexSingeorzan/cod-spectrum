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

## Honesty

This is **not** verified kill data yet. It is unverified classical-CV candidates plus
empty label slots. The detector localises kill rows (timing/count). The Phase-4
content reader (`KillfeedContentReader`, `killfeed_content_knn@0.1.0`) is now wired
to train from these labels and emit real `DeathEvent` / `WeaponEvent` / `TradeEvent`
facts, but it abstains until labels exist.

Current committed status:

- Detection candidates: `245`
- Complete core segments: `120`
- Content-labelled rows: `0`
- Content-reader accuracy: no claim

Until `annotations.jsonl` is labelled, `eval_killfeed.py` and
`eval_killfeed_content.py` report no real accuracy. Until field boxes are manually
reviewed, `eval_killfeed_segments.py` reports readiness only.
