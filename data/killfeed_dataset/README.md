# Killfeed detection dataset (Phase 4, deliverable 1)

Model-assisted annotation scaffold for the broadcast **killfeed** (the
attacker → weapon → victim kill-notification rows). The classical
`KillfeedDetector` pre-labels candidate kill onsets; a human then verifies them.
**No identities are invented here** — every label starts `null`.

## Files

- `manifest.json` — build metadata (source VOD/URL, sample window, detector version, counts, honesty note).
- `annotations.jsonl` — one candidate kill onset per line: timestamp, row-crop path, detector box + confidence, low-confidence team-colour hints, and an **empty `label`** slot.
- `rows/*.png` — the cropped kill-row strips to label (tracked).
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
make killfeed-content-eval   # attacker/victim/weapon reader eval once content-labelled
```

## Honesty

This is **not** verified kill data yet. It is unverified classical-CV candidates plus
empty label slots. The detector localises kill rows (timing/count). The Phase-4
content reader (`KillfeedContentReader`, `killfeed_content_knn@0.1.0`) is now wired
to train from these labels and emit real `DeathEvent` / `WeaponEvent` / `TradeEvent`
facts, but it abstains until labels exist.

Current committed status:

- Detection candidates: `245`
- Content-labelled rows: `0`
- Content-reader accuracy: no claim

Until `annotations.jsonl` is labelled, `eval_killfeed.py` and
`eval_killfeed_content.py` report no real accuracy.
