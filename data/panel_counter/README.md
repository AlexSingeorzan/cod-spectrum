# Panel kill-counter cache (Phase 4 — the kill spine)

Cached scoreboard readings for `PanelKillCounter`. The broadcast's top team panels
show each player's running **kills/deaths**; that counter is clean and **monotonic**,
so it is the authoritative answer to *how many kills and who got them*.

`run_panel_counter.py` does the slow part once (OCR-reads both panels frame by frame)
and caches it here, so the counting logic + evaluation run offline and reproducibly.

## Files

- `readings.jsonl` — one line per sampled frame: `{"t", "a": [[k,d]|null ×4], "b": [...]}` (tracked; the eval runs off this, no VOD/OCR needed).
- `manifest.json` — source VOD/URL, window, fps, slot→gamertag map, totals, killfeed reconciliation.
- `events.jsonl` — the emitted `KillEvent` / `DeathEvent` stream (tracked).
- `crops/` — evidence panel crops at each kill frame (gitignored; rebuild from the VOD).
- `eval_results.json` — scoring vs the verified post-game card (written by `make panel-eval`).

## Rebuild / evaluate

```bash
make panel-counter   # OCR the panels over the local VOD -> readings.jsonl + events
make panel-eval      # score vs verified post-game totals + reconcile with the killfeed
```

## Ground truth

Scored against the **human-verified post-game card** in
`backend/app/services/hardpoint_breakdown.py` (`PLAYER_MAP_STATS`, `TEAM_KILL_CHECKPOINTS`):
LAT **106** kills (Scrap 25, HyDra 32, aBeZy 31, Nium 18), VAN **79** (Craze 15,
Mamba 19, Lunarz 20, Nero 25), plus the mid-map 505 s checkpoint (LAT 73 / VAN 61).
See `eval_results.json` for the measured agreement.

## Honesty

The counter only counts increments it **observes** from the start of the window — it
never invents kills that happened before it started watching, and unreadable cells are
skipped, not guessed. A read is accepted only when it does not decrease, and an
implausible jump must be confirmed by a second frame, so a one-frame OCR glitch cannot
add a kill. Reading uses Tesseract (reliable on this clean panel font, unlike the
stylised scorebar). The panel counter is the kill **spine**; the killfeed detector is
the corroboration/weapon layer, and `eval_panel_counter.py` uses the counter as ground
truth to estimate the killfeed's precision/recall.
