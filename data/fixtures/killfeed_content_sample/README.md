# Killfeed Content Sample

Synthetic labelled fixture for `KillfeedContentReader`.

It demonstrates the event contract only:

- `KillEvent`
- `DeathEvent`
- `WeaponEvent`
- `TradeEvent`

Rebuild:

```bash
make killfeed-content-sample
```

The fixture is not real broadcast accuracy. The real LAT/VAN killfeed scaffold at
`data/killfeed_dataset/` still has `0` content-labelled rows, so the content reader
reports no real accuracy claim until those labels exist.
