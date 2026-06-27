# Weapon Icon Dataset

Phase 5 weapon-recognition scaffold generated from Stage B killfeed segments.

- Source segments: `data/killfeed_dataset/segments.jsonl`
- Icon crops: `icons/`
- Annotation file: `annotations.jsonl`
- Current icon crops: `120`
- Current labelled weapon icons: `0`
- Current real weapon accuracy: no claim

Build and evaluate:

```bash
make weapon-dataset
make weapon-eval
```

The dataset intentionally starts with empty labels:

```json
{
  "valid_weapon": null,
  "weapon": null,
  "weapon_family": null,
  "unclear": null
}
```

Only human-reviewed rows with `valid_weapon=true`, a non-null `weapon`, and
`label_source != "unlabeled"` are used for training/evaluation. Unclear crops
should be marked `unclear=true` or `weapon="unknown"` and excluded from accuracy.

`WeaponRecognizer` is independent from player-name OCR. It must classify only the
weapon icon crop, and it returns `weapon=null` when labels are missing or
confidence is below threshold.
