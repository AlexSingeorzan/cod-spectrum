# Kill-Type Icon Dataset

Phase 5 coarse kill-type dataset generated from Stage B killfeed icon segments.

- Source segments: `data/killfeed_dataset/segments.jsonl`
- Icon crops: `icons/`
- Annotation file: `annotations.jsonl`
- Current icon crops: `120`
- Current labelled kill-type icons: `0`
- Current real kill-type accuracy: no claim

Build and evaluate:

```bash
make kill-type-dataset
make kill-type-eval
```

The dataset intentionally starts with empty labels:

```json
{
  "valid_kill_type": null,
  "kill_type": null,
  "exact_weapon": null,
  "unclear": null
}
```

Only human-reviewed rows with `valid_kill_type=true`, a non-null `kill_type`, and `label_source != "unlabeled"` are used for training/evaluation. Use `unknown` only when a reviewer can see a valid kill cause but cannot assign one of the named classes; use `unclear=true` for ambiguous crops. `exact_weapon` is optional future metadata; downstream analytics consume `kill_type`.
