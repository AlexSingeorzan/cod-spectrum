# Killfeed Segmentation Sample

Synthetic fixture for Phase 4 Stage B.

It verifies the field-box contract only:

- attacker text crop
- weapon icon crop
- victim text crop
- optional headshot crop

Rebuild:

```bash
make killfeed-segment-sample
```

The sample is not real broadcast segmentation accuracy. Real readiness is
reported by:

```bash
make killfeed-segments
make killfeed-segment-eval
```
