# LAT/VAN Scorebar OCR Dataset

Phase 3 scorebar OCR fixture for `CdlScorebarOcrEngine`.

- Source: official CDL LAT vs Vancouver Hardpoint VOD labels already recorded in `backend.app.services.real_match.VERIFIED`
- Label source: `human_verified`
- Labeled by: `alex`
- Samples: 21 scorebar-present crops
- Digit glyphs: 107 normalized glyph images
- Excluded: final 685s label because the available nearby crop is a red post-map frame with no scorebar

Rebuild:

```bash
make scorebar-ocr-dataset
```

Evaluate:

```bash
make scorebar-ocr-eval
```

Current metrics:

- Operational gallery: `21/21` exact score matches
- Leave-one-crop-out: `10/21` exact score matches (`0.4762`)
- Leave-one-crop-out temporal decoding: `11/21` exact score matches (`0.5238`)
- Digit accuracy: `0.8131`

The gallery result is only a wiring/regression check. The leave-one-out score is
the honest confidence ceiling for runtime OCR. This dataset is one real map and
is not enough for production score OCR.
