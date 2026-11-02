# Local verification

Verified on Windows, Python 3.9.7, torch 2.8.0+cpu, torchvision 0.23.0+cpu.

`python -m unittest discover -s tests -v`: **11 tests passed**, including the v2 extension.

- Main architecture has exactly 4,480,474 parameters; logits have shape batch x 10;
  all trainable tensors receive finite gradients. An oversized variant is rejected.
- Validation split is reproducible, class-balanced, disjoint, and exhaustive.
- Submission writer preserves string IDs (including leading zeros), follows sample
  ordering even when test rows differ, and rejects invalid labels / duplicate IDs.
- CutMix label weights match the actual clipped patch area.
- EMA updates correctly and its averaging copy has no trainable parameters.
- Cosine schedule reaches its specified warmup and final endpoints.
- End-to-end synthetic-fixture test covers archive preparation, training, interrupted
  training/resumption, validation-only selection, held-out evaluation, full-data fit,
  and final CSV generation. CPU deterministic resumption matches the uninterrupted
  run's raw and EMA state tensors exactly and has identical best-validation metadata.

The original user-supplied archive was separately parsed in full: all 52,000 images
passed RGB/32x32 checks; 50,000 labeled rows, 2,000 test rows, matching sample IDs;
5,000 training images per class. Its SHA-256 is:

```text
397e06e5752bfe0576d1258765bbe1d020cca02db2fc21442cbab6d50257ca51
```

All Python files and all notebook code cells passed syntax checks. The notebook JSON
structure was checked; the full notebook has not been executed in a Colab runtime.
The CUDA/mixed-precision path and long-run accuracy require the user's Colab GPU run.
No synthetic-test accuracy is a competition result. Model weights and competition data
are not packaged in this repository. See `RESULTS.md` for the subsequent original-run audit
and the user-reported Kaggle score.

## V2 extension checks

- Both wider configurations have exactly **4,972,345** trainable parameters. Width 60
  is rejected for exceeding 5 million.
- AugMix returns three finite, correctly shaped views. Mixing labels with this
  consistency-training path is rejected. JSD is zero for identical distributions;
  mismatched predictions produce a positive value and finite gradients to all views.
- The AugMix pipeline passes the same end-to-end interrupted/resumed training test:
  raw and EMA tensors exactly match uninterrupted CPU training. Validation-only selection,
  full-data fitting, export, inference, and CSV ordering also pass for that path.
- Fixed noise diagnostics are deterministic per image. The integration test exercises
  all five diagnostic conditions using held-out synthetic fixtures.
- All v2 notebook Python cells compile. Competitive GPU training and execution of the
  full notebook on Colab remain to be performed by the user.
