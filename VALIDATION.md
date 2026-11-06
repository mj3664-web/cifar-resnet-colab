# Local verification

Verified on Windows, Python 3.9.7, torch 2.8.0+cpu, torchvision 0.23.0+cpu.

`python -m unittest discover -s tests -v`: **7 tests passed**.

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
These local checks do not re-run competitive GPU training. No synthetic-test accuracy
is a competition result. Model weights and competition data are stored separately.
See [RESULTS.md](RESULTS.md) for the subsequent original-run checkpoint audit and
user-reported Kaggle score.
