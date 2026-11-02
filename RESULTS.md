# Results record

Original-run results below come from user-provided checkpoints, logs, and submission files.
The Kaggle score is user-reported; hidden labels were not accessed. The development score
and all 2,000 final submission predictions were independently reproduced on CPU.

| Item | Actual result |
|---|---|
| Final architecture | Pre-activation Wide ResNet, depth 28, widths 56/112/224 |
| Trainable parameters | 4,480,474 |
| Allowed labeled images | 50,000 |
| Development split | 45,000 / 5,000, stratified, split seed 2026 |
| Main best validation accuracy | 97.36% (4,868 / 5,000), epoch 365 |
| Baseline best validation accuracy | 96.98%, from the selection report |
| Selected recipe / raw or EMA / TTA | Main / EMA / horizontal flip |
| Final fit epochs | 400, all 50,000 allowed training images |
| Training GPU and dependency versions | Pending; copy environment and pip freeze files |
| Final checkpoint location and SHA-256 | Private Drive: runs/final/final.pt; 8d463ace4e242a165250cf32871b7c360aa372420c881298ec7594a068bdef92 |
| Submission SHA-256 | e0002d9308d4dbda8c5ba6b6a6b2ed85746bc78a4d32fb5b49f9570fcb5fba8f |
| Kaggle accuracy (user-reported) | 89.7% |
| Code Git commit / repository URL | https://github.com/mj3664-web/cifar-resnet-colab ; record the training commit |

Attach the selected JSON config, selection report, final metadata, environment record,
and checkpoint link. Run `src.predict` with the preserved checkpoint and original archive
to reproduce the submitted predictions. Re-training can have numerical variation across devices.

## Wider / AugMix experiments

| Experiment | Parameters | Development accuracy | Kaggle accuracy |
|---|---:|---|---|
| wide_main | 4,972,345 | Pending Colab training | Not submitted |
| wide_augmix | 4,972,345 | Pending Colab training | Not submitted |

These recipes have software-test results only. The original validation-to-Kaggle gap is
not explained by a CSV mismatch or corrupt checkpoint; distribution differences remain
a hypothesis, not an established identification of the hidden evaluation dataset.
