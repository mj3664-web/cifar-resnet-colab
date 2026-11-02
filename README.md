# CIFAR competition: residual CNN under 5M parameters

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mj3664-web/cifar-resnet-colab/blob/main/Colab_Training.ipynb)

A complete Google Colab training and submission pipeline for the supplied `data.zip`.
The model is a pre-activation Wide ResNet with **4,480,474 trainable parameters**.
No pretrained weights, external images, downloaded CIFAR copies, or test labels are used.

**Status:** code and local CPU pipeline checks are complete. The competitive GPU runs
must be executed in Colab. No validation accuracy or Kaggle score is claimed yet.
The recipes are strong starting points, not a guarantee of the highest possible score.

## Start in Colab Pro

1. Create `MyDrive/cifar_resnet_project` in Google Drive.
2. Upload the supplied `data.zip` there. The notebook clones this public GitHub repository automatically when no code ZIP is present. Alternatively, upload the original provided `cifar-resnet-colab.zip` too.
   Keep both ZIP files intact. Do not upload the 52,000 images individually.
3. Open `Colab_Training.ipynb` using Colab's **File → Upload notebook**.
4. Select a GPU runtime, preferably A100 or L4 if offered. Available hardware varies.
5. Run the notebook in order. It mounts Drive, unpacks the code to the local runtime,
   prepares the supplied dataset, tests the code, and trains.
6. Leave `RUN_COMPARISON = True` to compare the 400-epoch main recipe and the
   300-epoch lighter-augmentation recipe. The winning recipe is then trained from
   scratch on all 50,000 permitted images for its full epoch budget.
7. Download `submission.csv` and submit it manually through the official competition page.
   Upload this repository's code to your team's GitHub repository.

The notebook also saves `submission.metadata.json`, configuration, split indices,
validation metrics, checkpoints, and environment details to Drive. It never submits to Kaggle.
Use the notebook's first three epochs to estimate your GPU time; no fixed runtime is promised.
Training all default stages requires 1,000–1,100 total epochs, depending on the winning recipe.

## Dataset contract

The inspected archive has this layout, optionally inside one enclosing directory:

```text
train_labels.csv        # id,filename,label; 50,000 rows, 5,000 per class
test.csv                # id,filename; 2,000 rows
sample_submission.csv   # id,label; required output order
train_images/*.png      # RGB, 32 x 32
test_images/*.png       # RGB, 32 x 32
```

CSV parsing handles the extra blank lines in the supplied training CSV. IDs remain strings.
Preparation validates the schemas, image dimensions, ID uniqueness, and test/sample ID sets.
It reads ZIP entries directly and creates local NumPy caches; it does not extract arbitrary paths.
No original image or dataset file is bundled in this repository or intended for GitHub.

## Model and training design

The model follows [Wide Residual Networks](https://arxiv.org/abs/1605.07146), with
the width adjusted to fit the competition. It uses a 3x3 stem (16 channels), three
stages with four residual blocks each (56, 112, 224 channels), 3x3 convolutions,
1x1 projection skips at transitions, global average pooling, and a 10-class head.
Stage transitions downsample with stride 2. There are no pretrained components.
The parameter guard counts every trainable tensor, including batch normalization and the head.
Both included configurations have exactly 4,480,474 parameters; dropout adds none.

| Setting | Main | Comparison baseline |
|---|---:|---:|
| Epochs | 400 | 300 |
| Batch size | 128 | 128 |
| Optimizer | Nesterov SGD, momentum 0.9 | same |
| Initial / final LR | 0.1 / 0.0001 | same |
| Warmup | 5 epochs | 5 epochs |
| Schedule | per-step cosine | same |
| Weight decay | 0.0005, excluding BN/bias | same |
| Random crop / horizontal flip | padding 4 / p=0.5 | same |
| RandAugment | 2 ops, magnitude 7 | 2 ops, magnitude 5 |
| MixUp / CutMix | alpha 0.2 / 1.0 | disabled |
| Probability of batch mixing | 0.8, equal choice of MixUp/CutMix | 0 |
| Last epochs without mixing | 20 | all |
| Dropout | 0 | 0.1 |
| Label smoothing | 0 | 0 |
| EMA decay | warmed up to 0.999 | same |
| Inference | image + horizontal flip | same |

The main recipe combines [RandAugment](https://arxiv.org/abs/1909.13719),
[MixUp](https://arxiv.org/abs/1710.09412), and [CutMix](https://arxiv.org/abs/1905.04899).
The comparison recipe checks whether less regularization is better for this particular task.
No externally learned AutoAugment policy is used. EMA maintains a running average of one
network's weights and floating-point buffers; exported inference checkpoints contain only
one network. Evaluating flipped views averages class probabilities from that same network.
The saved training state has both raw and EMA copies for resumption, not a two-model ensemble.

## Validation and model selection

The default split is 45,000 training / 5,000 validation images, with 500 held-out
examples per class and fixed split seed 2026. Normalization statistics are computed
only from the 45,000 training images. Validation has no random augmentation.
Every epoch evaluates raw and EMA weights with the configured inference transforms.
`best.pt` stores whichever has the highest validation accuracy, with earlier wins retained on ties.
`best_metrics.json` includes accuracy, correct/total counts, loss, and a 10x10 confusion matrix
(rows = true labels, columns = predicted labels).

`src.select` requires completed runs and matching validation splits. It selects the recipe
with the best validation accuracy and transfers the selected raw/EMA choice into a final config.
The final fit starts with random weights and uses all 50,000 training images, computes fresh
normalization statistics, and runs the selected recipe's full schedule. Its `final.pt` is the
submission model. Do not shorten a cosine schedule to the earlier best checkpoint epoch:
that would change the learning-rate trajectory that was validated.

The held-out score estimates performance of the development model, not the final full-data
model. Full-data training cannot legitimately report accuracy on the old validation images;
`src.evaluate` rejects that request. Only the official Kaggle evaluation can provide the hidden score.
The test set is used only for format checks and final inference, never for training,
normalization, hyperparameter tuning, pseudo-labeling, or checkpoint selection.

If compute permits further experiments, copy a config to a new filename and run a new output
directory. Useful controlled comparisons are magnitude 5 versus 7, mixing probability 0.5
versus 0.8, or a second training seed. Keep the split fixed. Avoid picking hyperparameters
from tiny validation differences after dozens of trials; confirm close results with a second
seed or split. All new model widths/depths are checked against the parameter limit at runtime.

## Local / command-line usage

Run all commands from the repository root with Python 3.10+ recommended.
CPU correctness was also checked with Python 3.9.7, torch 2.8.0+cpu, torchvision 0.23.0+cpu.
Install a matched PyTorch/torchvision pair suitable for your GPU; Colab uses its existing pair.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.data --zip /path/to/data.zip --out cache
python -m src.train --data cache --config configs/main.json --out runs/main --resume
python -m src.train --data cache --config configs/baseline.json --out runs/baseline --resume
python -m src.select --runs runs/main runs/baseline --out runs/selected.json
python -m src.evaluate --data cache --checkpoint runs/main/best.pt
python -m src.train --data cache --config runs/selected.json --out runs/final --full-data --resume
python -m src.predict --data cache --checkpoint runs/final/final.pt --out submission.csv
```

Use `--workers 0` for constrained Windows/CPU environments. CUDA training uses channels-last
layout and automatic mixed precision: BF16 when supported, otherwise FP16 with gradient scaling.
Inference and validation use float32. An A100/L4 should fit batch 128, but available memory is
not guaranteed. If memory is insufficient, begin a new run with batch 64 and LR 0.05, and
compare on validation; do not silently change a resumed run's configuration.

## Resuming, reproducibility, and artifacts

Each completed epoch writes `last.pt` via a temporary file followed by replacement.
It contains the optimizer, scaler, model, EMA, RNG states, configuration, archive SHA-256,
and progress. `--resume` starts fresh if no checkpoint exists, or continues it if present.
A disconnect loses at most the current unfinished epoch. Drive storage should have several
hundred MB available for multiple runs; local cache storage uses approximately 160 MB.
Keep the original ZIP, code archive, run outputs, and exported final checkpoint together.

To pause after a specific completed epoch while preserving the 400-epoch schedule:

```bash
python -m src.train --data cache --config configs/main.json --out runs/main --resume --stop-after 25
```

Resume later without `--stop-after`. It is an absolute completed-epoch limit, not an extra-epochs count.
Do not edit configs midway; the resume checks reject changed configs, dataset hashes, and data modes.
Re-running the notebook after a disconnect rebuilds local caches and resumes Drive checkpoints.
Leave the project name, recipe, and deterministic setting unchanged.

`--deterministic` enables deterministic PyTorch algorithms and disables cuDNN benchmarking.
Use it from the beginning if strict replay on the same hardware/software matters; set
`DETERMINISTIC = True` in the notebook. The default prioritizes GPU throughput. Identical
seeds do not guarantee bit-identical outcomes across CUDA, hardware, or library versions.
Epoch-specific loader seeds and saved RNG state support faithful resumption. Every run records
its dependency versions, GPU, configuration, split indices, and `pip freeze` output.
Only load resumption checkpoints you created: restoring RNG objects requires `weights_only=False`.
Exported inference checkpoints load with `weights_only=True`.

## Repository and competition hand-in

```text
Colab_Training.ipynb   end-to-end Colab workflow
configs/              complete training configurations
src/model.py          residual model and hard parameter-limit guard
src/data.py           archive validation, caching, split, preprocessing
src/engine.py         mixing, EMA, schedule, validation, inference transforms
src/train.py          training, checkpointing, resumption, exports
src/evaluate.py       held-out evaluation of development checkpoints
src/select.py         validation-only recipe selection
src/predict.py        ID-aligned, checked submission generation
tests/                model, data split, mixing, and submission tests
REFERENCES.md         method and library citations
RESULTS.md            record actual results after training
```

Create your team's GitHub repository and upload these source files. Do not commit the data,
cache, or private competition materials. Large checkpoint files should be stored separately
(for example in a GitHub release or permitted Drive location), with a link and SHA-256 in
`RESULTS.md`. Before handing in, fill in `RESULTS.md` using the generated run metadata and
official score; include the final config and environment files in your reproducibility record.
Repository: https://github.com/mj3664-web/cifar-resnet-colab

The final `submission.csv` has exactly `id,label` columns, integer labels 0–9, and exactly
one prediction per sample ID in the sample's original order. Submit it through the official
Kaggle page, following the competition's submission limits. Do not infer class labels from
filenames, test ordering, external CIFAR copies, or hidden evaluation data.
