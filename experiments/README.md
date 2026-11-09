# Design exploration and ablations

This folder holds the architecture study and ablation experiments behind Section 4 of the report.
The team's final submission is the WRN-28-56 model in the repository root (`src/`, `Colab_Training.ipynb`).

| File | What it is |
|---|---|
| `training_notebook.ipynb` | Google Colab notebook that trained every run below (ResNet A/B/C, augmentation, CutMix, SAM, SWA, multi-view TTA). The outputs saved in it belong to the last run executed (`arch_C_300ep_cutmix_bs128`: 4,781,374 parameters, 300-epoch log, final validation accuracy 96.96%, 97.14% with flip TTA). |
| `results_analysis.ipynb` | CPU-only notebook, already executed. It reads the records below and reproduces every table and figure of Section 4. It also prints the parameter count and MACs of each architecture, and confirms all of them are under 5M. |
| `experiments.csv` | One row per run, written automatically by the training notebook: config, parameter count, final / best validation accuracy, flip-TTA accuracy, robustness score, and time. |
| `logs/<run>/log.csv` | Per-epoch learning rate, train loss / accuracy, validation loss / accuracy, and seconds for the runs plotted in the report. |
| `figures/` | Figures produced by `results_analysis.ipynb` and used in the report. |

Validation numbers use a stratified 45,000 / 5,000 split of the provided training set (500 per class, seed 42).
Kaggle public scores were copied from the leaderboard by hand; they are entered in `results_analysis.ipynb`.

## Map from the report to the records

| Report item | Source |
|---|---|
| Table 3 (architectures A/B/C), Fig. 2 | `experiments.csv` rows `v5a_trivialaug`, `arch_B_deep_narrow`, `arch_C_wide_3stage`; `logs/` for the curves; parameter and MAC counts from `results_analysis.ipynb` §2 |
| Table 4 (ablations) | `experiments.csv` rows `v1`–`v5b` and `arch_C_*` |
| Fig. 3 (training length) | `logs/arch_C_wide_3stage`, `logs/arch_C_300ep` |
| Fig. 4 (validation vs Kaggle) | `experiments.csv` plus the Kaggle scores in `results_analysis.ipynb` |
| Table 5 (TTA) | outputs of the TTA cells of `training_notebook.ipynb` (§12) |
