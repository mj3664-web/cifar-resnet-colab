# Full-data AugMix on NYU Torch

This launches **only** `configs/wide_augmix.json`: 400 epochs, 50,000 allowed training
images, 4,972,345 trainable parameters, EMA final weights, and flip-averaged prediction.
It starts from scratch (or resumes this same HPC run); no Colab checkpoint is required.
The user chose this full-data experiment before development selection completed. Do not
claim it was the validation winner or report a held-out score for the full-data model.

## Prerequisites

- An authenticated SSH session as your NYU NetID and an active Slurm allocation.
- The exact original `data.zip`, uploaded to your own persistent project/scratch directory.
- A working CUDA-enabled PyTorch/torchvision Python environment, activated in your shell.
  The launcher does not change or install software into any existing environment.

Torch's documented login is `ssh NETID@login.torch.hpc.nyu.edu`. Use NYU networking / VPN
as applicable and complete password/MFA directly in your own terminal. Do not share credentials.
Run `my_slurm_accounts` after signing in to find the appropriate allocation; it is not your NetID.

Official references:

- https://services.rt.nyu.edu/docs/hpc/connecting_to_hpc/connecting_to_hpc/
- https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/
- https://services.rt.nyu.edu/docs/hpc/tools_and_software/conda_environments/
- https://services.rt.nyu.edu/docs/hpc/storage/data_transfers/

## Prepare your project

Example commands in your authenticated Torch shell (replace the account below with a real
value from `my_slurm_accounts`). This uses a fresh project and preserves your Colab runs:

```bash
mkdir -p "$SCRATCH/cifar_augmix_full"
git clone https://github.com/mj3664-web/cifar-resnet-colab.git "$SCRATCH/cifar_augmix_full/repository"
```

Upload `data.zip` to that project directory using scp/SFTP or the NYU data-transfer service.
For this user's workstation, a PowerShell example is:

```powershell
scp "C:/Users/jmq/Desktop/All Files/DL project/data.zip" mj3664@login.torch.hpc.nyu.edu:/scratch/mj3664/cifar_augmix_full/data.zip
```

Confirm the destination agrees with `echo "$SCRATCH"` on Torch before transferring.
Keep the archive private; do not upload it to GitHub.

Activate your existing CUDA PyTorch environment. Check its imports:

```bash
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda)'
```

CUDA device availability is expected to be false on a CPU-only login node. The allocated
job verifies real GPU access before training. The code requires the versions in the repo's
`requirements.txt`; it was tested locally with torch 2.8 / torchvision 0.23. A CPU-only wheel
will not work. Container environments need a suitable `apptainer exec --nv` wrapper in the
sbatch script rather than just a Python executable path; the supplied launcher targets an
activated non-container environment. Ask for adaptation if you use a container.

## Submit

From your activated environment on Torch:

```bash
cd "$SCRATCH/cifar_augmix_full/repository"
my_slurm_accounts
bash hpc/submit_torch.sh YOUR_SLURM_ACCOUNT "$SCRATCH/cifar_augmix_full"
```

The job requests **1 GPU, 8 CPU cores, 32 GB RAM, and 24 hours**. It does not manually select
a partition. Any compatible GPU is accepted by default; to restrict hardware, before submission:

```bash
export CIFAR_GPU_CONSTRAINT='a100|h100|h200'
```

The actual GPU allocation, queue time, driver compatibility, and runtime are not yet verified.
The 24-hour wall-time request is a limit, not an estimated runtime. Torch may enforce low GPU
utilization policies; check logs and GPU utilization if a job is canceled. Do not launch this
training directly on a login node.

The launcher snapshots the committed code and pins its hash per project. Training writes
checkpoints every epoch to `runs/full_wide_augmix`; a process lock prevents two jobs from
training into that same output directory simultaneously. Submission generation runs after
all 400 epochs complete. Nothing is submitted to Kaggle automatically.

## Monitor and resume

```bash
squeue -u "$USER"
tail -f "$SCRATCH/cifar_augmix_full/logs/cifar-augmix-full-JOBID.out"
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
```

If the job times out or is preempted, first confirm it is no longer running, then rerun the
same submission command from the same Python environment. It resumes `last.pt` and repeats
at most the unfinished epoch. Keep the original config and pinned code. Do not run `sbatch`
again while the existing job is pending/running; the lock prevents simultaneous writes, but
does not remove duplicate queued jobs or cancel resource allocations.

Files to keep:

```text
PROJECT/runs/full_wide_augmix/last.pt           # resumption
PROJECT/runs/full_wide_augmix/final.pt          # completed single-model export
PROJECT/runs/full_wide_augmix/history.jsonl     # full-data training log
PROJECT/submission/submission.csv              # upload to Kaggle
PROJECT/submission/submission.metadata.json    # checkpoint / CSV hashes
PROJECT/code_commit.txt                        # code version used
```

Scratch is not archival storage. Copy final artifacts to permitted durable storage after
completion. No job has been submitted merely by downloading this repository or running
`--validate-only`; a successful `sbatch` response with a job ID is required.
