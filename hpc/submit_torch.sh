#!/bin/bash
# Run after authenticating to Torch and activating a CUDA-enabled PyTorch environment.
# Usage: bash hpc/submit_torch.sh YOUR_SLURM_ACCOUNT /scratch/NETID/cifar_augmix_full
set -euo pipefail
account="${1:?Usage: bash hpc/submit_torch.sh SLURM_ACCOUNT PROJECT_DIRECTORY}"
project="${2:?Provide a persistent project directory containing data.zip}"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo"
[[ -f "$project/data.zip" ]] || { echo "Missing $project/data.zip" >&2; exit 1; }
mkdir -p -- "$project/logs" "$project/code"
project="$(cd -- "$project" && pwd)"
export CIFAR_PROJECT="$project"
export CIFAR_PYTHON="$(command -v python)"
"$CIFAR_PYTHON" -c 'import torch, torchvision; print("torch", torch.__version__, "torchvision", torchvision.__version__); assert torch.version.cuda is not None, "Install a CUDA-enabled PyTorch build"'

# Snapshot committed code, including this launcher; never mutate a running job's source.
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked code has local changes. Commit them before creating a training snapshot." >&2
    exit 1
fi
commit="$(git rev-parse HEAD)"
pin="$project/code_commit.txt"
if [[ -f "$pin" ]]; then
    commit="$(cat -- "$pin")"
fi
snapshot="$project/code/$commit"
if [[ ! -f "$snapshot/.snapshot_complete" ]]; then
    mkdir -p -- "$snapshot"
    git archive "$commit" | tar -x -C "$snapshot"
    printf '%s\n' "$commit" > "$snapshot/code_commit.txt"
    touch "$snapshot/.snapshot_complete"
fi
[[ -f "$pin" ]] || printf '%s\n' "$commit" > "$pin"
"$CIFAR_PYTHON" "$snapshot/hpc/run_full_augmix.py" --project "$project" --validate-only

# GPU type is optional: e.g. export CIFAR_GPU_CONSTRAINT='a100|h100|h200'.
# Torch documentation says not to specify a partition manually.
extra=()
if [[ -n "${CIFAR_GPU_CONSTRAINT:-}" ]]; then
    extra+=(--constraint="$CIFAR_GPU_CONSTRAINT")
fi
job="$(sbatch --parsable --account="$account" --export=ALL --chdir="$snapshot" \
    --output="$project/logs/%x-%j.out" --error="$project/logs/%x-%j.err" \
    "${extra[@]}" "$snapshot/hpc/torch_full_augmix.sbatch")"
printf '%s\n' "$job" > "$project/last_job_id.txt"
echo "Submitted job $job"
echo "Check: squeue -j ${job%%;*}"
echo "Log: $project/logs/cifar-augmix-full-${job%%;*}.out"
echo "Checkpoint: $project/runs/full_wide_augmix/last.pt"
