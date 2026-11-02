#!/bin/bash
#SBATCH --job-name=cifar-augmix-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
: "${CIFAR_PROJECT:?Use hpc/submit_torch.sh to set the persistent project directory}"
: "${CIFAR_PYTHON:?Activate your CUDA PyTorch environment before submission}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "Slurm job: ${SLURM_JOB_ID}; node: ${SLURMD_NODENAME:-unknown}"
echo "Python: ${CIFAR_PYTHON}"
nvidia-smi
"${CIFAR_PYTHON}" -u hpc/run_full_augmix.py --project "${CIFAR_PROJECT}" --workers 6
