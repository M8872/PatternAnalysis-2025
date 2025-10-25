#!/bin/bash
#SBATCH --job-name=alz-train
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=runs/logs/%x-%j.out

# Create log/checkpoint dirs if they do not exist
mkdir -p runs/logs runs/checkpoints runs/metrics

# Quick GPU sanity print
nvidia-smi || true

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

# Dataset root on Rangpur (contains AD_NC/train and AD_NC/test)
DATA_ROOT="/home/groups/comp3710/ADNI/AD_NC"

# Configure total target epochs (can be overridden by env EPOCHS)
EPOCHS="${EPOCHS:-20}"

# Run training (resumable across jobs)
python -u recog/src/train.py \
  --data-root "$DATA_ROOT" \
  --epochs "$EPOCHS" \
  --batch-size 16 \
  --lr 1e-3 \
  --slices-per-volume 8 \
  --resume \
  --checkpoints-dir runs/checkpoints \
  --plots-dir runs/metrics