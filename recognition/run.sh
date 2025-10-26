#!/bin/bash
#SBATCH --job-name=alz-train
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=runs/logs/%x-%j.out

# Ensure we run from the directory containing this script (the recognition folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create log/checkpoint dirs if they do not exist
mkdir -p runs/logs runs/checkpoints runs/metrics

# Quick GPU sanity print
nvidia-smi || true

# Use headless backend for matplotlib on compute nodes
export MPLBACKEND=Agg

# Activate Python venv (pip-based). Override with VENV_PATH if different.
VENV_PATH="${VENV_PATH:-$HOME/venvs/torch-venv}"
if [[ -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
else
  echo "Python venv not found at $VENV_PATH. Set VENV_PATH to your venv path." >&2
  exit 1
fi

# Dataset root (must contain AD/ and CN/ subfolders with .nii/.nii.gz)
: "${DATA_DIR:?Set DATA_DIR to your dataset root (contains AD/ and CN/)}"
DATA_ROOT="$DATA_DIR"

# Configure total target epochs (can be overridden by env EPOCHS)
EPOCHS="${EPOCHS:-20}"

# Run training (resumable across jobs) from repo root using module mode
python -u -m src.train \
  --data-root "$DATA_ROOT" \
  --epochs "$EPOCHS" \
  --batch-size 16 \
  --lr 1e-3 \
  --slices-per-volume 8 \
  --resume \
  --checkpoints-dir runs/checkpoints \
  --plots-dir runs/metrics