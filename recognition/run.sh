#!/bin/bash
#SBATCH --job-name=alz-train
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=runs/logs/%x-%j.out


# Always run from the project directory so outputs land in ~/recognition/runs
cd "$HOME/recognition"

# Create log/checkpoint dirs if they do not exist
mkdir -p runs/logs runs/checkpoints runs/metrics

# Quick GPU sanity print
nvidia-smi || true

# Use headless backend for matplotlib on compute nodes
export MPLBACKEND=Agg

# Activate Python venv (fixed location)
VENV_PATH="$HOME/venvs/torch-venv"
if [[ -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
else
  echo "Python venv not found at $VENV_PATH" >&2
  exit 1
fi

# Dataset root (JPEG ImageFolder layout under AD_NC/{train,test}/{AD,CN})
DATA_ROOT="/home/groups/comp3710/ADNI/AD_NC"

# Configure total target epochs (fixed)
EPOCHS="25"

# Run training (resumable across jobs) using local package path
python -u -m src.train \
  --data-root "$DATA_ROOT" \
  --epochs "$EPOCHS" \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --classifier-dropout 0.3 \
  --resume \
  --checkpoints-dir runs/checkpoints \
  --plots-dir runs/metrics
