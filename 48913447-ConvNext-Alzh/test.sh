#!/bin/bash
#SBATCH --job-name=alz-test
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=runs/logs/%x-%j.out

# Ensure we run from the project workspace
cd "$HOME/recognition"

# Create output directories
mkdir -p runs/logs runs/test

# GPU status (optional, safe if GPU not available)
nvidia-smi || true

# Headless backend for matplotlib
export MPLBACKEND=Agg

# Activate virtual environment
VENV_PATH="$HOME/venvs/torch-venv"
if [[ -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
else
  echo "Python venv not found at $VENV_PATH" >&2
  exit 1
fi

# Dataset root (matches training configuration)
DATA_ROOT="/home/groups/comp3710/ADNI/AD_NC"

python -u -m src.predict \
  --data-root "$DATA_ROOT" \
  --batch-size 64 \
  --num-workers 4 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --seed 42 \
  --classifier-dropout 0.3 \
  --checkpoints-dir runs/checkpoints \
  --predictions-dir runs/test
