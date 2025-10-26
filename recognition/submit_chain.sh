#!/usr/bin/env bash
set -euo pipefail

# Submit N chained sbatch jobs that resume training via checkpoints.
# Usage: ./recognition/submit_chain.sh <num_jobs>

# Always operate from the project directory so outputs land under ~/recognition/runs
cd "$HOME/recognition"

NUM_JOBS=${1:-3}

if ! [[ "$NUM_JOBS" =~ ^[0-9]+$ ]]; then
  echo "First arg num_jobs must be an integer" >&2
  exit 1
fi

mkdir -p runs/logs runs/checkpoints runs/metrics

JOBID=""
for i in $(seq 1 "$NUM_JOBS"); do
  if [[ -z "$JOBID" ]]; then
    # First job
    JOBID=$(sbatch --chdir="$HOME/recognition" "$HOME/recognition/run.sh" | awk '{print $4}')
  else
    # Chain dependent jobs; start after previous finishes (even if it fails)
    JOBID=$(sbatch --dependency=afterany:$JOBID --chdir="$HOME/recognition" "$HOME/recognition/run.sh" | awk '{print $4}')
  fi
  echo "Submitted job $i/$NUM_JOBS with ID $JOBID"
done

echo "All jobs submitted. Use: squeue -u $USER"


