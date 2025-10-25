#!/usr/bin/env bash
set -euo pipefail

# Submit N chained sbatch jobs on a100-test that resume training.
# Usage: ./recognition/submit_chain.sh <num_jobs> [epochs]
# - num_jobs: how many sequential jobs to submit (each 20m max)
# - epochs: total target epochs to aim for (default 20)

NUM_JOBS=${1:-3}
TARGET_EPOCHS=${2:-20}

if ! [[ "$NUM_JOBS" =~ ^[0-9]+$ ]]; then
  echo "First arg num_jobs must be an integer" >&2
  exit 1
fi

mkdir -p runs/logs runs/checkpoints runs/metrics

JOBID=""
for i in $(seq 1 "$NUM_JOBS"); do
  if [[ -z "$JOBID" ]]; then
    # First job
    JOBID=$(sbatch --export=ALL,EPOCHS=$TARGET_EPOCHS recognition/run.sh | awk '{print $4}')
  else
    # Chain dependent jobs; start after previous finishes
    JOBID=$(sbatch --dependency=afterany:$JOBID --export=ALL,EPOCHS=$TARGET_EPOCHS recognition/run.sh | awk '{print $4}')
  fi
  echo "Submitted job $i/$NUM_JOBS with ID $JOBID"
done

echo "All jobs submitted. Use: squeue -u $USER"


