#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory so we always sync to the local recog folder containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_OUT_DIR="$SCRIPT_DIR/runs"
REMOTE_HOST="rangpur"
REMOTE_DIR="~/recognition/runs"

mkdir -p "$LOCAL_OUT_DIR"
rsync -avz --progress \
  --delete \
  --exclude '.DS_Store' \
  "$REMOTE_HOST:$REMOTE_DIR/" \
  "$LOCAL_OUT_DIR/"