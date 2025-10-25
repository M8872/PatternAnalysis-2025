#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory so we always sync the local recog folder containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$SCRIPT_DIR"
REMOTE_HOST="rangpur"
REMOTE_DIR="~/recog"

ssh "$REMOTE_HOST" 'mkdir -p ~/recog'
rsync -avz --progress \
  --delete \
  --exclude 'out/*' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude '.DS_Store' \
  "$LOCAL_DIR/" \
  "$REMOTE_HOST:$REMOTE_DIR/"