#!/usr/bin/env bash
set -euo pipefail

# This script downloads results from two possible locations on the remote host:
# 1) ~/recognition/runs  (primary default)
# 2) ~/runs              (alternate/fallback or legacy location)
#
# It syncs both to the local runs/ directory (relative to this script), overlaying files.
# Heavy comments for student readability.

# Figure out the directory where this script lives, so we copy to that folder's runs/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_OUT_DIR="$SCRIPT_DIR/runs"
REMOTE_HOST="rangpur"
REMOTE_DIRS=("~/recognition/runs" "~/runs" )

# Make sure the local runs directory exists
mkdir -p "$LOCAL_OUT_DIR"

# Loop through remote directories to sync each one
for REMOTE_DIR in "${REMOTE_DIRS[@]}"; do
  echo "Syncing from $REMOTE_HOST:$REMOTE_DIR to $LOCAL_OUT_DIR ..."
  # The trailing slash on source means "copy contents of directory"
  rsync -avz --progress \
    --delete \
    --exclude '.DS_Store' \
    "$REMOTE_HOST:$REMOTE_DIR/" \
    "$LOCAL_OUT_DIR/" || \
    echo "Warning: Could not sync from $REMOTE_HOST:$REMOTE_DIR (may not exist or have permissions)"
done

echo "Sync complete."