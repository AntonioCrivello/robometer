#!/bin/bash
set -euo pipefail

# Define cluster information
CLUSTER="della-fisac"
REMOTE_USER="ac8755"
REMOTE_BASE="/scratch/gpfs/FISAC/ac8755/robometer"

# Resolve project root relative to this script (scripts/../)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. Sync local repo to cluster
echo "Syncing ${PROJECT_ROOT} to ${CLUSTER}..."
rsync -av \
    "${PROJECT_ROOT}/" \
    ${CLUSTER}:${REMOTE_BASE}/
# echo "Syncing ${PROJECT_ROOT} to ${CLUSTER}..."
# rsync -av \
#     --exclude ".pixi" \
#     --exclude "wandb/" \
#     --exclude ".git" \
#     --exclude "checkpoints/" \
#     "${PROJECT_ROOT}/" \
#     ${CLUSTER}:${REMOTE_BASE}/

