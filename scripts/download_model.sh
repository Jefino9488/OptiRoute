#!/usr/bin/env bash
# Download Qwen2.5-3B-Instruct GGUF for local development.
#
# The same download happens during `docker build` via Stage 2.
# Use this script to get the model locally before running agent.py.
#
# Usage:
#   bash scripts/download_model.sh
#
# Override defaults:
#   MODEL_DIR=my/dir MODEL_REPO=bartowski/Qwen2.5-3B-Instruct-GGUF \
#   MODEL_FILE=Qwen2.5-3B-Instruct-Q4_K_M.gguf bash scripts/download_model.sh

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models}"
MODEL_REPO="${MODEL_REPO:-bartowski/Qwen2.5-3B-Instruct-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen2.5-3B-Instruct-Q4_K_M.gguf}"
TARGET="${MODEL_DIR}/${MODEL_FILE}"

mkdir -p "$MODEL_DIR"

if [ -f "$TARGET" ]; then
    echo "✓ Model already exists: $TARGET"
    exit 0
fi

echo "Downloading ${MODEL_FILE} (~1.9 GB) from ${MODEL_REPO}..."
echo "This may take several minutes depending on your connection."

python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='${MODEL_REPO}',
    filename='${MODEL_FILE}',
    local_dir='${MODEL_DIR}',
    local_dir_use_symlinks=False,
)
print(f'✓ Downloaded: {path}')
"
