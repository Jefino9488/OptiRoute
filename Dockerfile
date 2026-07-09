# OptiRoute — Evaluation Container
#
# Multi-stage build:
#   Stage 1 (builder)         — install Python deps including llama-cpp-python (ROCm)
#   Stage 2 (model-downloader)— download Qwen2.5-3B-Instruct GGUF (~1.9 GB)
#   Stage 3 (final)           — assemble runtime image (no build tools)
#
# The final image is self-contained:
#   - No internet access needed at runtime
#   - No Ollama or external model runtime required
#   - FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS injected at runtime by harness
#   - Runtime GPU detection: ROCm if available, else CPU-only fallback
#
# Build for linux/amd64 (required by judging VM, even on Apple Silicon):
#   docker buildx build --platform linux/amd64 -t optiroute:latest .
#
# Run locally (simulating harness):
#   docker run --rm --memory=4g --cpus=2 \
#     -v /path/to/input:/input:ro \
#     -v /path/to/output:/output \
#     -e FIREWORKS_API_KEY=your_key \
#     -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
#     -e ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code \
#     optiroute:latest

# ── Stage 1: Install Python dependencies ──────────────────────────────────────
FROM python:3.12-slim AS builder

# Build tools needed to compile llama-cpp-python (or its C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install all deps (llama-cpp-python from ROCm wheel index — falls back to CPU at runtime)
RUN uv sync --frozen --no-cache

# ── Stage 2: Download GGUF model weights ──────────────────────────────────────
FROM python:3.12-slim AS model-downloader

# huggingface-hub for downloading from HF Hub
RUN pip install --no-cache-dir "huggingface-hub>=0.23"

# Download Qwen2.5-3B-Instruct-Q4_K_M.gguf (~1.9 GB)
# 2B-3B 4-bit models fit comfortably in 4 GB RAM (per official hackathon guidance)
RUN mkdir -p /models && python -c "\
from huggingface_hub import hf_hub_download; \
path = hf_hub_download( \
    repo_id='bartowski/Qwen2.5-3B-Instruct-GGUF', \
    filename='Qwen2.5-3B-Instruct-Q4_K_M.gguf', \
    local_dir='/models', \
    local_dir_use_symlinks=False, \
); \
print(f'Downloaded: {path}') \
"

# ── Stage 3: Final runtime image ──────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy installed packages (includes compiled llama-cpp-python — no build tools needed)
COPY --from=builder /app/.venv /app/.venv

# Copy GGUF model weights (~1.9 GB — bundled in image, no internet at runtime)
COPY --from=model-downloader /models /app/models

# Copy application source
COPY pyproject.toml uv.lock ./
COPY app/ ./app/
COPY data/ ./data/
COPY main.py agent.py ./

# Local model configuration
ENV LOCAL_MODEL_ENABLED=true
ENV LOCAL_MODEL_PATH=models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
ENV LOCAL_MODEL_NAME=local:qwen-2.5-3b
ENV LOCAL_MODEL_CONTEXT_LENGTH=4096
ENV LOCAL_MODEL_THREADS=2
ENV LOCAL_ROUTER_ENABLED=true

# NOTE: These are injected by the evaluation harness at runtime.
# DO NOT set them here — the harness values override container defaults.
# FIREWORKS_API_KEY=<injected>
# FIREWORKS_BASE_URL=<injected>   ← ALL Fireworks API calls MUST go through this
# ALLOWED_MODELS=<injected>       ← comma-separated full model paths

# Port for local HTTP development (not used by evaluation harness)
EXPOSE 8000

# Evaluation entrypoint:
#   Reads  /input/tasks.json  → [{task_id, prompt}, ...]
#   Writes /output/results.json → [{task_id, answer}, ...]
#
# Use venv Python directly (faster startup, no uv lock resolution at runtime)
CMD ["/app/.venv/bin/python", "agent.py"]
