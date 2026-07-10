# OptiRoute — Evaluation Container
#
# Multi-stage build:
#   Stage 1 (llama-downloader) — fetch pre-built llama.cpp server binary (b9940, x64)
#   Stage 2 (model-setup)      — copy local model weights or download from HF Hub
#   Stage 3 (final)            — assemble lean runtime image with Python deps
#
# The final image is self-contained:
#   - llama-server runs as a background HTTP service (localhost:8080)
#   - Python agent calls it via OpenAI-compatible /v1/chat/completions
#   - No llama-cpp-python, no build tools, no Ollama
#
# Build for linux/amd64 (required by judging VM):
#   docker buildx build --platform linux/amd64 -t optiroute:latest .
#
# Run locally (simulating harness):
#   docker run --rm \
#     -v /tmp/input:/input:ro -v /tmp/output:/output \
#     -e FIREWORKS_API_KEY=your_key \
#     -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
#     -e ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code \
#     optiroute:latest

# ── Stage 1: Download llama.cpp pre-built server binary ───────────────────────
FROM python:3.12-slim AS llama-downloader

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tar \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Extract all files flat into /opt/llama so sibling .so plugin loading works.
# llama-server dynamically loads libggml-cpu-*.so from its own directory at
# runtime (ggml_backend_load_all). Splitting binary and libs breaks this.
RUN mkdir -p /opt/llama && \
    curl -L https://github.com/ggml-org/llama.cpp/releases/download/b9940/llama-b9940-bin-ubuntu-x64.tar.gz | \
    tar -xz -C /opt/llama --strip-components=1

# ── Stage 2: Model Setup ──────────────────────────────────────────────────────
FROM python:3.12-slim AS model-setup

RUN pip install --no-cache-dir "huggingface-hub>=0.23"

RUN mkdir -p /models

# Mount local models/ dir at build time; download from HF Hub if not present.
RUN --mount=type=bind,source=models,target=/local_models \
    if [ -f /local_models/Qwen2.5-3B-Instruct-Q4_K_M.gguf ]; then \
        echo "Found local model, copying..."; \
        cp /local_models/Qwen2.5-3B-Instruct-Q4_K_M.gguf /models/Qwen2.5-3B-Instruct-Q4_K_M.gguf; \
    else \
        echo "Local model not found, downloading from HF Hub..."; \
        python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='bartowski/Qwen2.5-3B-Instruct-GGUF', filename='Qwen2.5-3B-Instruct-Q4_K_M.gguf', local_dir='/models', local_dir_use_symlinks=False)"; \
    fi

# ── Stage 3: Final runtime image ──────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# curl — health check polling in entrypoint.sh
# libgomp1 — GNU OpenMP runtime required by llama-server CPU kernels
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Co-locate ALL llama.cpp files in /opt/llama so backend plugin loader
# (ggml_backend_load_all) finds libggml-cpu-*.so siblings at startup.
COPY --from=llama-downloader /opt/llama/ /opt/llama/

# Expose llama-server on PATH via symlink; register libs with ldconfig.
RUN ln -s /opt/llama/llama-server /usr/local/bin/llama-server && \
    echo "/opt/llama" > /etc/ld.so.conf.d/llama.conf && \
    ldconfig

# Plugin loader also searches LD_LIBRARY_PATH — belt and braces.
ENV LD_LIBRARY_PATH=/opt/llama

# Copy GGUF model weights
COPY --from=model-setup /models /app/models

# Install Python dependencies (no llama-cpp-python — not needed)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# Copy application source
COPY app/ ./app/
COPY data/ ./data/
COPY main.py agent.py ./

# ── Local model env vars ───────────────────────────────────────────────────────
ENV LOCAL_MODEL_ENABLED=true
ENV LOCAL_MODEL_PATH=models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
ENV LOCAL_MODEL_NAME=local:qwen-2.5-3b
ENV LOCAL_MODEL_CONTEXT_LENGTH=8192
ENV LOCAL_MODEL_THREADS=2
ENV LOCAL_ROUTER_ENABLED=false
ENV LOCAL_SERVER_URL=http://localhost:8080/v1

# ── Entrypoint: start llama-server, wait until healthy, then run agent ─────────
RUN printf '#!/bin/sh\n\
echo "[startup] Launching llama-server..."\n\
llama-server \\\n\
  -m models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \\\n\
  --port 8080 --host 0.0.0.0 \\\n\
  --threads 2 -c 8192 \\\n\
  -b 512 --ubatch-size 512 \\\n\
  --cont-batching -np 1 \\\n\
  --no-mmap \\\n\
  --reasoning off \\\n\
  > /app/llama_server.log 2>&1 &\n\
LLAMA_PID=$!\n\
\n\
echo "[startup] Waiting for llama-server to be ready (up to 60s)..."\n\
READY=0\n\
for i in $(seq 1 60); do\n\
    if ! kill -0 $LLAMA_PID > /dev/null 2>&1; then\n\
        echo "[startup] ERROR: llama-server process exited unexpectedly!"\n\
        echo "[startup] --- llama-server logs ---"\n\
        cat /app/llama_server.log\n\
        exit 1\n\
    fi\n\
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then\n\
        echo "[startup] llama-server ready after ${i}x2s"\n\
        READY=1\n\
        break\n\
    fi\n\
    sleep 2\n\
done\n\
\n\
if [ "$READY" = "0" ]; then\n\
    echo "[startup] ERROR: llama-server did not respond within 120s"\n\
    echo "[startup] --- llama-server logs ---"\n\
    cat /app/llama_server.log\n\
    exit 1\n\
fi\n\
\n\
echo "[startup] Starting evaluation agent..."\n\
exec /app/.venv/bin/python agent.py\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

EXPOSE 8000 8080

ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
