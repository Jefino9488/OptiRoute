# OptiRoute

**Adaptive Capability-Based Hybrid AI Routing Framework**

> Minimizes inference cost by automatically selecting the cheapest execution pipeline — deterministic tools, a local LLM, or a Fireworks API model — capable of achieving the required accuracy for each prompt.

## Core Concept

```
Prompt → Normalise → Extract Features → Build Task Vector
    → Decision Engine:
        1. Deterministic tools     ($0, instant)
        2. Local LLM router        ($0 Fireworks tokens) ← primary path
        3. Cheapest Fireworks model meeting accuracy threshold
    → Execute → Validate Confidence → Escalate if needed
    → Return Response + Metrics
```

OptiRoute doesn't blindly send everything to the most expensive model. Instead, it:

1. **Routes** using a local Qwen2.5-3B model (max_tokens=15, temp=0.0) — fast, $0 cost
2. **Executes** simple tasks (factual QA, sentiment, NER, summarisation) locally at $0
3. **Delegates** code tasks to `kimi-k2p7-code` and complex reasoning to `minimax-m3`
4. **Validates** output quality and escalates only if local confidence is too low
5. **Caches** results to avoid redundant API calls

## Three-Stage Execution Policy

| Priority | Executor | Cost | Latency | When Used |
|---|---|---|---|---|
| 1️⃣ | Deterministic Tools | $0 | Instant | Pure arithmetic, JSON parsing |
| 2️⃣ | Local Model (Qwen2.5-3B) | **$0 Fireworks tokens** | ~2–5s | Factual QA, sentiment, NER, summarisation |
| 3️⃣ | Fireworks API | $$$ | ~1–3s | Code debug/gen, complex math, logical reasoning |

## Active Models

### Fireworks API

| Model | Architecture | Best For | Cost |
|---|---|---|---|
| `kimi-k2p7-code` | 1T MoE (32B active) | Code debugging & generation | 💲💲💲 |
| `minimax-m3` | 428B MoE | Complex math, logical reasoning, frontier quality | 💲💲💲💲 |

> **Note**: Gemma models are currently non-functional on Fireworks and have been removed.

### Local Model

| Model | Format | GGUF Size | RAM | Cost |
|---|---|---|---|---|
| `Qwen2.5-3B-Instruct-Q4_K_M` (default) | GGUF Q4_K_M | ~1.9 GB | ~2.3 GB | 🆓 $0 always |

---

## Requirements

### System

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.12+ | 3.12+ |
| RAM | 3 GB (without local model) | 4 GB (with local model) |
| Disk | 500 MB | 3 GB (with GGUF weights) |
| CPU | 2 vCPU | 2+ vCPU |

### Python packages

Managed by [uv](https://docs.astral.sh/uv/). Key dependencies:

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | HTTP API server (local dev) |
| `openai` | Fireworks API client (OpenAI-compatible) |
| `llama-cpp-python` | Local GGUF inference (no Ollama needed) |
| `pydantic-settings` | Environment variable config |
| `structlog` | Structured JSON logging |

Install everything:

```bash
uv sync
```

> `llama-cpp-python` is installed from a pre-built **CPU wheel** — no cmake or gcc required.

### External services

| Service | Required | Notes |
|---|---|---|
| Fireworks AI API key | ✅ Yes | Get one at [fireworks.ai](https://fireworks.ai) |
| Ollama / model runtime | ❌ No | Weights run via `llama-cpp-python` directly |
| Redis / vector store | ❌ No | In-memory cache only |
| Internet at runtime | ❌ No | Docker image is fully self-contained |

---

## Installation

```bash
# 1. Clone
git clone https://github.com/Jefino9488/OptiRoute.git
cd OptiRoute

# 2. Install Python dependencies (creates .venv automatically)
uv sync

# 3. Copy and fill in environment variables
cp .env.example .env          # or create .env manually (see below)
```

Minimal `.env`:

```dotenv
FIREWORKS_API_KEY=your_fireworks_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
```

---

## Local Model Setup

The local Qwen2.5-3B model handles **~50–60% of tasks at $0 cost**. You need to download it once before running.

### Option A — Download script (recommended)

```bash
bash scripts/download_model.sh
```

Downloads `Qwen2.5-3B-Instruct-Q4_K_M.gguf` (~1.9 GB) from HuggingFace into `models/`.

**Custom model or directory:**

```bash
# Different target directory
MODEL_DIR=/data/models bash scripts/download_model.sh

# Different model file (e.g. a 1.5B variant for less RAM)
MODEL_REPO=bartowski/Qwen2.5-1.5B-Instruct-GGUF \
MODEL_FILE=Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
bash scripts/download_model.sh
```

### Option B — Manual download from HuggingFace Hub

```bash
pip install huggingface-hub   # one-time, for download only
mkdir -p models

python3 - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="bartowski/Qwen2.5-3B-Instruct-GGUF",
    filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    local_dir="models",
    local_dir_use_symlinks=False,
)
print("Done.")
EOF
```

### Option C — `huggingface-cli`

```bash
pip install huggingface-hub
huggingface-cli download bartowski/Qwen2.5-3B-Instruct-GGUF \
    Qwen2.5-3B-Instruct-Q4_K_M.gguf \
    --local-dir models
```

### Verify the download

```bash
ls -lh models/
# models/Qwen2.5-3B-Instruct-Q4_K_M.gguf   1.9G
```

### Disable local model (API-only mode)

If you want to skip the local model and route everything via Fireworks:

```bash
LOCAL_MODEL_ENABLED=false uv run python agent.py
```

---

## Usage

### Mode 1 — Batch Evaluation (Hackathon / Offline)

This is the primary evaluation mode. Reads a list of tasks and writes answers to a file.

**Input:** `/input/tasks.json`
```json
[
  {"task_id": "t1", "prompt": "What is the capital of Australia?"},
  {"task_id": "t2", "prompt": "Write a Python function that returns the second-largest number in a list."},
  {"task_id": "t3", "prompt": "Classify the sentiment: 'Battery life is great but the screen scratches easily.'"}
]
```

**Run:**
```bash
INPUT_PATH=./input/tasks.json \
OUTPUT_PATH=./output/results.json \
FIREWORKS_API_KEY=your_key \
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code \
uv run python agent.py
```

**Output:** `./output/results.json`
```json
[
  {"task_id": "t1", "answer": "Canberra, near Lake Burley Griffin."},
  {"task_id": "t2", "answer": "def second_largest(nums):\n    unique = sorted(set(nums), reverse=True)\n    return unique[1] if len(unique) > 1 else None"},
  {"task_id": "t3", "answer": "Mixed sentiment: positive about battery life, negative about screen durability."}
]
```

**Expected routing:**
| Task | Routed to | Cost |
|---|---|---|
| t1 — capital of Australia | `local:qwen-2.5-3b` | $0 |
| t2 — Python function (code gen) | `kimi-k2p7-code` | $$$ |
| t3 — sentiment classification | `local:qwen-2.5-3b` | $0 |

---

### Mode 2 — Local HTTP API (Development)

Run the FastAPI server for interactive testing and API exploration.

```bash
# With local model (default)
FIREWORKS_API_KEY=your_key uv run uvicorn main:app --reload --port 8000

# API-only mode (no GGUF required)
FIREWORKS_API_KEY=your_key LOCAL_MODEL_ENABLED=false \
uv run uvicorn main:app --reload --port 8000
```

**Route a prompt:**
```bash
curl -X POST http://localhost:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "required_accuracy": 0.75}'
```

**Response:**
```json
{
  "response": "The capital of France is Paris.",
  "model_used": "local:qwen-2.5-3b",
  "cost": 0.0,
  "latency_ms": 1842.3,
  "confidence": 0.95,
  "cache_hit": false,
  "escalated": false,
  "routing_explanation": "Local LLM router → local model ($0 Fireworks tokens)"
}
```

**Force a specific model:**
```bash
curl -X POST http://localhost:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Fix this bug: def add(a,b): return a-b", "force_model": "kimi-k2p7-code"}'
```

#### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/route` | Route a prompt to the optimal model |
| `GET` | `/v1/models` | List all models with capabilities |
| `GET` | `/v1/metrics` | View routing metrics and cost summary |
| `GET` | `/v1/cache/stats` | Cache hit/miss statistics |
| `GET` | `/health` | Health check |

---

### Mode 3 — Docker (Production / Evaluation Container)

The Docker image bundles everything: Python runtime, `llama-cpp-python`, and the GGUF model weights. **No internet access required at runtime.**

**Build:**
```bash
# Build for linux/amd64 (required for evaluation environment)
docker buildx build --platform linux/amd64 -t optiroute:latest .

# The multi-stage build automatically downloads the GGUF (~1.9 GB) during build
# Stage 1: install Python deps (llama-cpp-python compiled)
# Stage 2: download Qwen2.5-3B-Instruct-Q4_K_M.gguf from HuggingFace
# Stage 3: assemble lean runtime image — CMD runs /app/.venv/bin/python agent.py
```

**Run (simulating the evaluation harness):**
```bash
mkdir -p /tmp/input /tmp/output

cat > /tmp/input/tasks.json << 'EOF'
[
  {"task_id": "t1", "prompt": "What is the capital of Australia?"},
  {"task_id": "t2", "prompt": "Write a binary search function in Python."},
  {"task_id": "t3", "prompt": "Classify the sentiment: 'Great product, terrible support.'"}
]
EOF

docker run --rm \
  --memory=4g --cpus=2 \
  -v /tmp/input:/input:ro \
  -v /tmp/output:/output \
  -e FIREWORKS_API_KEY=your_key \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code \
  optiroute:latest

cat /tmp/output/results.json
```

---

## Running Tests

```bash
# Full test suite (no GGUF or API key needed)
FIREWORKS_API_KEY=dummy uv run pytest tests/ -v

# New tests only
FIREWORKS_API_KEY=dummy uv run pytest tests/test_local_executor.py tests/test_agent.py -v
```

Expected output: `154 passed, 0 failed`

---

## Architecture

```
app/
├── config.py             ← Settings (env vars, ALLOWED_MODELS parsing)
├── api/                  ← FastAPI routes and Pydantic schemas
├── features/             ← Normalizer, extractor, vectorizer
├── router/               ← Capability matrix, decision engine, pipeline
├── executors/
│   ├── local.py          ← Qwen2.5-3B via llama-cpp-python ($0 cost)
│   ├── fireworks.py      ← Fireworks API via OpenAI SDK
│   └── tools.py          ← Deterministic: calculator, JSON parser
├── confidence/           ← Output quality validation
├── cache/                ← In-memory two-tier cache
├── metrics/              ← Cost/latency tracking
└── benchmark/            ← Offline benchmarking and matrix generation

agent.py                  ← Batch evaluation entrypoint (reads /input, writes /output)
main.py                   ← FastAPI app (local dev HTTP server)
models/
└── *.gguf                ← Local model weights (gitignored; baked into Docker image)
data/
├── capability_matrix.json   ← Model capability scores (drives routing decisions)
└── benchmarks/              ← Benchmark datasets per category
scripts/
└── download_model.sh        ← Download GGUF weights from HuggingFace
docs/
├── ARCHITECTURE.md          ← Component diagrams and data flow
├── ROUTING_ALGORITHM.md     ← Decision engine algorithm details
├── CAPABILITY_MATRIX.md     ← Matrix schema and specification
├── IMPLEMENTATION_PLAN.md   ← Full implementation plan
└── PROGRESS.md              ← Implementation progress tracker
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `FIREWORKS_API_KEY` | ✅ | — | Fireworks AI API key |
| `FIREWORKS_BASE_URL` | ❌ | `https://api.fireworks.ai/inference/v1` | API base URL — must use harness-injected value during evaluation |
| `ALLOWED_MODELS` | ❌ (harness injects) | `minimax-m3,kimi-k2p7-code` | Comma-separated full model paths from harness |
| `DEFAULT_ACCURACY_THRESHOLD` | ❌ | `0.75` | Min accuracy for model selection |
| `MAX_ESCALATION_DEPTH` | ❌ | `2` | Max escalation retries |
| `CONFIDENCE_THRESHOLD` | ❌ | `0.7` | Min confidence before escalation |
| `LOG_LEVEL` | ❌ | `INFO` | Logging verbosity |
| `LOCAL_MODEL_ENABLED` | ❌ | `true` | Enable/disable local model |
| `LOCAL_MODEL_PATH` | ❌ | `models/Qwen2.5-3B-Instruct-Q4_K_M.gguf` | Path to GGUF weights |
| `LOCAL_MODEL_NAME` | ❌ | `local:qwen-2.5-3b` | Identifier in capability matrix |
| `LOCAL_MODEL_CONTEXT_LENGTH` | ❌ | `4096` | Max context window (tokens) |
| `LOCAL_MODEL_THREADS` | ❌ | `2` | CPU threads for local inference |
| `LOCAL_ROUTER_ENABLED` | ❌ | `true` | Use local model for routing decisions |
| `INPUT_PATH` | ❌ | `/input/tasks.json` | Batch agent input path |
| `OUTPUT_PATH` | ❌ | `/output/results.json` | Batch agent output path |

---

## Key Design Decisions

1. **Local LLM as router** — Qwen2.5-3B makes routing decisions at max_tokens=15 (fast, $0 cost)
2. **Weighted average scoring** — `Σ(task[d]×cap[d]) / Σ(task[d])` prevents rewarding irrelevant model strengths
3. **ALLOWED_MODELS from env** — Parsed from harness-injected env var at runtime, never hardcoded
4. **Batch CLI entrypoint** — `agent.py` reads/writes files directly; no HTTP server needed for evaluation
5. **Context length pre-check** — Skip local model if prompt exceeds 90% of 4096-token context
6. **Kimi cost penalty** — Output tokens multiplied by 1.3× for mandatory thinking mode
7. **Max 2 escalation levels** — Prevents runaway token cost
8. **In-memory cache only** — No Redis/FAISS; evaluation container runs max 10 minutes
9. **Offline benchmarking only** — Capability scores generated offline, never inside eval container
10. **Multi-stage Docker build** — Build tools excluded from final image; GGUF baked in at build time

---

## License

MIT
