# OptiRoute — Agent Rules & Project Context

## Project Summary

OptiRoute is an **Adaptive Capability-Based Hybrid AI Routing Framework** for a hackathon (Track 1: Hybrid Token-Efficient Routing Agent). It minimizes inference cost by dynamically selecting the cheapest execution pipeline — deterministic tools, a local LLM, or a Fireworks AI model — capable of meeting the required accuracy for each prompt.

**Core question the system answers for every request:**
> "What is the cheapest way to answer this prompt correctly — a deterministic tool, a local model, or a Fireworks API model?"

## Hackathon Constraints

- **Runtime**: Evaluation container runs for max **10 minutes**
- **Inference**: Fireworks API tokens are scored; local model outputs count toward accuracy at **$0 token cost**
- **Scoring**: `accuracy vs total_token_cost` — cheapest tokens for correct answers wins
- **Prompts**: Hidden, unseen evaluation prompts — no hardcoding allowed
- **Output**: Must produce valid JSON in required evaluation format
- **Hardware**: Evaluation container provides **4 GB RAM**, **2 vCPUs**, **10 GB compressed image limit**
- **Local Models**: Allowed; must be bundled into Docker image (no Ollama pre-installed)

## Allowed Models

### Fireworks API Models

| Model ID | Fireworks API Name | Architecture | Primary Use |
|---|---|---|---|
| `minimax-m3` | `accounts/fireworks/models/minimax-m3` | 428B MoE | Frontier: best quality, moderate cost |
| `kimi-k2p7-code` | `accounts/fireworks/models/kimi-k2p7-code` | 1T MoE (32B active) | Code specialist — **mandatory thinking mode adds ~30% token overhead** |
| `gemma-4-31b-it` | `accounts/fireworks/models/gemma-4-31b-it` | 31B dense | Strong general reasoning |
| `gemma-4-26b-a4b-it` | `accounts/fireworks/models/gemma-4-26b-a4b-it` | 26B sparse | Cheapest — lightweight generalist |
| `gemma-4-31b-it-nvfp4` | `accounts/fireworks/models/gemma-4-31b-it-nvfp4` | 31B quantized NF4 | Budget reasoning — slightly degraded vs full 31B |

### Local Model (Configurable)

| Model ID | Format | Size | Primary Use |
|---|---|---|---|
| `local:qwen-2.5-3b` | GGUF Q4_K_M | ~1.93 GB | Simple QA, retrieval, creative — **$0 Fireworks tokens** |

> The local model is fully configurable via env vars. Swap to Gemma 3 1B (~806 MB) if memory is tight.

## Architecture Overview

```
Prompt → Normalizer → Feature Extractor → Task Vector Builder
    → Decision Engine:
        1. Deterministic tools ($0, instant)
        2. Local LLM ($0 Fireworks tokens)
        3. Cheapest Fireworks model meeting accuracy threshold
    → Executor (Local or Fireworks) → Confidence Validator
    → (Escalation if needed) → Response + Metrics
```

### Key Components

| Component | Location | Purpose |
|---|---|---|
| Config | `app/config.py` | Pydantic Settings — env vars, model registry, local model config |
| Normalizer | `app/features/normalizer.py` | Strip filler, canonical form, hashing |
| Extractor | `app/features/extractor.py` | Regex/keyword feature detection (no ML) |
| Vectorizer | `app/features/vectorizer.py` | Generate Task/Resource/Risk vectors |
| Capability Matrix | `app/router/capability_matrix.py` | Load/query model capabilities from JSON |
| Decision Engine | `app/router/decision_engine.py` | **Core algorithm** — filter → score → cheapest |
| Escalation Policy | `app/router/policy.py` | Retry with next-cheapest if confidence low |
| Pipeline | `app/router/pipeline.py` | Orchestrates full request lifecycle |
| Fireworks Executor | `app/executors/fireworks.py` | OpenAI-compatible API via `openai` SDK |
| Local Executor | `app/executors/local.py` | Local LLM via `llama-cpp-python` — $0 cost |
| Deterministic Tools | `app/executors/tools.py` | Calculator, JSON parser, regex — $0 cost |
| Confidence Validator | `app/confidence/validator.py` | Check output quality, trigger escalation |
| Metrics | `app/metrics/collector.py` | Track cost, latency, cache hits, escalations |
| Cache | `app/cache/manager.py` | In-memory exact + normalized prompt cache |
| Benchmark Runner | `app/benchmark/runner.py` | Run datasets through all models |
| Evaluator | `app/benchmark/evaluator.py` | Score model outputs per category |
| Matrix Generator | `app/benchmark/matrix_generator.py` | Build capability_matrix.json from benchmarks |

### Data Files

| File | Purpose |
|---|---|
| `data/capability_matrix.json` | Model capabilities — **the heart of routing** (includes local model at $0) |
| `data/benchmarks/*.json` | Curated benchmark datasets per category |
| `models/*.gguf` | Local model weights (bundled in Docker image) |

## Coding Standards

### Language & Framework
- **Python 3.12+**
- **FastAPI** for API layer
- **Pydantic v2** for all data models and validation
- **async/await** throughout — all I/O is async
- **uv** as package manager (not pip)

### Conventions
- Type hints on all function signatures
- Docstrings on all public classes and methods
- `structlog` for all logging (structured JSON output)
- No global mutable state — use dependency injection via FastAPI's `Depends()`
- All configuration via environment variables loaded through `app/config.py`
- Tests in `tests/` directory, mirroring `app/` structure
- Run tests with: `uv run pytest tests/ -v`

### Key Design Decisions
1. **Weighted average scoring, NOT dot product** — prevents rewarding irrelevant model strengths
2. **Kimi cost includes thinking overhead** — multiply output tokens by 1.3x
3. **Failure pattern tracking** — models with `fails_on` entries are excluded for matching task types
4. **Deterministic bypass** — pure math/JSON tasks skip LLM entirely
5. **Local model at $0 cost** — auto-selected by decision engine when accuracy is sufficient
6. **Context length pre-check** — skip local model if prompt exceeds its context window
7. **Max 2 escalation levels** — prevents runaway token cost
8. **In-memory cache only** — no Redis/FAISS, container runs for 10 minutes max
9. **Configurable local model** — swap model via env vars without code changes
10. **Offline benchmarking only** — capability scores generated offline, never inside eval container

### Environment Variables
```
FIREWORKS_API_KEY=<required>
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
DEFAULT_ACCURACY_THRESHOLD=0.8
MAX_ESCALATION_DEPTH=2
CONFIDENCE_THRESHOLD=0.7
LOG_LEVEL=INFO
LOCAL_MODEL_ENABLED=true
LOCAL_MODEL_PATH=models/qwen2.5-3b-instruct-q4_k_m.gguf
LOCAL_MODEL_NAME=local:qwen-2.5-3b
LOCAL_MODEL_CONTEXT_LENGTH=2048
LOCAL_MODEL_THREADS=2
```

## Important Warnings

> **DO NOT** add Redis, FAISS, or any external service dependencies — the container is self-contained.

> **DO NOT** hardcode responses to specific prompts — evaluation uses unseen prompt variations.

> **DO NOT** always route to the most expensive model — this defeats the purpose and scores poorly.

> **DO NOT** use heavyweight ML models for feature extraction — the extractor must be instant (regex/keywords only).

> **DO NOT** benchmark models inside the evaluation container — capability scores must come from offline benchmarks.

## Reference Documents

- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) — Full technical plan with architecture details
- [Architecture Guide](docs/ARCHITECTURE.md) — Component diagrams and data flow
- [Capability Matrix Spec](docs/CAPABILITY_MATRIX.md) — How the capability matrix works
- [Routing Algorithm](docs/ROUTING_ALGORITHM.md) — Decision engine algorithm details
- [Progress Tracker](docs/PROGRESS.md) — Implementation progress checklist
