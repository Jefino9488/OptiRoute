# OptiRoute — Implementation Plan

> Canonical copy of the approved implementation plan. See `.agents/AGENTS.md` for quick agent onboarding.

## Project Vision

OptiRoute is an **Adaptive Capability-Based Hybrid AI Routing Framework**. It minimizes inference cost by automatically selecting the cheapest execution pipeline — deterministic tools, a local LLM, or a Fireworks API model — capable of achieving the required accuracy.

**Core question:**
> "What is the cheapest way to answer this prompt correctly — a deterministic tool, a local model, or a Fireworks API model?"

---

## The Models

### Fireworks API Models

| Model | Architecture | Strengths | Input $/1M | Output $/1M | Context |
|---|---|---|---|---|---|
| **minimax-m3** | 428B MoE (sparse) | Frontier-level coding, reasoning, 1M context | $0.30 | $1.20 | 512K |
| **kimi-k2p7-code** | 1T MoE (32B active) | Elite coding, mandatory thinking mode (+1.3× output) | $0.95 | $4.00 | 256K |

> **Note**: Gemma models (`gemma-4-31b-it`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it-nvfp4`) are currently non-functional on Fireworks and have been removed from the active capability matrix.

### Local Model (Configurable)

| Model | Format | Size | Context | Cost |
|---|---|---|---|---|
| **Qwen2.5-3B-Instruct** (default) | GGUF Q4_K_M | ~1.9 GB | 4096 | $0.00 |

Injected by env var `LOCAL_MODEL_PATH`. Swap without code changes.

---

## Architecture Flow

```
Prompt
  ↓
Normalizer (strip filler, canonical form, hash)
  ↓
Cache Check (exact hash → normalized hash)
  ↓ (miss)
Feature Extractor (regex/keyword patterns)
  ↓
Task Vector Builder ({math, reasoning, code, creative, ...})
  ↓
Decision Engine:
  1. Deterministic check (calculator, JSON parser, regex)
  2. Failure filtering (exclude models known to fail on task type)
  3. Capability scoring (weighted average over FULL task vector)
  4. Cost estimation (local model = $0, Fireworks = $$$)
  5. Select cheapest eligible model
  ↓
Context Length Pre-Check:
  - If local model selected AND prompt exceeds 90% of max_context → skip to Fireworks
  ↓
Executor:
  - "local:*" prefix → Local Executor (llama-cpp-python)
  - Otherwise → Fireworks Executor (OpenAI-compatible API)
  ↓
Confidence Validator (empty check, JSON valid, code syntax, length)
  ↓
Confident? → Yes → Cache + Log + Return
           → No  → Escalate (max 2 levels) → Re-execute
```

---

## Implementation Phases

### Phase 1 — Foundation ✅
- `pyproject.toml`, config, logging, schemas, main.py, directory structure

### Phase 2 — Feature Pipeline ✅
- Normalizer, extractor, vectorizer + unit tests

### Phase 3 — Decision Engine ✅
- Capability matrix, decision engine, escalation policy + tests

### Phase 4 — Executors + Pipeline ✅
- Fireworks executor, deterministic tools, pipeline, API endpoints, cache

### Phase 5 — Confidence + Metrics ✅
- Confidence validator, metrics collector, escalation integration

### Phase 6 — Benchmark Engine ✅
- Benchmark datasets, runner, evaluator, matrix generator

### Phase 7 — Polish ✅
- README, Dockerfile, edge cases, final testing

### Phase 8 — Local Model Executor ✅
- [x] Add `llama-cpp-python` to `pyproject.toml` with pre-built CPU wheel index
- [x] Parse `ALLOWED_MODELS` from harness env at runtime in `app/config.py`
- [x] Update `data/capability_matrix.json` — remove gemma, update Qwen2.5-3B scores
- [x] Create `app/executors/local.py` — LocalExecutor with route() + execute()
- [x] Update `app/router/pipeline.py` — 4-level routing chain + local dispatch
- [x] Create `agent.py` — batch evaluation entrypoint (reads /input, writes /output)
- [x] Create `scripts/download_model.sh` — model download helper
- [x] Update `Dockerfile` — multi-stage build, CMD → agent.py
- [x] Create `tests/test_local_executor.py` (21 tests) + `tests/test_agent.py` (6 tests)
- [x] Full test suite: **154 passed, 0 failed**
- [ ] End-to-end verification under 4GB RAM / 2 vCPU constraints (requires GGUF download)

---

## Key Design Decisions

1. **Weighted average scoring** — `Σ(task[dim] × model[dim]) / Σ(task[dim])` — avoids rewarding irrelevant strengths
2. **Full task vector routing** — Uses all 8 dimensions, not just dominant task type, for multi-skill prompts
3. **Kimi thinking overhead** — output tokens multiplied by 1.3x for cost estimation
4. **Failure pattern exclusion** — models with `fails_on` entries skipped for matching tasks
5. **Deterministic bypass** — pure math/JSON skip LLM ($0 cost)
6. **Local model at $0 cost** — auto-selected by decision engine when accuracy is sufficient
7. **Context length pre-check** — skip local model if prompt exceeds its context window
8. **Configurable local model** — swap model via env vars without code changes
9. **Offline benchmarking only** — capability scores generated offline, never inside eval container
10. **Max 2 escalation levels** — prevents runaway cost
11. **In-memory cache only** — no external services, container runs 10 min max
