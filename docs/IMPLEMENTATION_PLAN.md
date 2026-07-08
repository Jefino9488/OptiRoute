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
| **kimi-k2p7-code** | 1T MoE (32B active) | Elite coding, mandatory thinking mode | $0.95 | $4.00 | 256K |
| **gemma-4-31b-it** | 31B dense | Strong general reasoning | ~$0.20 | ~$0.40 | 262K |
| **gemma-4-26b-a4b-it** | 26B (A4B sparse) | Cheapest, fast | ~$0.10 | ~$0.30 | 256K |
| **gemma-4-31b-it-nvfp4** | 31B quantized (NF4) | Faster/cheaper than full 31B | ~$0.15 | ~$0.35 | 262K |

### Local Model (Configurable)

| Model | Format | Size | Context | Cost |
|---|---|---|---|---|
| **Qwen 2.5 3B Instruct** (default) | GGUF Q4_K_M | ~1.93 GB | 2048 | $0.00 |
| **Gemma 3 1B** (fallback) | GGUF Q4_K_M | ~806 MB | 2048 | $0.00 |

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

### Phase 8 — Local Model Executor 🔄
- [ ] Update all documentation for hybrid architecture
- [ ] Add configurable local model settings to `app/config.py`
- [ ] Add local model entry to `data/capability_matrix.json`
- [ ] Create `app/executors/local.py` — LocalExecutor class
- [ ] Update `app/router/pipeline.py` — prefix dispatch + context length pre-check
- [ ] Add `llama-cpp-python` to `pyproject.toml`
- [ ] Create `scripts/download_model.sh`
- [ ] Update `Dockerfile` — bundle model weights
- [ ] Create `tests/test_local_executor.py`
- [ ] End-to-end verification under 4GB RAM / 2 vCPU constraints

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
