# OptiRoute — Implementation Plan

> Canonical copy of the approved implementation plan. See `.agents/AGENTS.md` for quick agent onboarding.

## Project Vision

OptiRoute is an **Adaptive Capability-Based Hybrid AI Routing Framework**. It minimizes inference cost by automatically selecting the cheapest Fireworks AI execution pipeline capable of achieving the required accuracy.

**Core question:**
> "Among the allowed Fireworks models, which one is the cheapest that can still solve this task accurately?"

---

## The 5 Models

| Model | Architecture | Strengths | Input $/1M | Output $/1M | Context |
|---|---|---|---|---|---|
| **minimax-m3** | 428B MoE (sparse) | Frontier-level coding, reasoning, 1M context | $0.30 | $1.20 | 512K |
| **kimi-k2p7-code** | 1T MoE (32B active) | Elite coding, mandatory thinking mode | $0.95 | $4.00 | 256K |
| **gemma-4-31b-it** | 31B dense | Strong general reasoning | ~$0.20 | ~$0.40 | 262K |
| **gemma-4-26b-a4b-it** | 26B (A4B sparse) | Cheapest, fast | ~$0.10 | ~$0.30 | 256K |
| **gemma-4-31b-it-nvfp4** | 31B quantized (NF4) | Faster/cheaper than full 31B | ~$0.15 | ~$0.35 | 262K |

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
  3. Capability scoring (weighted average, NOT dot product)
  4. Cost estimation (including Kimi's 1.3x thinking overhead)
  5. Select cheapest eligible model
  ↓
Fireworks Executor (OpenAI-compatible API)
  ↓
Confidence Validator (empty check, JSON valid, code syntax, length)
  ↓
Confident? → Yes → Cache + Log + Return
           → No  → Escalate (max 2 levels) → Re-execute
```

---

## Implementation Phases

### Phase 1 — Foundation (1–2 hours)
- `pyproject.toml`, config, logging, schemas, main.py, directory structure

### Phase 2 — Feature Pipeline (3–4 hours)
- Normalizer, extractor, vectorizer + unit tests

### Phase 3 — Decision Engine (3–4 hours)
- Capability matrix, decision engine, escalation policy + tests

### Phase 4 — Executors + Pipeline (3–4 hours)
- Fireworks executor, deterministic tools, pipeline, API endpoints, cache

### Phase 5 — Confidence + Metrics (2–3 hours)
- Confidence validator, metrics collector, escalation integration

### Phase 6 — Benchmark Engine (4–6 hours)
- Benchmark datasets, runner, evaluator, matrix generator

### Phase 7 — Polish (2–3 hours)
- README, Dockerfile, edge cases, final testing

---

## Key Design Decisions

1. **Weighted average scoring** — `Σ(task[dim] × model[dim]) / Σ(task[dim])` — avoids rewarding irrelevant strengths
2. **Kimi thinking overhead** — output tokens multiplied by 1.3x for cost estimation
3. **Failure pattern exclusion** — models with `fails_on` entries skipped for matching tasks
4. **Deterministic bypass** — pure math/JSON skip LLM ($0 cost)
5. **Max 2 escalation levels** — prevents runaway cost
6. **In-memory cache only** — no external services, container runs 10 min max
