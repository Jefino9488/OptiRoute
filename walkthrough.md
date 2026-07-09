# OptiRoute — Build Walkthrough

## Summary

Built the complete **Adaptive Capability-Based Hybrid AI Routing Framework** from scratch across 6 conventional commits. The system dynamically selects the cheapest Fireworks AI model capable of meeting the required accuracy for each prompt.

## Git History

```
f446444 docs(project): add README, Dockerfile, and project metadata
4250623 test(suite): add comprehensive tests for pipeline components and evaluator
5ef9476 feat(benchmark): add benchmark runner, evaluator, and matrix generator
c17d8ed feat(pipeline): wire full routing pipeline with API endpoints
a62c3d8 feat(executors): add Fireworks executor, deterministic tools, cache, confidence, and metrics
86389b7 feat(init): scaffold project foundation with FastAPI, config, schemas, and docs
```

All pushed to: **https://github.com/Jefino9488/OptiRoute**

## Components Built

### Phase 1 — Foundation
| File | Purpose |
|------|---------|
| [config.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/config.py) | Pydantic Settings: 5-model registry, thresholds, env vars |
| [logging.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/logging.py) | structlog JSON output |
| [schemas.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/api/schemas.py) | 8 Pydantic v2 request/response models |
| [main.py](file:///home/jefino9488/PycharmProjects/OptiRoute/main.py) | FastAPI app with lifespan, CORS, health |

### Phase 2 — Feature Pipeline
| File | Purpose |
|------|---------|
| [normalizer.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/features/normalizer.py) | Filler removal, whitespace collapse, SHA-256 hashing |
| [extractor.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/features/extractor.py) | Regex/keyword task detection (code, math, JSON, creative, etc.) |
| [vectorizer.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/features/vectorizer.py) | TaskVector + ResourceVector + RiskVector generation |

### Phase 3 — Decision Engine
| File | Purpose |
|------|---------|
| [capability_matrix.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/router/capability_matrix.py) | Load/query/update model capabilities, cost estimation |
| [decision_engine.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/router/decision_engine.py) | 7-step routing: bypass → filter → score → cost → cheapest |
| [policy.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/router/policy.py) | Escalation with depth budget (max 2) |
| [capability_matrix.json](file:///home/jefino9488/PycharmProjects/OptiRoute/data/capability_matrix.json) | 5-model capability scores + costs |

### Phase 4 — Executors & Support
| File | Purpose |
|------|---------|
| [fireworks.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/executors/fireworks.py) | Async OpenAI-compatible executor with per-task temp/token tuning |
| [tools.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/executors/tools.py) | Calculator, JSON parser, regex ($0 bypass) |
| [manager.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/cache/manager.py) | Two-tier in-memory cache (exact + normalized) |
| [validator.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/confidence/validator.py) | Heuristic output quality scoring |
| [collector.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/metrics/collector.py) | Cost/latency/cache/escalation tracking |

### Phase 5 — Pipeline
| File | Purpose |
|------|---------|
| [pipeline.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/router/pipeline.py) | Full request lifecycle orchestrator |
| [router.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/api/router.py) | `/v1/route`, `/v1/models`, `/v1/metrics`, `/v1/cache/stats` |

### Phase 6 — Benchmarks
| File | Purpose |
|------|---------|
| [runner.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/benchmark/runner.py) | Dataset loading + execution across all models |
| [evaluator.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/benchmark/evaluator.py) | Per-category scoring (math, code, QA, creative, translation) |
| [matrix_generator.py](file:///home/jefino9488/PycharmProjects/OptiRoute/app/benchmark/matrix_generator.py) | Generates capability_matrix.json from benchmark results |
| 6 JSON datasets | 90 prompts: math(20), code(20), QA(15), reasoning(15), creative(10), translation(10) |

## Test Results

```
118 passed in 0.42s
```

Test coverage across:
- Normalizer: 11 tests (filler, whitespace, hash, edge cases)
- Extractor: 22 tests (code, math, JSON, creative, translation, reasoning, retrieval, complexity)
- Vectorizer: 16 tests (dominance, bounds, resources, risk, serialization)
- Decision Engine: 15 tests (routing, bypass, fallback, escalation, cost)
- Pipeline Components: 27 tests (tools, cache, confidence, metrics)
- Evaluator: 12 tests (math, code, QA, creative scoring, aggregation)

## What's Ready

- ✅ Full API server starts with `uv run uvicorn main:app`
- ✅ Decision engine routes to cheapest capable model
- ✅ Deterministic bypass for simple math at $0 cost
- ✅ Kimi K2.7 cost includes 1.3× output multiplier
- ✅ Failure pattern filtering (Kimi excluded from creative/translation)
- ✅ Max 2 escalation levels with confidence validation
- ✅ In-memory cache with hit rate tracking
- ✅ Docker-ready deployment

## What Could Be Tuned Before Hackathon

1. **Capability matrix values** — Run actual benchmarks against Fireworks API to get real accuracy numbers
2. **Temperature tuning** — The per-task-type defaults may need adjustment based on real output quality
3. **Confidence thresholds** — May need tuning based on how well the heuristic validator performs
4. **System prompts** — Could add task-specific system prompts to improve model output quality
