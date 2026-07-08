# OptiRoute — Architecture Guide

## System Architecture

```
                          Client / Evaluation Harness
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   FastAPI API Layer    │
                         │   POST /v1/route       │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  Request Normalizer    │
                         │  (filler removal,      │
                         │   canonical form,      │
                         │   hashing)             │
                         └───────────┬───────────┘
                                     │
                              ┌──────┴──────┐
                              │ Cache Check  │
                              │ Exact → Norm │
                              └──────┬──────┘
                                     │
                          Hit? ──Yes──→ Return cached
                                     │
                                    No
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  Feature Extractor     │
                         │  (regex, keywords,     │
                         │   pattern matching)    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  Task Vector Builder   │
                         │                        │
                         │  Task:  {math, code,   │
                         │    reasoning, ...}     │
                         │  Resource: {tokens,    │
                         │    context, runtime}   │
                         │  Risk: {json, strict,  │
                         │    deterministic}      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   Decision Engine      │
                         │                        │
                         │  1. Deterministic?     │──Yes──→ Tool Executor ($0)
                         │  2. Filter failures    │
                         │  3. Score capabilities  │
                         │  4. Estimate costs     │
                         │  5. Pick cheapest      │
                         └───────────┬───────────┘
                                     │
                         ┌───────────┴───────────┐
                         │                       │
                   Model prefix?           Model prefix?
                   "local:*"               (remote)
                         │                       │
                         ▼                       ▼
              ┌──────────────────┐   ┌───────────────────────┐
              │  Context Length   │   │  Fireworks Executor    │
              │  Pre-Check       │   │  (OpenAI-compatible)   │
              │  (fits local?)   │   └───────────┬───────────┘
              └────────┬─────────┘               │
                       │                         │
                 Fits?─Yes──→ Local Executor      │
                       │      (llama-cpp-python)  │
                      No──→ Fireworks Executor ───┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Confidence Validator   │
                         │ (empty, JSON, syntax,  │
                         │  length, repetition)   │
                         └───────────┬───────────┘
                                     │
                         Confident? ─No──→ Escalation Policy
                                     │         (next cheapest model,
                                    Yes         max depth = 2)
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  Cache + Log + Return  │
                         │  Metrics Collector     │
                         └───────────────────────┘
```

---

## Three-Stage Execution Policy

```
Priority 1: Deterministic Tools     → $0 cost, instant, 100% accurate
Priority 2: Local Model (2B-3B)     → $0 Fireworks token cost, ~2-5s latency
Priority 3: Fireworks API Models    → $$ cost, lowest latency, highest accuracy
```

The decision engine doesn't explicitly implement these priorities. Instead, it uses the capability matrix where:
- Deterministic tools are checked first via a hardcoded bypass
- The local model has `cost_per_1k = $0.00` in the matrix
- Fireworks models have their real API costs

The existing "pick cheapest meeting accuracy threshold" algorithm naturally selects the local model whenever it's accurate enough, and falls back to Fireworks when it's not.

---

## Component Map

```
app/
├── config.py              ← Settings (env vars, model registry, local model config)
├── logging.py             ← Structured logging (structlog)
│
├── api/
│   ├── router.py          ← FastAPI endpoints
│   └── schemas.py         ← Pydantic request/response models
│
├── features/
│   ├── normalizer.py      ← Prompt normalization + hashing
│   ├── extractor.py       ← Regex/keyword feature detection
│   └── vectorizer.py      ← Task/Resource/Risk vector generation
│
├── router/
│   ├── capability_matrix.py ← Load/query/update capability JSON
│   ├── decision_engine.py   ← CORE: filter → score → cheapest
│   ├── policy.py            ← Escalation rules
│   └── pipeline.py          ← Full request lifecycle orchestration
│
├── executors/
│   ├── base.py            ← ExecutionResult dataclass
│   ├── fireworks.py       ← Fireworks API (OpenAI SDK)
│   ├── local.py           ← Local LLM (llama-cpp-python) — $0 cost
│   └── tools.py           ← Deterministic: calculator, JSON, regex
│
├── confidence/
│   └── validator.py       ← Output quality scoring
│
├── metrics/
│   └── collector.py       ← Per-request metrics tracking
│
├── cache/
│   └── manager.py         ← In-memory exact + normalized cache
│
└── benchmark/
    ├── runner.py           ← Run datasets through all models
    ├── evaluator.py        ← Score model outputs
    └── matrix_generator.py ← Generate capability_matrix.json

models/
└── *.gguf                 ← Local model weights (bundled in Docker image)
```

---

## Data Flow

### Request Processing
```
RouteRequest(prompt, required_accuracy=0.8)
    ↓
NormalizedPrompt(raw, normalized, hash)
    ↓
FeatureVector(contains_code, contains_math, json_required, ...)
    ↓
TaskVector({math: 0.82, reasoning: 0.61, code: 0.05, ...})
ResourceVector({expected_input_tokens: 45, expected_output_tokens: 150, ...})
RiskVector({needs_json: false, strict_formatting: false, ...})
    ↓
RoutingDecision:
  - Simple QA? → model="local:qwen-2.5-3b", cost=$0
  - Complex code? → model="kimi-k2p7-code", cost=$$$
  - General reasoning? → model="gemma-4-26b-a4b-it", cost=$
    ↓
ExecutionResult(response="...", tokens_out=142, cost=0.0, confidence=0.92)
    ↓
RouteResponse(response, model_used, cost, latency_ms, confidence, cache_hit, ...)
```

### Capability Matrix Data Model
```json
{
  "<model-id>": {
    "capabilities": {
      "math": 0.0-1.0,
      "reasoning": 0.0-1.0,
      "code": 0.0-1.0,
      "creative": 0.0-1.0,
      "translation": 0.0-1.0,
      "extraction": 0.0-1.0,
      "retrieval": 0.0-1.0,
      "general_qa": 0.0-1.0
    },
    "cost_per_1k_input": float,
    "cost_per_1k_output": float,
    "avg_output_multiplier": float,
    "max_context": int,
    "fails_on": ["category", ...],
    "avg_latency_ms": float,
    "is_local": bool          // NEW: distinguishes local from API models
  }
}
```

---

## Decision Engine Algorithm

```python
def select_model(task_vector, resource_vector, risk_vector, required_accuracy):
    # Step 1: Deterministic check
    if is_pure_math(task_vector): return "calculator_tool"
    if is_json_parse(task_vector): return "json_tool"

    # Step 2: Failure filtering
    dominant_task = get_dominant_task(task_vector)
    candidates = [m for m in models if dominant_task not in m.fails_on]
    candidates = [m for m in candidates if m.max_context >= resource_vector.context]

    # Step 3: Capability scoring (weighted average over FULL task vector)
    for model in candidates:
        model.predicted_accuracy = (
            sum(task_vector[d] * model.capabilities[d] for d in dimensions)
            / sum(task_vector[d] for d in dimensions)
        )

    # Step 4: Filter by accuracy threshold
    eligible = [m for m in candidates if m.predicted_accuracy >= required_accuracy]

    # Step 5: Cost estimation (local model = $0.00)
    for model in eligible:
        model.estimated_cost = (
            resource_vector.input_tokens * model.cost_per_1k_input / 1000
            + resource_vector.output_tokens * model.avg_output_multiplier
              * model.cost_per_1k_output / 1000
        )

    # Step 6: Select cheapest (local model wins when eligible!)
    return min(eligible, key=lambda m: m.estimated_cost)
```

### Context Length Pre-Check (Pipeline Level)

Before dispatching to the local model, the pipeline verifies:
```python
if current_model.startswith("local:"):
    matrix_entry = capability_matrix.get_model_capabilities(current_model)
    if resource_vector["input_tokens"] > matrix_entry["max_context"] * 0.9:
        # Prompt too long for local model — skip to next cheapest
        continue
```
