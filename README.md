# OptiRoute

**Adaptive Capability-Based Hybrid AI Routing Framework**

> An intelligent routing framework that minimizes inference cost by automatically selecting the cheapest execution pipeline capable of achieving the required accuracy.

## Core Concept

```
Prompt → Normalise → Extract Features → Build Task Vector
    → Decision Engine (capability matching + cost estimation)
    → Execute via Fireworks AI → Validate Confidence → Escalate if needed
    → Return Response + Metrics
```

OptiRoute doesn't blindly send everything to the most expensive model. Instead, it:

1. **Understands** what the task needs (math, code, reasoning, creative, etc.)
2. **Estimates** difficulty and resource requirements
3. **Finds** the cheapest Fireworks model that meets the accuracy threshold
4. **Validates** output quality and escalates only if confidence is low
5. **Caches** results to avoid redundant API calls

## Allowed Models

| Model | Architecture | Best For | Relative Cost |
|-------|-------------|----------|---------------|
| `gemma-4-26b-a4b-it` | 26B sparse | Lightweight general tasks | 💲 Cheapest |
| `gemma-4-31b-it-nvfp4` | 31B quantized | Budget reasoning | 💲💲 |
| `gemma-4-31b-it` | 31B dense | Strong reasoning/general | 💲💲💲 |
| `kimi-k2p7-code` | 1T MoE (32B active) | Code specialist | 💲💲💲💲 |
| `minimax-m3` | 428B MoE | Frontier quality | 💲💲💲💲💲 |

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Fireworks AI API key

### Setup

```bash
# Clone the repo
git clone https://github.com/Jefino9488/OptiRoute.git
cd OptiRoute

# Install dependencies
uv sync

# Set your API key
export FIREWORKS_API_KEY=your_key_here

# Run the server
uv run uvicorn main:app --reload --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/route` | Route a prompt to the optimal model |
| `GET` | `/v1/models` | List all models with capabilities |
| `GET` | `/v1/metrics` | View routing metrics |
| `GET` | `/v1/cache/stats` | Cache hit/miss statistics |
| `GET` | `/health` | Health check |

### Example Request

```bash
curl -X POST http://localhost:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 15 * 23?", "required_accuracy": 0.8}'
```

### Example Response

```json
{
  "response": "345",
  "model_used": "deterministic:calculator",
  "cost": 0.0,
  "latency_ms": 0.1,
  "confidence": 1.0,
  "cache_hit": false,
  "escalated": false,
  "task_vector": {"math": 0.95, "reasoning": 0.45, "code": 0.05, ...},
  "routing_explanation": "Task handled by deterministic tool (deterministic:calculator). No LLM needed — $0 cost."
}
```

## Architecture

```
app/
├── api/              # FastAPI routes and Pydantic schemas
├── features/         # Normalizer, extractor, vectorizer
├── router/           # Capability matrix, decision engine, pipeline
├── executors/        # Fireworks AI + deterministic tools
├── confidence/       # Output quality validation
├── cache/            # In-memory two-tier cache
├── metrics/          # Cost/latency tracking
├── benchmark/        # Model benchmarking and matrix generation
data/
├── capability_matrix.json    # Model capabilities (drives routing)
└── benchmarks/               # Benchmark datasets per category
docs/
├── ARCHITECTURE.md           # Component diagrams
├── ROUTING_ALGORITHM.md      # Decision engine details
├── CAPABILITY_MATRIX.md      # Matrix specification
└── IMPLEMENTATION_PLAN.md    # Full implementation plan
```

## Key Design Decisions

1. **Weighted average scoring** — `Σ(task[d]×cap[d]) / Σ(task[d])` prevents rewarding irrelevant model strengths
2. **Kimi cost penalty** — Output tokens multiplied by 1.3× due to mandatory thinking mode
3. **Deterministic bypass** — Simple math/JSON tasks skip LLM entirely at $0 cost
4. **Max 2 escalation levels** — Prevents runaway token cost
5. **In-memory cache only** — No Redis/FAISS; the evaluation container runs for max 10 minutes

## Running Tests

```bash
FIREWORKS_API_KEY=test_key uv run pytest tests/ -v
```

## Docker

```bash
docker build -t optiroute .
docker run -e FIREWORKS_API_KEY=your_key -p 8000:8000 optiroute
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FIREWORKS_API_KEY` | ✅ | — | Fireworks AI API key |
| `FIREWORKS_BASE_URL` | ❌ | `https://api.fireworks.ai/inference/v1` | API base URL |
| `DEFAULT_ACCURACY_THRESHOLD` | ❌ | `0.8` | Min accuracy for model selection |
| `MAX_ESCALATION_DEPTH` | ❌ | `2` | Max escalation retries |
| `CONFIDENCE_THRESHOLD` | ❌ | `0.7` | Min confidence before escalation |
| `LOG_LEVEL` | ❌ | `INFO` | Logging level |

## License

MIT
