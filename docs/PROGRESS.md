# Local Model Executor — Progress Tracker

> **Branch:** `feat/local-model-executor`
> **Last Updated:** 2026-07-08

## Phase 1: Documentation & Architecture Updates
- [x] Update `docs/ARCHITECTURE.md` — Add local model layer to diagrams
- [x] Update `docs/ROUTING_ALGORITHM.md` — Add context length pre-check + local routing
- [x] Update `docs/CAPABILITY_MATRIX.md` — Document `is_local` flag + $0 cost entries
- [x] Update `docs/IMPLEMENTATION_PLAN.md` — Add Phase 8: Local Model Executor
- [x] Update `.agents/AGENTS.md` — Update architecture, constraints, component table
- [x] Commit: `docs: update architecture for hybrid local+remote routing`

## Phase 2: Configuration & Capability Matrix
- [x] Add local model settings to `app/config.py`
- [x] Add local model entry to `data/capability_matrix.json`
- [x] Commit: `feat(config): add configurable local model settings and capability entry`

## Phase 3: Local Model Executor
- [x] Create `app/executors/local.py` — LocalExecutor class (route + execute)
- [x] Update `app/router/pipeline.py` — 4-level routing chain + local dispatch
- [x] Add `llama-cpp-python` to `pyproject.toml` with pre-built CPU wheel index
- [x] Create `scripts/download_model.sh` — Model download helper
- [x] Commit: `feat(executor): add LocalExecutor — Qwen2.5-3B GGUF via llama-cpp-python`
- [x] Commit: `feat(router): wire LocalExecutor into RoutingPipeline with 4-level decision chain`

## Phase 4: Batch Evaluation Entrypoint
- [x] Create `agent.py` — reads `/input/tasks.json`, writes `/output/results.json`, exits 0
- [x] Commit: `feat(agent): add batch evaluation entrypoint for hackathon submission`

## Phase 5: Docker & Deployment
- [x] Update `Dockerfile` — Multi-stage build (builder → model-downloader → final)
- [x] Add `.dockerignore` to exclude spec/, docs/, tests/, .env from build context
- [x] CMD changed from `uvicorn` to `uv run python agent.py`
- [x] Commit: `build(docker): multi-stage build with bundled GGUF model and agent.py CMD`

## Phase 6: Tests
- [x] Create `tests/test_local_executor.py` — 21 mock-based unit tests (no GGUF needed)
  - All 8 evaluation categories covered: factual, sentiment, NER, summarisation,
    code debug, code gen, complex math, logical reasoning
- [x] Create `tests/test_agent.py` — 6 integration tests for batch I/O contract
- [x] Update `tests/test_decision_engine.py` — Fix 3 stale tests referencing removed gemma models
- [x] Run full test suite: **154 passed, 0 failed**
- [x] Commit: `test: add LocalExecutor and agent tests; fix stale decision engine tests`

## Phase 7: End-to-End Verification
- [ ] Local test: simple prompt → local model selected (cost: $0)
- [ ] Local test: complex code prompt → kimi-k2p7-code selected
- [ ] Local test: long prompt → local model skipped (context overflow)
- [ ] Docker build: verify image < 10 GB compressed
- [ ] Docker run: verify works under `--memory=4g --cpus=2`

## Design Decisions
1. **Local LLM router** — Qwen2.5-3B used for routing decisions (max_tokens=15, temp=0.0) AND execution
2. **Batch CLI entrypoint** — `agent.py` replaces HTTP server for evaluation harness
3. **ALLOWED_MODELS from env** — Parsed from harness-injected env var at runtime, not hardcoded
4. **Qwen2.5-3B Q4_K_M** — Chosen over 1.5B for better NER/sentiment/summarisation accuracy
5. **required_accuracy=0.75** — Lowered from 0.80 to allow local model to win routing for its strong categories
6. **Context length pre-check** — Skip local model if estimated tokens > 90% of max_context (4096)
7. **Gemma models removed** — Not working on Fireworks; removed from matrix and fallback list
