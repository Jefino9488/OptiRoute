# Local Model Executor — Progress Tracker

> **Branch:** `feat/local-model-executor`
> **Last Updated:** 2026-07-08

## Phase 1: Documentation & Architecture Updates
- [ ] Update `docs/ARCHITECTURE.md` — Add local model layer to diagrams
- [ ] Update `docs/ROUTING_ALGORITHM.md` — Add context length pre-check + local routing
- [ ] Update `docs/CAPABILITY_MATRIX.md` — Document `is_local` flag + $0 cost entries
- [ ] Update `docs/IMPLEMENTATION_PLAN.md` — Add Phase 8: Local Model Executor
- [ ] Update `.agents/AGENTS.md` — Update architecture, constraints, component table
- [ ] Update `README.md` — Update features, architecture diagram, env vars
- [ ] Commit: `docs: update architecture for hybrid local+remote routing`

## Phase 2: Configuration & Capability Matrix
- [ ] Add local model settings to `app/config.py`
- [ ] Add local model entry to `data/capability_matrix.json`
- [ ] Commit: `feat(config): add configurable local model settings and capability entry`

## Phase 3: Local Model Executor
- [ ] Create `app/executors/local.py` — LocalExecutor class
- [ ] Update `app/router/pipeline.py` — Add prefix dispatch + context length pre-check
- [ ] Add `llama-cpp-python` to `pyproject.toml`
- [ ] Create `scripts/download_model.sh` — Model download helper
- [ ] Commit: `feat(executor): add local model executor with llama-cpp-python`

## Phase 4: Docker & Deployment
- [ ] Update `Dockerfile` — Bundle model weights + build deps
- [ ] Commit: `feat(docker): bundle local model weights in container image`

## Phase 5: Tests
- [ ] Create `tests/test_local_executor.py` — Unit tests (mock-based)
- [ ] Run full test suite regression
- [ ] Commit: `test(local): add unit tests for local model executor`

## Phase 6: End-to-End Verification
- [ ] Local test: simple prompt → local model selected (cost: $0)
- [ ] Local test: complex code prompt → Fireworks model selected
- [ ] Local test: long prompt → local model skipped (context overflow)
- [ ] Local test: `required_accuracy=0.9` → local model skipped (accuracy too low)
- [ ] Docker build: verify image < 10 GB compressed
- [ ] Docker run: verify works under `--memory=4g --cpus=2`

## Design Decisions
1. **Configurable model** — Not hardcoded; swap via env vars (`LOCAL_MODEL_PATH`, etc.)
2. **Full task vector routing** — Uses 8-dim weighted accuracy from capability matrix, not just task type
3. **Context length pre-check** — Skip local model if estimated tokens > `max_context`
4. **Offline benchmarking only** — Capability scores come from offline benchmarks, never computed in eval container
5. **Zero decision engine changes** — Local model at `cost=$0` is auto-preferred by existing algorithm
