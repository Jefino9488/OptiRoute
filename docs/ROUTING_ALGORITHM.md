# OptiRoute — Routing Algorithm Details

## Overview

The routing algorithm answers one question:
> "What is the cheapest way to answer this prompt correctly — a deterministic tool, a local model, or a Fireworks API model?"

It is fully explainable — no black-box ML, no magic.

---

## Algorithm Steps

### Step 1: Deterministic Check

Before involving any LLM, check if the task can be solved deterministically:

| Task Pattern | Tool | Cost |
|---|---|---|
| Pure arithmetic (`2+2`, `15*23`) | `CalculatorTool` | $0 |
| JSON parsing/validation | `JsonParserTool` | $0 |
| Pattern matching/extraction | `RegexTool` | $0 |

Detection is via the Task Vector — if a single dimension dominates (e.g., `math > 0.95`) AND the prompt is simple enough (short, no explanation needed), bypass LLM.

### Step 2: Failure Filtering

Exclude models known to fail on the dominant task type:

```python
dominant_task = max(task_vector, key=task_vector.get)
candidates = [
    model for model in all_models  # includes local + Fireworks models
    if dominant_task not in model.fails_on
    and model.max_context >= resource_vector.expected_context_length
]
```

**Why this matters**: Sending a creative writing task to Kimi wastes tokens because Kimi is code-specialized and will produce lower-quality creative output. Similarly, sending a complex code task to the local model wastes time since it will likely fail and trigger escalation.

### Step 3: Capability Scoring

For each candidate model (including the local model), predict accuracy using **weighted average across the full task vector**:

```python
predicted_accuracy = (
    sum(task_vector[dim] * model.capabilities[dim] for dim in dimensions)
    / sum(task_vector[dim] for dim in dimensions)
)
```

**Why weighted average, not dot product or cosine similarity?**

Consider a prompt that's 95% math, 5% creative:
- Task vector: `{math: 0.95, creative: 0.05, code: 0.0, ...}`
- Kimi capabilities: `{math: 0.85, creative: 0.65, code: 0.98, ...}`

**Dot product** would reward Kimi for its 0.98 code score, even though coding is irrelevant.
**Cosine similarity** normalizes magnitudes but still rewards alignment with irrelevant dimensions.
**Weighted average** gives math 95% weight and creative 5% weight — correctly focuses on what the task actually needs.

**Why full task vector, not just dominant task type?**

Many prompts contain multiple skills. A prompt like "Write a Python function that calculates Fibonacci numbers and explain the time complexity" has:
- `code: 0.75, reasoning: 0.55, math: 0.30`

Using only `task_type = "code"` would miss the reasoning and math requirements. The full 8-dimensional task vector captures this accurately.

### Step 4: Accuracy Filtering

```python
eligible = [
    model for model in candidates
    if model.predicted_accuracy >= required_accuracy
]
```

Default `required_accuracy = 0.8`. Can be tuned per-request.

### Step 5: Cost Estimation

```python
for model in eligible:
    model.estimated_cost = (
        resource_vector.input_tokens * model.cost_per_1k_input / 1000
        + resource_vector.output_tokens
          * model.avg_output_multiplier  # Kimi = 1.3x
          * model.cost_per_1k_output / 1000
    )
```

**Local model cost**: `cost_per_1k_input = 0.0` and `cost_per_1k_output = 0.0`, so `estimated_cost = $0.00` always. This means the local model is automatically preferred whenever it meets the accuracy threshold.

**Kimi's hidden cost**: Kimi K2.7 Code uses mandatory thinking mode, generating ~30% more output tokens than other models for the same task. `avg_output_multiplier = 1.3` captures this.

### Step 6: Select Cheapest

```python
winner = min(eligible, key=lambda m: m.estimated_cost)
```

If no model is eligible (all predicted accuracy < threshold), fall back to the most capable model (minimax-m3).

**In practice**: For simple QA/retrieval tasks, the local model wins. For complex code/math/reasoning, Fireworks models win.

---

## Context Length Pre-Check

Before executing on the local model, the pipeline performs a context length pre-check:

```python
if current_model.startswith("local:"):
    model_entry = capability_matrix.get_model_capabilities(current_model)
    max_ctx = model_entry.get("max_context", 2048)
    estimated_tokens = resource_vector.get("input_tokens", 0)
    if estimated_tokens > max_ctx * 0.9:
        # Skip local model — prompt too long
        # Continue to next cheapest (Fireworks) model
        continue
```

This prevents wasting 2-5 seconds on a local model inference that will almost certainly fail due to context overflow.

---

## Escalation

After execution, the Confidence Validator checks the response. If confidence < threshold:

1. Mark current model as failed for this specific prompt
2. Select next cheapest eligible model
3. Re-execute (max 2 escalation levels)
4. If all escalations exhausted, return best response seen

**Local → Fireworks escalation**: If the local model produces a low-confidence response, the escalation policy automatically tries the next cheapest model — which will be a Fireworks API model. This means a failed local attempt has zero Fireworks token cost.

---

## Routing Decision Output

Every decision includes a human-readable explanation:

```json
{
  "model_selected": "local:qwen-2.5-3b",
  "estimated_cost": 0.0,
  "predicted_accuracy": 0.73,
  "reasoning": "Task is primarily general_qa (0.73). local:qwen-2.5-3b meets accuracy threshold (0.73 >= 0.70) at lowest cost ($0.00000). Zero Fireworks token cost.",
  "alternatives_considered": [
    {"model": "gemma-4-26b-a4b-it", "cost": 0.00003, "accuracy": 0.85},
    {"model": "gemma-4-31b-it-nvfp4", "cost": 0.00004, "accuracy": 0.87}
  ]
}
```
