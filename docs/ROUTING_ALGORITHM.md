# OptiRoute — Routing Algorithm Details

## Overview

The routing algorithm answers one question:
> "Among the allowed Fireworks models, which one is the cheapest that can still solve this task accurately?"

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
    model for model in all_models
    if dominant_task not in model.fails_on
    and model.max_context >= resource_vector.expected_context_length
]
```

**Why this matters**: Sending a creative writing task to Kimi wastes tokens because Kimi is code-specialized and will produce lower-quality creative output.

### Step 3: Capability Scoring

For each candidate model, predict accuracy using **weighted average**:

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

**Kimi's hidden cost**: Kimi K2.7 Code uses mandatory thinking mode, generating ~30% more output tokens than other models for the same task. `avg_output_multiplier = 1.3` captures this.

### Step 6: Select Cheapest

```python
winner = min(eligible, key=lambda m: m.estimated_cost)
```

If no model is eligible (all predicted accuracy < threshold), fall back to the most capable model (minimax-m3).

---

## Escalation

After execution, the Confidence Validator checks the response. If confidence < threshold:

1. Mark current model as failed for this specific prompt
2. Select next cheapest eligible model
3. Re-execute (max 2 escalation levels)
4. If all escalations exhausted, return best response seen

---

## Routing Decision Output

Every decision includes a human-readable explanation:

```json
{
  "model_selected": "gemma-4-26b-a4b-it",
  "estimated_cost": 0.00003,
  "predicted_accuracy": 0.85,
  "reasoning": "Task is primarily general_qa (0.72). gemma-4-26b-a4b-it meets accuracy threshold (0.85 >= 0.80) at lowest cost ($0.00003). Excluded kimi-k2p7-code (fails_on includes creative). Preferred over gemma-4-31b-it ($0.00005).",
  "alternatives_considered": [
    {"model": "gemma-4-31b-it-nvfp4", "cost": 0.00004, "accuracy": 0.87},
    {"model": "gemma-4-31b-it", "cost": 0.00005, "accuracy": 0.89}
  ]
}
```
