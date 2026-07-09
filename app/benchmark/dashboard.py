"""Routing Analytics Dashboard Generator."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import structlog

from app.benchmark.evaluator import BenchmarkEvaluator
from app.benchmark.runner import BenchmarkPrompt, BenchmarkResult
from app.router.pipeline import RoutingPipeline

logger = structlog.get_logger(__name__)


async def generate_dashboard(prompts: list[BenchmarkPrompt]) -> dict[str, Any]:
    """Run all prompts through the full routing pipeline and generate stats."""
    pipeline = RoutingPipeline()
    evaluator = BenchmarkEvaluator()
    
    distribution: dict[str, int] = defaultdict(int)
    escalations: dict[str, int] = defaultdict(int)
    cost_by_category: dict[str, float] = defaultdict(float)
    count_by_category: dict[str, int] = defaultdict(int)
    
    results_by_category: dict[str, list[BenchmarkResult]] = defaultdict(list)
    
    for bp in prompts:
        decision, result = await pipeline.route(bp.prompt, task_type=bp.category)
        
        model_used = result.model_used
        distribution[model_used] += 1
        
        cost_by_category[bp.category] += result.cost
        count_by_category[bp.category] += 1
        
        if decision.alternatives_considered:
            if len(decision.alternatives_considered) > 0:
                first_model = decision.alternatives_considered[0].get("model", "unknown")
                if first_model != model_used:
                    path = f"{first_model} -> {model_used}"
                    escalations[path] += 1
                    
        br = BenchmarkResult(
            prompt=bp.prompt,
            expected_output=bp.expected_output,
            actual_output=result.response,
            category=bp.category,
            model_id=model_used,
            cost=result.cost,
            tokens_input=result.tokens_input,
            tokens_output=result.tokens_output,
        )
        results_by_category[bp.category].append(br)

    accuracy_stats = {}
    total_escalations_for_rate = sum(escalations.values())
    
    for category, results in results_by_category.items():
        evaluator.evaluate(results)
        
        local_success = 0
        local_total = 0
        remote_success = 0
        remote_total = 0
        
        for r in results:
            if r.model_id.startswith("local:"):
                local_total += 1
                local_success += int(r.score >= 0.5)
            else:
                remote_total += 1
                remote_success += int(r.score >= 0.5)
                
        total = local_total + remote_total
        
        accuracy_stats[category] = {
            "local_success_rate": f"{(local_success / local_total * 100) if local_total else 0:.1f}%",
            "remote_success_rate": f"{(remote_success / remote_total * 100) if remote_total else 0:.1f}%",
            "escalation_rate": f"{(total_escalations_for_rate / total * 100) if total > 0 else 0:.1f}%",
            "total_routed": total
        }

    total_prompts = sum(distribution.values())
    dist_percentages = {
        model: f"{(count / total_prompts * 100):.1f}%" 
        for model, count in distribution.items()
    }
    
    avg_costs = {
        cat: f"${(cost_by_category[cat] / count_by_category[cat]):.6f}"
        for cat in cost_by_category
    }

    dashboard = {
        "routing_distribution": dist_percentages,
        "escalations": dict(escalations),
        "average_cost_per_category": avg_costs,
        "accuracy_by_category": accuracy_stats
    }
    
    return dashboard
