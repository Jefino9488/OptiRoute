"""Capability matrix generator — builds capability_matrix.json from benchmark results.

Consumes evaluator output and produces the JSON file that drives all
routing decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from app.benchmark.evaluator import BenchmarkEvaluator
from app.benchmark.runner import BenchmarkResult
from app.router.capability_matrix import CapabilityMatrix, TASK_DIMENSIONS

logger = structlog.get_logger(__name__)

# Failure threshold — if a model scores below this on a category,
# the category is added to its fails_on list.
_FAILURE_THRESHOLD: float = 0.5


class CapabilityMatrixGenerator:
    """Generate a capability matrix from benchmark evaluation results.

    Parameters
    ----------
    matrix_path : str
        Path to the capability matrix JSON file.
    """

    def __init__(self, matrix_path: str = "data/capability_matrix.json") -> None:
        self._matrix_path = Path(matrix_path)
        self._evaluator = BenchmarkEvaluator()

    def generate(
        self,
        all_results: dict[str, list[BenchmarkResult]],
    ) -> dict[str, Any]:
        """Generate an updated capability matrix from benchmark results.

        Parameters
        ----------
        all_results : dict[str, list[BenchmarkResult]]
            Mapping of model_id → list of (scored) results.

        Returns
        -------
        dict[str, Any]
            The generated matrix data.
        """
        # Score all results
        scored_results: list[BenchmarkResult] = []
        sample_counts: dict[str, dict[str, int]] = {}
        for model_id, results in all_results.items():
            self._evaluator.evaluate(results)
            scored_results.extend(results)
            
            sample_counts[model_id] = {}
            for r in results:
                sample_counts[model_id][r.category] = sample_counts[model_id].get(r.category, 0) + 1

        # Aggregate scores
        aggregated = self._evaluator.aggregate_scores(scored_results)

        # Load existing matrix (preserve cost data)
        try:
            matrix = CapabilityMatrix(str(self._matrix_path))
            existing_data = {
                model_id: matrix.get_model_capabilities(model_id) or {}
                for model_id in matrix.get_all_models()
            }
        except Exception:
            existing_data = {}

        # Build updated matrix
        updated: dict[str, Any] = {}
        
        # First, copy over existing models that weren't tested this time
        for model_id, data in existing_data.items():
            if model_id not in aggregated:
                updated[model_id] = data

        for model_id, category_scores in aggregated.items():
            # Preserve existing cost data for the tested models
            existing = existing_data.get(model_id, {})

            # Map category scores to capability dimensions
            capabilities: dict[str, float] = {}
            for dim in TASK_DIMENSIONS:
                # Use benchmark score if available, else preserve existing
                if dim in category_scores:
                    capabilities[dim] = category_scores[dim]
                elif dim in existing.get("capabilities", {}):
                    capabilities[dim] = existing["capabilities"][dim]
                else:
                    capabilities[dim] = 0.5  # Default

            samples: dict[str, int] = sample_counts.get(model_id, {})

            updated[model_id] = {
                "capabilities": capabilities,
                "cost_per_1k_input": existing.get("cost_per_1k_input", 0.0002),
                "cost_per_1k_output": existing.get("cost_per_1k_output", 0.0004),
                "avg_output_multiplier": existing.get("avg_output_multiplier", 1.0),
                "max_context": existing.get("max_context", 256000),
                "samples": samples,
                "avg_latency_ms": self._compute_avg_latency(
                    all_results.get(model_id, [])
                ),
            }

        logger.info(
            "matrix_generator.complete",
            models=list(updated.keys()),
            path=str(self._matrix_path),
        )
        return updated

    def generate_and_save(
        self,
        all_results: dict[str, list[BenchmarkResult]],
    ) -> dict[str, Any]:
        """Generate the matrix and save to disk.

        Returns the generated matrix data.
        """
        import datetime
        data = self.generate(all_results)
        self._matrix_path.parent.mkdir(parents=True, exist_ok=True)
        with self._matrix_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        version_dir = self._matrix_path.parent / "benchmarks" / timestamp
        version_dir.mkdir(parents=True, exist_ok=True)
        with (version_dir / "capability_matrix.json").open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            
        logger.info("matrix_generator.saved", path=str(self._matrix_path))
        return data

    def generate_report(
        self,
        all_results: dict[str, list[BenchmarkResult]],
    ) -> str:
        """Generate a human-readable markdown benchmark report.

        Returns
        -------
        str
            Markdown-formatted report.
        """
        # Score all results
        scored_results: list[BenchmarkResult] = []
        for results in all_results.values():
            self._evaluator.evaluate(results)
            scored_results.extend(results)

        aggregated = self._evaluator.aggregate_scores(scored_results)

        lines: list[str] = [
            "# OptiRoute Benchmark Report\n",
            "## Model Performance by Capability\n",
        ]

        # Group by category -> model
        by_category: dict[str, dict[str, list[BenchmarkResult]]] = {}
        for model_id, results in all_results.items():
            for r in results:
                if r.category not in by_category:
                    by_category[r.category] = {}
                if model_id not in by_category[r.category]:
                    by_category[r.category][model_id] = []
                by_category[r.category][model_id].append(r)

        for category in sorted(by_category.keys()):
            lines.append(f"### {category.capitalize()}\n")
            lines.append("| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |")
            lines.append("|---|---|---|---|---|---|")
            
            for model_id in sorted(by_category[category].keys()):
                results = by_category[category][model_id]
                accuracy = sum(r.score for r in results) / len(results)
                avg_latency = sum(r.latency_ms for r in results) / len(results)
                avg_out_tokens = sum(r.tokens_output for r in results) / len(results)
                avg_cost = sum(r.cost for r in results) / len(results)
                
                # Recommended routing threshold
                if accuracy < _FAILURE_THRESHOLD:
                    rec = "DO NOT ROUTE"
                else:
                    rec = f">= {accuracy - 0.05:.2f}"
                    
                lines.append(f"| {model_id} | {accuracy:.2f} | {avg_latency:.0f}ms | {avg_out_tokens:.0f} | ${avg_cost:.6f} | {rec} |")
            lines.append("")

        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _compute_avg_latency(results: list[BenchmarkResult]) -> float:
        """Compute average latency across results."""
        if not results:
            return 0.0
        return round(
            sum(r.latency_ms for r in results) / len(results), 1
        )
