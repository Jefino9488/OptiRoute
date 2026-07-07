"""Benchmark runner — executes benchmark datasets against all models.

Runs curated prompt sets through each Fireworks model, collecting
responses, latency, and token usage for capability matrix generation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from app.config import get_settings
from app.executors.fireworks import FireworksExecutor

logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkPrompt:
    """A single benchmark prompt with expected output."""

    prompt: str
    expected_output: str
    category: str
    difficulty: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of running a single prompt through a model."""

    prompt: str
    expected_output: str
    actual_output: str
    category: str
    model_id: str
    tokens_input: int = 0
    tokens_output: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    score: float = 0.0  # Set by evaluator


class BenchmarkRunner:
    """Run benchmark datasets through all available models.

    Parameters
    ----------
    benchmarks_dir : str
        Path to directory containing benchmark JSON files.
    """

    def __init__(self, benchmarks_dir: str = "data/benchmarks") -> None:
        self._benchmarks_dir = Path(benchmarks_dir)
        self._executor = FireworksExecutor()

    def load_dataset(self, category: str) -> list[BenchmarkPrompt]:
        """Load a benchmark dataset by category name.

        Parameters
        ----------
        category : str
            Category name (matches ``{category}.json`` file).

        Returns
        -------
        list[BenchmarkPrompt]
        """
        path = self._benchmarks_dir / f"{category}.json"
        if not path.exists():
            logger.warning("benchmark.dataset_not_found", path=str(path))
            return []

        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        prompts = []
        for item in data:
            prompts.append(BenchmarkPrompt(
                prompt=item["prompt"],
                expected_output=item.get("expected_output", ""),
                category=item.get("category", category),
                difficulty=item.get("difficulty", "medium"),
                metadata=item.get("metadata", {}),
            ))
        logger.info("benchmark.loaded", category=category, count=len(prompts))
        return prompts

    def load_all_datasets(self) -> dict[str, list[BenchmarkPrompt]]:
        """Load all benchmark datasets from the benchmarks directory.

        Returns
        -------
        dict[str, list[BenchmarkPrompt]]
            Mapping of category → list of prompts.
        """
        datasets: dict[str, list[BenchmarkPrompt]] = {}
        if not self._benchmarks_dir.exists():
            return datasets

        for path in sorted(self._benchmarks_dir.glob("*.json")):
            category = path.stem
            prompts = self.load_dataset(category)
            if prompts:
                datasets[category] = prompts
        return datasets

    async def run_model(
        self,
        model_id: str,
        prompts: list[BenchmarkPrompt],
    ) -> list[BenchmarkResult]:
        """Run all prompts through a single model.

        Parameters
        ----------
        model_id : str
            Short model identifier.
        prompts : list[BenchmarkPrompt]
            Prompts to execute.

        Returns
        -------
        list[BenchmarkResult]
        """
        results: list[BenchmarkResult] = []
        for i, bp in enumerate(prompts):
            logger.info(
                "benchmark.running",
                model=model_id,
                category=bp.category,
                progress=f"{i + 1}/{len(prompts)}",
            )
            exec_result = await self._executor.execute(
                prompt=bp.prompt,
                model_id=model_id,
                task_type=bp.category,
            )
            results.append(BenchmarkResult(
                prompt=bp.prompt,
                expected_output=bp.expected_output,
                actual_output=exec_result.response,
                category=bp.category,
                model_id=model_id,
                tokens_input=exec_result.tokens_input,
                tokens_output=exec_result.tokens_output,
                cost=exec_result.cost,
                latency_ms=exec_result.latency_ms,
            ))
        return results

    async def run_all_models(
        self,
        prompts: list[BenchmarkPrompt],
    ) -> dict[str, list[BenchmarkResult]]:
        """Run all prompts through every allowed model.

        Returns
        -------
        dict[str, list[BenchmarkResult]]
            Mapping of model_id → results.
        """
        settings = get_settings()
        all_results: dict[str, list[BenchmarkResult]] = {}
        for model_id in settings.allowed_models:
            logger.info("benchmark.model_start", model=model_id)
            results = await self.run_model(model_id, prompts)
            all_results[model_id] = results
            logger.info(
                "benchmark.model_complete",
                model=model_id,
                count=len(results),
            )
        return all_results
