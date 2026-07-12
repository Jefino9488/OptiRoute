#!/usr/bin/env python3
"""Run offline benchmarks to update the capability matrix."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from app.benchmark.runner import BenchmarkRunner
from app.benchmark.matrix_generator import CapabilityMatrixGenerator
from app.config import get_settings


async def main() -> None:
    settings = get_settings()
    print(f"Running benchmarks for allowed models: {settings.allowed_models}")
    
    runner = BenchmarkRunner()
    datasets = runner.load_all_datasets()
    if not datasets:
        print("No benchmark datasets found in data/benchmarks/")
        sys.exit(1)
        
    all_prompts = []
    for prompts in datasets.values():
        all_prompts.extend(prompts)
        
    print(f"Loaded {len(all_prompts)} prompts across {len(datasets)} categories.")
    
    all_results = await runner.run_all_models(all_prompts)
    
    generator = CapabilityMatrixGenerator()
    generator.generate_and_save(all_results)
    
    report = generator.generate_report(all_results)
    report_path = Path("data/benchmark_report.md")
    report_path.write_text(report, encoding="utf-8")
    
    print("Generating Routing Analytics Dashboard...")
    from app.benchmark.dashboard import generate_dashboard
    dashboard = await generate_dashboard(all_prompts)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    version_dir = Path(f"data/benchmarks/{timestamp}")
    version_dir.mkdir(parents=True, exist_ok=True)
    
    (version_dir / "routing_dashboard.json").write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    (version_dir / "report.md").write_text(report, encoding="utf-8")
    
    print(f"Benchmarks complete. Matrix updated at data/capability_matrix.json")
    print(f"Report saved to {report_path}")
    print(f"Dashboard and versioned assets saved to {version_dir}")

if __name__ == "__main__":
    asyncio.run(main())
