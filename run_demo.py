"""OptiRoute Execution Trace Demo.

Runs a prompt through the pipeline and prints a step-by-step visual trace 
of the routing decision and execution, designed for hackathon demonstration.
"""
import asyncio
import json
import os
import sys

# Optional rich support if installed
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from app.router.pipeline import RoutingPipeline
from app.config import get_settings

def print_step(title: str, content: str = "", color: str = "\033[94m"):
    """Fallback standard print for steps."""
    reset = "\033[0m"
    print(f"\n{color}=== {title} ==={reset}")
    if content:
        print(content)

def print_rich_step(title: str, content: str, style: str = "blue"):
    """Rich print for steps."""
    if HAS_RICH:
        console.print(Panel(content, title=title, style=style, expand=False))
    else:
        print_step(title, content)

async def run_demo(prompt: str):
    settings = get_settings()
    if not settings.fireworks_api_key:
        print("ERROR: FIREWORKS_API_KEY environment variable is missing.")
        sys.exit(1)

    print("\n🚀 Initializing OptiRoute Pipeline...")
    pipeline = RoutingPipeline()
    await pipeline.initialize()
    print("✅ Pipeline ready.\n")

    if HAS_RICH:
        console.print(Panel(prompt, title="User Prompt", style="bold green", expand=False))
    else:
        print_step("User Prompt", prompt, "\033[92m")

    # Step 1: Normalization (internal to pipeline, but let's just route and trace)
    # We will hook into the pipeline execution by just calling route() and printing the trace data
    
    # We don't have a built-in step-by-step yield, but the `route` returns a detailed payload.
    # To get true step-by-step without rewriting pipeline, we'll re-run the vectorizer here just for the UI.
    
    print("\n🔍 Extracting Features & Building Vectors...")
    features = pipeline._extractor.extract(prompt)
    task_vec, resource_vec, risk_vec = pipeline._vectorizer.generate(features)
    
    if HAS_RICH:
        table = Table(title="Task Vector (Capabilities Required)")
        table.add_column("Dimension", justify="right", style="cyan")
        table.add_column("Weight", justify="left", style="magenta")
        for k, v in task_vec.to_dict().items():
            if v > 0:
                table.add_row(k, f"{v:.2f}")
        console.print(table)
        
        rtable = Table(title="Resource & Risk Vector")
        rtable.add_column("Property", justify="right", style="cyan")
        rtable.add_column("Value", justify="left", style="yellow")
        rtable.add_row("Input Tokens", str(resource_vec.expected_input_tokens))
        rtable.add_row("Output Budget", resource_vec.output_budget_bucket)
        rtable.add_row("Requires JSON", str(risk_vec.requires_json))
        rtable.add_row("Complexity", f"{resource_vec.complexity:.2f}")
        console.print(rtable)
    else:
        print_step("Task Vector", json.dumps({k: v for k, v in task_vec.to_dict().items() if v > 0}, indent=2))
        print_step("Resource Vector", json.dumps(resource_vec.to_dict(), indent=2))

    print("\n🧠 Evaluating Candidate Models...")
    
    # Execute full pipeline
    result = await pipeline.route(prompt)
    
    if HAS_RICH:
        console.print(Panel(
            result["routing_explanation"], 
            title="Decision Engine Output", 
            style="bold cyan", 
            expand=False
        ))
    else:
        print_step("Decision Engine Output", result["routing_explanation"], "\033[96m")

    if result["escalated"]:
        print(f"\n⚠️  Initial model failed confidence validation. Escalated (depth: {result['escalation_depth']}).")

    if HAS_RICH:
        console.print(Panel(
            f"Cost: ${result['cost']:.6f}\nLatency: {result['latency_ms']}ms\nConfidence: {result['confidence']:.2f}",
            title=f"Execution Metrics ({result['model_used']})",
            style="bold yellow",
            expand=False
        ))
        
        # Determine if output is code/json for syntax highlighting
        is_code = "```" in result["response"] or "{" in result["response"][:10]
        if is_code:
            console.print(Panel(
                Syntax(result["response"], "markdown", theme="monokai", word_wrap=True),
                title="Final Response",
                style="bold green"
            ))
        else:
            console.print(Panel(result["response"], title="Final Response", style="bold green"))
    else:
        print_step(f"Execution Metrics ({result['model_used']})", f"Cost: ${result['cost']:.6f} | Latency: {result['latency_ms']}ms | Confidence: {result['confidence']:.2f}", "\033[93m")
        print_step("Final Response", result["response"], "\033[92m")
        
    await pipeline.shutdown()


if __name__ == "__main__":
    default_prompt = "Write a python script to reverse a linked list, but format the output strictly as a JSON object with 'code' and 'explanation' fields."
    
    prompt = sys.argv[1] if len(sys.argv) > 1 else default_prompt
    
    asyncio.run(run_demo(prompt))
