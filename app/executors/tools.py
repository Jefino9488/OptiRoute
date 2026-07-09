"""Deterministic tool executors — handle tasks without LLM inference.

These tools bypass the Fireworks API entirely, producing results at
$0 cost and near-zero latency.
"""

from __future__ import annotations

import ast
import json
import math
import re
import time
from typing import Any

import structlog

from app.executors.base import ExecutionResult

logger = structlog.get_logger(__name__)


class CharCounterTool:
    """Exact character or substring counting — always correct, $0 cost.

    Handles prompts like:
      - "Count exactly how many times the letter 'e' appears in '...'"
      - "How many times does 'the' appear in '...'"
      - "How many X are in '...'"
    """

    _PATTERN = re.compile(
        r"(?:count\s+(?:exactly\s+)?(?:how\s+many\s+times\s+)?(?:the\s+)?(?:letter\s+|character\s+|word\s+)?['\"]?(\w+)['\"]?\s+(?:appears?|occurs?|is\s+(?:in|there))"
        r"|how\s+many\s+times\s+(?:does\s+)?['\"]?(\w+)['\"]?\s+(?:appears?|occurs?)"
        r"|how\s+many\s+['\"]?(\w+)['\"]?\s+(?:are\s+(?:there\s+)?in|appear))",
        re.IGNORECASE,
    )
    # Single-quoted and double-quoted strings in the prompt
    _BODY = re.compile(r"""(?:'([^']+)'|\"([^\"]+)\")""")

    def can_handle(self, prompt: str) -> bool:
        """Return True if the prompt is an exact-count request with quoted body."""
        if not self._PATTERN.search(prompt):
            return False
        # _BODY has two groups (single-quoted, double-quoted); flatten and filter
        bodies = [
            g for pair in self._BODY.findall(prompt) for g in pair if g and len(g) >= 10
        ]
        return len(bodies) >= 1

    def execute(self, prompt: str) -> ExecutionResult:
        """Count occurrences of the target in the quoted body text."""
        start = time.perf_counter()
        m = self._PATTERN.search(prompt)
        target = next((g for g in (m.groups() if m else []) if g), None)
        # Body = longest quoted segment (exclude the target itself)
        # _BODY has two groups; flatten tuples from findall
        all_pairs = self._BODY.findall(prompt)
        all_quoted = [g for pair in all_pairs for g in pair if g]
        body = max(
            (b for b in all_quoted if b != target and len(b) >= 10),
            key=len,
            default=None,
        )
        if not target or not body:
            return ExecutionResult(
                response="Cannot parse counting request — target or body not found.",
                model_used="deterministic:counter",
                confidence=0.0,
            )
        count = body.lower().count(target.lower())
        elapsed = (time.perf_counter() - start) * 1000
        logger.info("counter.result", target=target, count=count)
        return ExecutionResult(
            response=str(count),
            model_used="deterministic:counter",
            cost=0.0,
            latency_ms=round(elapsed, 3),
            confidence=1.0,
        )


class CalculatorTool:
    """Safely evaluate simple arithmetic expressions.

    Supports: ``+``, ``-``, ``*``, ``/``, ``**``, ``%``, ``()``.
    Uses ``ast.literal_eval`` for safety — no arbitrary code execution.
    """

    # Match the core arithmetic expression embedded in natural language.
    _EXPR_RE = re.compile(
        r"([\d]+(?:\.\d+)?\s*[+\-*/%^]\s*[\d]+(?:\.\d+)?(?:\s*[+\-*/%^]\s*[\d]+(?:\.\d+)?)*)"
    )

    def can_handle(self, prompt: str) -> bool:
        """Return True if the prompt contains a simple arithmetic expression."""
        return bool(self._EXPR_RE.search(prompt))

    def execute(self, prompt: str) -> ExecutionResult:
        """Evaluate the arithmetic expression found in *prompt*."""
        start = time.perf_counter()
        match = self._EXPR_RE.search(prompt)
        if not match:
            return ExecutionResult(
                response="Could not find an arithmetic expression.",
                model_used="deterministic:calculator",
                confidence=0.0,
            )

        expr = match.group(1).replace("^", "**")
        try:
            # Compile and evaluate safely.
            tree = ast.parse(expr, mode="eval")
            # Only allow numeric literals and basic operators.
            for node in ast.walk(tree):
                if not isinstance(
                    node,
                    (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                     ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                     ast.FloorDiv, ast.USub),
                ):
                    raise ValueError(f"Unsupported operation: {type(node).__name__}")
            result = eval(compile(tree, "<calc>", "eval"))  # noqa: S307
        except Exception as exc:
            logger.warning("calculator.eval_error", expr=expr, error=str(exc))
            return ExecutionResult(
                response=f"Error evaluating expression: {exc}",
                model_used="deterministic:calculator",
                confidence=0.0,
            )

        elapsed = (time.perf_counter() - start) * 1000
        logger.info("calculator.result", expr=expr, result=result)
        return ExecutionResult(
            response=str(result),
            model_used="deterministic:calculator",
            cost=0.0,
            latency_ms=round(elapsed, 3),
            confidence=1.0,
        )


class JsonParserTool:
    """Parse, validate, and pretty-print JSON."""

    def can_handle(self, prompt: str) -> bool:
        """Return True if the prompt is asking to parse/validate JSON."""
        lower = prompt.lower()
        return ("parse" in lower or "validate" in lower or "format" in lower) and "json" in lower

    def execute(self, prompt: str) -> ExecutionResult:
        """Try to find and parse JSON in the prompt."""
        start = time.perf_counter()
        # Try to find a JSON object or array in the prompt.
        json_match = re.search(r'(\{[^}]+\}|\[[^\]]+\])', prompt)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                result = json.dumps(parsed, indent=2)
                elapsed = (time.perf_counter() - start) * 1000
                return ExecutionResult(
                    response=result,
                    model_used="deterministic:json",
                    cost=0.0,
                    latency_ms=round(elapsed, 3),
                    confidence=1.0,
                )
            except json.JSONDecodeError as exc:
                elapsed = (time.perf_counter() - start) * 1000
                return ExecutionResult(
                    response=f"Invalid JSON: {exc}",
                    model_used="deterministic:json",
                    cost=0.0,
                    latency_ms=round(elapsed, 3),
                    confidence=0.5,
                )

        elapsed = (time.perf_counter() - start) * 1000
        return ExecutionResult(
            response="No JSON found in the prompt.",
            model_used="deterministic:json",
            cost=0.0,
            latency_ms=round(elapsed, 3),
            confidence=0.0,
        )


class RegexTool:
    """Pattern matching and extraction using regex."""

    def can_handle(self, prompt: str) -> bool:
        """Return True if the prompt is asking for regex matching."""
        lower = prompt.lower()
        return "regex" in lower or "pattern" in lower or "match" in lower

    def execute(self, prompt: str) -> ExecutionResult:
        """Attempt basic regex operations described in the prompt."""
        start = time.perf_counter()
        elapsed = (time.perf_counter() - start) * 1000
        return ExecutionResult(
            response="Regex tool requires LLM for complex pattern generation.",
            model_used="deterministic:regex",
            cost=0.0,
            latency_ms=round(elapsed, 3),
            confidence=0.3,
        )


class DeterministicExecutor:
    """Orchestrate deterministic tools — try each in priority order.

    If a tool can handle the task, it is used.  Otherwise returns None
    and the pipeline falls through to LLM execution.
    """

    def __init__(self) -> None:
        self._counter = CharCounterTool()
        self._calculator = CalculatorTool()
        self._json_parser = JsonParserTool()
        self._regex = RegexTool()

    def try_execute(self, prompt: str, tool_hint: str | None = None) -> ExecutionResult | None:
        """Attempt deterministic execution.

        Parameters
        ----------
        prompt : str
            Raw user prompt.
        tool_hint : str | None
            Optional hint from the decision engine (e.g. ``"deterministic:calculator"``).

        Returns
        -------
        ExecutionResult | None
            Result if a tool handled it, else ``None``.
        """
        # Exact character/substring counting — checked first (higher precision)
        if tool_hint == "deterministic:counter" or self._counter.can_handle(prompt):
            result = self._counter.execute(prompt)
            if result.confidence > 0.5:
                return result

        if tool_hint == "deterministic:calculator" or self._calculator.can_handle(prompt):
            result = self._calculator.execute(prompt)
            if result.confidence > 0.5:
                return result

        if tool_hint == "deterministic:json" or self._json_parser.can_handle(prompt):
            result = self._json_parser.execute(prompt)
            if result.confidence > 0.5:
                return result

        return None
