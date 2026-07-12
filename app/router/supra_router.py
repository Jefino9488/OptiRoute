"""Supra-Router-51M — ML-based prompt classification and routing.

Uses a small 51M-parameter model running on a dedicated llama-server instance
(port 8081) to classify prompts and decide small vs big model routing.

Input format:  Task: [Prompt]\nAnalysis:
Output format: Domain: X | Complexity: N | Math: True/False | Code: True/False |
               Route: small model/big model | Justification: [reasoning]

Replaces the regex-based FeatureExtractor with learned routing decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SupraRouterResult:
    """Structured output from Supra-Router-51M."""

    domain: str
    complexity: float  # normalized 0..1 (model outputs 2-5, we rescale)
    needs_math: bool
    needs_code: bool
    route: str  # "small model" | "big model"
    justification: str
    raw_output: str
    success: bool = True


# ---------------------------------------------------------------------------
# Supra-Router client
# ---------------------------------------------------------------------------

class SupraRouter:
    """Client for Supra-Router-51M running on a dedicated llama-server.

    The router model is tiny (~37MB quantized, 3840 context) and runs on
    a separate llama-server instance (default port 8081). Inference is
    sub-millisecond with greedy decoding for deterministic output.
    """

    def __init__(self, server_url: str = "http://localhost:8081") -> None:
        self._server_url = server_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        """Shut down the HTTP client."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Core classification
    # ------------------------------------------------------------------

    async def classify(self, prompt: str) -> SupraRouterResult:
        """Classify a prompt using Supra-Router-51M.

        Parameters
        ----------
        prompt : str
            The user prompt to classify.

        Returns
        -------
        SupraRouterResult
            Structured routing decision with domain, complexity, and
            small/big model recommendation.
        """
        # Supra-Router input format
        router_input = f"Task: {prompt}\nAnalysis: "

        messages = [{"role": "user", "content": router_input}]

        try:
            response = await self._client.post(
                f"{self._server_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 256,
                    "do_sample": False,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning("supra_router.classify_failed", error=str(exc))
            return SupraRouterResult(
                domain="",
                complexity=0.5,
                needs_math=False,
                needs_code=False,
                route="",
                justification="",
                raw_output="",
                success=False,
            )

        choices = data.get("choices", [])
        raw_output: str = (
            choices[0].get("message", {}).get("content", "") if choices else ""
        )

        logger.info(
            "supra_router.classify_complete",
            raw_output=raw_output[:200],
        )

        return self._parse_output(raw_output)

    async def classify_batch(self, prompts: list[str]) -> list[SupraRouterResult]:
        """Classify multiple prompts in parallel.

        Parameters
        ----------
        prompts : list[str]
            List of prompts to classify.

        Returns
        -------
        list[SupraRouterResult]
            Results in the same order as input prompts.
        """
        import asyncio
        return await asyncio.gather(*[self.classify(p) for p in prompts])

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_output(raw: str) -> SupraRouterResult:
        """Parse pipe-separated Supra-Router output into structured result.

        Expected format:
            Domain: Semantic Field | Complexity: 0.75 | Math: True | Code: False |
            Route: small model | Justification: [reasoning]
        """
        result = SupraRouterResult(
            domain="",
            complexity=0.5,
            needs_math=False,
            needs_code=False,
            route="",
            justification="",
            raw_output=raw,
        )

        if not raw.strip():
            result.success = False
            return result

        # Split by pipe separator
        parts = [p.strip() for p in raw.split("|")]

        # First part may be a bare domain label (no "Domain:" prefix).
        # Detect: if the first part has no colon, treat it as the domain.
        if parts and ":" not in parts[0]:
            result.domain = parts[0]
            parts = parts[1:]

        for part in parts:
            if ":" not in part:
                continue

            key, _, value = part.partition(":")
            key = key.strip().lower()
            value = value.strip()

            if key == "domain":
                result.domain = value
            elif key == "complexity":
                try:
                    raw_complexity = float(value)
                    # Supra-Router outputs 2-5; normalize to 0-1
                    # Use a wider range so 2→0.2, 3→0.5, 5→1.0
                    result.complexity = max(0.0, min(1.0, (raw_complexity - 1.0) / 4.0))
                except (ValueError, TypeError):
                    result.complexity = 0.5
            elif key == "math":
                result.needs_math = value.lower() in ("true", "1", "yes")
            elif key == "code":
                result.needs_code = value.lower() in ("true", "1", "yes")
            elif key == "route":
                result.route = value.lower()
            elif key == "justification":
                result.justification = value

        return result
