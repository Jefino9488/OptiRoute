"""Inference Policy Compiler — generates deterministic system prompts from routing features.

This module acts as a Prompt Compiler that translates the features extracted 
by the vectorizer into a highly optimized, model-specific system prompt,
without altering the original user prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any
import hashlib
import json

@dataclass(slots=True)
class CompilerResult:
    """The result of compiling the inference policy."""
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class Rule:
    """A single inference policy rule."""
    name: str
    category: str
    priority: int
    condition: Callable[[dict[str, float], dict[str, Any], dict[str, bool], str], bool]
    instruction: str


class RuleRegistry:
    """Registry of deterministic execution rules."""

    def __init__(self):
        self._rules: list[Rule] = []

    def register(
        self,
        name: str,
        category: str,
        priority: int,
        condition: Callable[[dict[str, float], dict[str, Any], dict[str, bool], str], bool],
        instruction: str,
    ) -> None:
        """Register a new rule."""
        self._rules.append(Rule(name, category, priority, condition, instruction))

    def evaluate(
        self,
        task_vector: dict[str, float],
        resource_vector: dict[str, Any],
        risk_vector: dict[str, bool],
        model: str,
    ) -> tuple[list[str], list[str]]:
        """Evaluate all rules and return the winning instructions and rule names."""
        active_rules = [
            r for r in self._rules
            if r.condition(task_vector, resource_vector, risk_vector, model)
        ]

        # Resolve conflicts by category (highest priority wins)
        winners: dict[str, Rule] = {}
        for r in active_rules:
            if r.category not in winners or r.priority > winners[r.category].priority:
                winners[r.category] = r

        # Sort final rules by priority descending
        sorted_winners = sorted(winners.values(), key=lambda x: x.priority, reverse=True)
        
        instructions = [r.instruction for r in sorted_winners]
        names = [r.name for r in sorted_winners]
        
        return instructions, names


class InferencePolicyCompiler:
    """Compiles routing features into a definitive system prompt."""

    def __init__(self):
        self._registry = RuleRegistry()
        self._cache: dict[str, tuple[str, dict[str, Any]]] = {}
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Populate the registry with standard optimization rules."""
        
        # --- FORMAT RULES (High Priority) ---
        self._registry.register(
            name="format:json",
            category="format",
            priority=100,
            condition=lambda t, res, risk, m: risk.get("json_required", False),
            instruction="Output must be valid JSON. Return only the requested fields with no commentary."
        )
        self._registry.register(
            name="format:translation",
            category="format",
            priority=90,
            condition=lambda t, res, risk, m: t.get("translation", 0.0) > 0.6,
            instruction="Output translated text only. Preserve formatting and meaning perfectly. Do not add explanations."
        )

        # --- TASK RULES (Medium Priority) ---
        self._registry.register(
            name="task:math",
            category="task",
            priority=80,
            condition=lambda t, res, risk, m: risk.get("high_accuracy_required", False) and t.get("math", 0.0) > 0.5,
            instruction=(
                "Work step by step. Label each arithmetic step (Step 1:, Step 2:, ...). "
                "Show each intermediate result on its own line. "
                "State the final answer on the last line as: **Answer: [value]**. "
                "Do not skip steps."
            )
        )
        self._registry.register(
            name="task:math_counting",
            category="task_counting",
            priority=85,
            condition=lambda t, res, risk, m: t.get("math", 0.0) > 0.5 and risk.get("is_counting", False),
            instruction=(
                "For counting tasks: first list every occurrence you find, numbered (1. ..., 2. ...). "
                "Then state the total count as a final line: Total: N. "
                "Do not guess. Enumerate every item before totalling."
            )
        )
        self._registry.register(
            name="task:code",
            category="task",
            priority=75,
            condition=lambda t, res, risk, m: t.get("code", 0.0) > 0.6,
            instruction="Ensure code is syntactically correct and production-ready. Avoid unnecessary explanation."
        )
        self._registry.register(
            name="task:reasoning",
            category="task",
            priority=70,
            condition=lambda t, res, risk, m: t.get("reasoning", 0.0) > 0.6,
            instruction=(
                "Read all constraints carefully before answering. "
                "If the constraints are contradictory, explicitly state: CONTRADICTION DETECTED and explain which constraints conflict. "
                "Then provide the most logically consistent resolution. "
                "Use at most 4 concise bullet points for your final answer."
            )
        )
        self._registry.register(
            name="task:sentiment",
            category="task",
            priority=78,
            condition=lambda t, res, risk, m: t.get("sentiment", 0.0) > 0.5,
            instruction=(
                "Begin your response with exactly ONE word on the first line: Positive, Negative, or Neutral. "
                "Then give a single sentence explaining your reasoning. "
                "No other format is acceptable."
            )
        )
        self._registry.register(
            name="task:ner",
            category="task",
            priority=72,
            condition=lambda t, res, risk, m: t.get("ner", 0.0) > 0.5,
            instruction=(
                "Extract named entities into explicit categories. Use this exact format:\n"
                "People: [comma-separated list or none]\n"
                "Organizations: [comma-separated list or none]\n"
                "Locations: [comma-separated list or none]\n"
                "Dates: [comma-separated list or none]\n"
                "Include only entities explicitly stated in the text."
            )
        )
        self._registry.register(
            name="task:extraction",
            category="task",
            priority=60,
            condition=lambda t, res, risk, m: t.get("extraction", 0.0) > 0.6,
            instruction=(
                "Extract exactly what is requested. Pay close attention to the requested unit of information "
                "(e.g. domains, not full email addresses; years, not full dates). "
                "Return only the extracted data in the requested format. No commentary."
            )
        )
        self._registry.register(
            name="task:creative",
            category="task",
            priority=55,
            condition=lambda t, res, risk, m: t.get("creative", 0.0) > 0.6,
            instruction="Be creative but concise. Maximum 200 words."
        )
        self._registry.register(
            name="task:summarization",
            category="task",
            priority=65,
            condition=lambda t, res, risk, m: t.get("summarization", 0.0) > 0.5,
            instruction=(
                "Summarize concisely. Follow any explicit format requirements (bullet points, sentences, word limits) exactly. "
                "Do not add any information not present in the source text."
            )
        )
        self._registry.register(
            name="task:general_qa",
            category="task",
            priority=50,
            condition=lambda t, res, risk, m: t.get("general_qa", 0.0) > 0.6,
            instruction="Answer accurately and directly. Do not over-explain. Maximum 3 sentences."
        )

        # --- BUDGET RULES (Lower Priority) ---
        self._registry.register(
            name="budget:small",
            category="budget",
            priority=50,
            condition=lambda t, res, risk, m: res.get("output_budget_bucket") == "Small",
            instruction="Keep the answer highly concise. Return only the requested information with no preamble or filler."
        )
        self._registry.register(
            name="budget:large",
            category="budget",
            priority=40,
            condition=lambda t, res, risk, m: res.get("output_budget_bucket") == "Large",
            instruction="Provide a comprehensive and detailed response."
        )

        # --- MODEL SPECIFIC RULES (Lowest Priority) ---
        self._registry.register(
            name="model:kimi",
            category="model",
            priority=30,
            condition=lambda t, res, risk, m: "kimi" in m.lower(),
            instruction="Prioritize correctness over brevity. Avoid markdown unless requested."
        )
        self._registry.register(
            name="model:minimax",
            category="model",
            priority=25,
            condition=lambda t, res, risk, m: "minimax" in m.lower(),
            instruction="Keep responses concise. Avoid repeating information."
        )
        self._registry.register(
            name="model:local",
            category="model",
            priority=20,
            condition=lambda t, res, risk, m: "local" in m.lower(),
            instruction=(
                "Produce compact, direct answers. "
                "For any calculation, show each step explicitly. "
                "Do not include unnecessary preamble."
            )
        )


    def compile(
        self,
        prompt: str,
        task_vector: dict[str, float],
        resource_vector: dict[str, Any],
        risk_vector: dict[str, bool],
        model: str,
        base_system_prompt: str = "",
    ) -> CompilerResult:
        """Compile the final system prompt based on vectors.
        
        Args:
            prompt: The untouched original user prompt.
            task_vector: Task affinities.
            resource_vector: Resource estimates.
            risk_vector: Risk flags.
            model: The selected execution model.
            base_system_prompt: Optional base instructions (like preprocessor guards).
            
        Returns:
            CompilerResult containing system_prompt, user_prompt, and metadata.
        """
        # Create a deterministic cache key
        state = {
            "t": task_vector,
            "res": resource_vector,
            "risk": risk_vector,
            "m": model,
            "base": base_system_prompt,
        }
        state_json = json.dumps(state, sort_keys=True)
        cache_key = hashlib.sha256(state_json.encode("utf-8")).hexdigest()

        if cache_key in self._cache:
            # We must still return a fresh CompilerResult because user_prompt might differ
            cached_sys, cached_meta = self._cache[cache_key]
            return CompilerResult(
                system_prompt=cached_sys,
                user_prompt=prompt,
                metadata=cached_meta
            )

        instructions, rules_applied = self._registry.evaluate(
            task_vector, resource_vector, risk_vector, model
        )

        final_instructions = []
        if base_system_prompt:
            final_instructions.append(base_system_prompt.strip())
            
        final_instructions.extend(instructions)
        
        system_prompt = "\n\n".join(final_instructions).strip()
        
        # Fallback if no rules matched and no base prompt
        if not system_prompt:
            system_prompt = "Answer concisely and accurately."

        metadata = {
            "compiler": {
                "rules": rules_applied,
                "system_tokens": int(len(system_prompt.split()) * 1.3), # rough estimate
            }
        }

        # Cache the resulting system prompt and metadata
        self._cache[cache_key] = (system_prompt, metadata)

        return CompilerResult(
            system_prompt=system_prompt,
            user_prompt=prompt, # Untouched
            metadata=metadata,
        )
