"""
LLM Explanation layer.
Takes an AnomalyContext and returns a plain-English explanation + suggested actions.
Uses LangChain with OpenAI (or any compatible LLM) under the hood.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableSequence
from langchain.schema.output_parser import StrOutputParser

from .context_builder import AnomalyContext
from .prompts import build_explanation_prompt, build_suggestions_prompt

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Output from the LLM explainer."""
    explanation: str
    suggested_actions: list[str]
    domain: str
    model_used: str
    confidence: Optional[float] = None   # derived from anomaly score if available

    def to_dict(self) -> dict:
        return {
            "explanation": self.explanation,
            "suggested_actions": self.suggested_actions,
            "domain": self.domain,
            "model_used": self.model_used,
            "confidence": self.confidence,
        }


class AnomalyExplainer:
    """
    LangChain-powered LLM layer that converts AnomalyContext → natural-language explanation.

    Args:
        model_name:     OpenAI model to use (default: "gpt-4o-mini").
        temperature:    LLM temperature (lower = more deterministic).
        baseline_window: Context window size shown in prompts.
        openai_api_key: API key; falls back to OPENAI_API_KEY env var.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        baseline_window: int = 50,
        openai_api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.baseline_window = baseline_window

        llm_kwargs = {"model_name": model_name, "temperature": temperature}
        if openai_api_key:
            llm_kwargs["openai_api_key"] = openai_api_key

        self._llm = ChatOpenAI(**llm_kwargs)
        self._str_parser = StrOutputParser()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain(self, context: AnomalyContext) -> ExplanationResult:
        """
        Generate a plain-English explanation for one anomalous event.

        Args:
            context: Populated AnomalyContext from ContextBuilder.

        Returns:
            ExplanationResult with explanation text and suggested actions.
        """
        domain = context.domain or "general"

        # --- Step 1: Generate explanation ---
        explanation_prompt = build_explanation_prompt(domain)
        explanation_chain: RunnableSequence = explanation_prompt | self._llm | self._str_parser

        prompt_vars = context.to_prompt_dict()
        prompt_vars["baseline_window"] = self.baseline_window

        explanation = explanation_chain.invoke(prompt_vars)
        logger.debug("Generated explanation (%d chars)", len(explanation))

        # --- Step 2: Extract structured suggested actions ---
        suggested_actions = self._extract_actions(explanation)

        # --- Step 3: Derive confidence from anomaly score ---
        confidence = round(context.anomaly_score, 3) if context.anomaly_score else None

        return ExplanationResult(
            explanation=explanation.strip(),
            suggested_actions=suggested_actions,
            domain=domain,
            model_used=self.model_name,
            confidence=confidence,
        )

    def explain_batch(self, contexts: list[AnomalyContext]) -> list[ExplanationResult]:
        """
        Explain a list of anomaly contexts.  Runs sequentially to avoid rate limits.

        Args:
            contexts: List of AnomalyContext objects.

        Returns:
            List of ExplanationResult objects (same order as input).
        """
        results = []
        for i, ctx in enumerate(contexts):
            logger.info("Explaining anomaly %d / %d", i + 1, len(contexts))
            try:
                results.append(self.explain(ctx))
            except Exception as exc:
                logger.error("Failed to explain anomaly at %s: %s", ctx.timestamp, exc)
                results.append(self._fallback_result(ctx, str(exc)))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_actions(self, explanation: str) -> list[str]:
        """Use a follow-up LLM call to extract structured action items."""
        suggestions_prompt = build_suggestions_prompt()
        chain: RunnableSequence = suggestions_prompt | self._llm | self._str_parser

        try:
            raw = chain.invoke({"explanation": explanation})
            raw = raw.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            actions = json.loads(raw)
            if isinstance(actions, list):
                return [str(a) for a in actions[:4]]
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Could not parse action list: %s", exc)
        return []

    @staticmethod
    def _fallback_result(context: AnomalyContext, error: str) -> ExplanationResult:
        return ExplanationResult(
            explanation=(
                f"Anomaly detected at {context.timestamp} "
                f"(value={context.anomalous_value}, score={context.anomaly_score:.2f}). "
                f"Automatic explanation failed: {error}. Please investigate manually."
            ),
            suggested_actions=["Investigate manually", "Check sensor logs"],
            domain=context.domain or "general",
            model_used="fallback",
            confidence=context.anomaly_score,
        )
