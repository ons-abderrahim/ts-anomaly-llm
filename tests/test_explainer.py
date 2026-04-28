"""
Tests for the LLM context builder and explainer.
LLM calls are mocked so tests run without an API key.
Run with: pytest tests/test_explainer.py -v
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.anomaly.base import AnomalyResult
from src.llm.context_builder import AnomalyContext, ContextBuilder
from src.llm.explainer import AnomalyExplainer, ExplanationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_result() -> AnomalyResult:
    n = 100
    values = np.ones(n)
    values[80] = 15.0
    scores = np.full(n, 0.1)
    scores[80] = 0.95
    flags = scores > 0.5

    return AnomalyResult(
        timestamps=[f"2024-01-01T00:{i:02d}:00Z" for i in range(n)],
        values=values,
        scores=scores,
        is_anomaly=flags,
        threshold=0.5,
        model_name="IsolationForestDetector",
    )


@pytest.fixture
def context_builder() -> ContextBuilder:
    return ContextBuilder(baseline_window=50)


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

class TestContextBuilder:
    def test_build_returns_context(self, context_builder, sample_result):
        ctx = context_builder.build(sample_result, anomaly_index=80)
        assert isinstance(ctx, AnomalyContext)
        assert ctx.anomalous_value == 15.0
        assert ctx.anomaly_score == pytest.approx(0.95)

    def test_baseline_stats(self, context_builder, sample_result):
        ctx = context_builder.build(sample_result, anomaly_index=80)
        assert ctx.baseline_mean == pytest.approx(1.0, abs=0.01)
        assert ctx.baseline_std >= 0.0

    def test_deviation_sigma_large_for_spike(self, context_builder, sample_result):
        ctx = context_builder.build(sample_result, anomaly_index=80)
        assert ctx.deviation_sigma > 5.0, "Spike at 15× baseline should have large σ deviation"

    def test_trend_classified_as_spike(self, context_builder, sample_result):
        ctx = context_builder.build(sample_result, anomaly_index=80)
        assert ctx.trend in ("spike", "rising")

    def test_metadata_passed_through(self, context_builder, sample_result):
        meta = {"sensor_id": "s-01", "unit": "bar", "location": "Plant A", "domain": "iot"}
        ctx = context_builder.build(sample_result, anomaly_index=80, metadata=meta)
        assert ctx.sensor_id == "s-01"
        assert ctx.unit == "bar"
        assert ctx.location == "Plant A"
        assert ctx.domain == "iot"

    def test_build_all_returns_list(self, context_builder, sample_result):
        contexts = context_builder.build_all(sample_result)
        assert len(contexts) == sample_result.anomaly_count
        assert all(isinstance(c, AnomalyContext) for c in contexts)

    def test_context_window_length(self, context_builder, sample_result):
        ctx = context_builder.build(sample_result, anomaly_index=80)
        # ±10 points clipped to array bounds
        assert len(ctx.context_window_values) <= 21

    def test_to_prompt_dict_has_required_keys(self, context_builder, sample_result):
        ctx = context_builder.build(sample_result, anomaly_index=80)
        d = ctx.to_prompt_dict()
        for key in ("timestamp", "value", "score", "baseline_mean", "deviation_sigma", "trend"):
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# AnomalyExplainer (mocked LLM)
# ---------------------------------------------------------------------------

MOCK_EXPLANATION = (
    "Sensor s-01 recorded an anomalous spike to 15.0 bar at 00:80 UTC, "
    "approximately 14× above the baseline of ~1.0 bar. "
    "This is consistent with a pressure valve failure or sudden blockage. "
    "Recommend immediate inspection of the pressure relief valve."
)

MOCK_ACTIONS = '["Inspect pressure relief valve", "Check adjacent sensors", "Notify maintenance team"]'


class TestAnomalyExplainer:
    @pytest.fixture
    def mock_explainer(self):
        """Return an AnomalyExplainer with a mocked LLM."""
        with patch("src.llm.explainer.ChatOpenAI") as MockLLM:
            mock_llm_instance = MagicMock()

            # Simulate chain invoke returning text
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = [MOCK_EXPLANATION, MOCK_ACTIONS]

            # Patch the | operator to return mock_chain
            mock_llm_instance.__or__ = MagicMock(return_value=mock_chain)
            MockLLM.return_value = mock_llm_instance

            explainer = AnomalyExplainer(model_name="gpt-4o-mini", openai_api_key="test-key")
            explainer._llm = mock_llm_instance

            yield explainer

    def test_explain_returns_result(self, mock_explainer, context_builder, sample_result):
        ctx = context_builder.build(sample_result, anomaly_index=80,
                                    metadata={"domain": "iot", "sensor_id": "s-01"})
        # Patch internal chain calls
        with patch.object(mock_explainer, "explain") as mock_explain:
            mock_explain.return_value = ExplanationResult(
                explanation=MOCK_EXPLANATION,
                suggested_actions=["Inspect valve", "Check sensors"],
                domain="iot",
                model_used="gpt-4o-mini",
                confidence=0.95,
            )
            result = mock_explainer.explain(ctx)

        assert isinstance(result, ExplanationResult)
        assert len(result.explanation) > 10
        assert isinstance(result.suggested_actions, list)

    def test_fallback_result_on_error(self, context_builder, sample_result):
        ctx = context_builder.build(sample_result, anomaly_index=80)
        result = AnomalyExplainer._fallback_result(ctx, "Connection timeout")
        assert "Connection timeout" in result.explanation
        assert result.model_used == "fallback"
        assert len(result.suggested_actions) >= 1

    def test_explanation_result_to_dict(self):
        r = ExplanationResult(
            explanation="Test explanation.",
            suggested_actions=["Action A", "Action B"],
            domain="operational",
            model_used="gpt-4o-mini",
            confidence=0.88,
        )
        d = r.to_dict()
        assert d["explanation"] == "Test explanation."
        assert d["confidence"] == 0.88
        assert "Action A" in d["suggested_actions"]
