"""
Prompt templates for the LLM explanation layer.
Each domain (IoT, financial, operational) gets a specialised system prompt
and a shared structured user prompt.
"""

from __future__ import annotations

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# ---------------------------------------------------------------------------
# System prompts per domain
# ---------------------------------------------------------------------------

_SYSTEM_GENERAL = """\
You are an expert site-reliability and data analyst who explains anomalies \
detected in time series data to non-technical operations teams. \
Your explanations are concise (3–5 sentences), factual, and actionable. \
Always state: what happened, why it likely happened (root cause hypotheses), \
and what the operator should do next. Avoid jargon.
"""

_SYSTEM_IOT = """\
You are an expert IoT systems engineer and data analyst. \
You explain anomalies detected in sensor streams (temperature, pressure, vibration, \
flow rate, etc.) to plant operators and maintenance teams. \
Your explanations are concise (3–5 sentences), factual, and actionable. \
Always state: what was observed, the most likely root cause (equipment fault, \
calibration issue, environmental factor), and immediate recommended action.
"""

_SYSTEM_FINANCIAL = """\
You are a senior quantitative analyst and risk manager. \
You explain anomalies detected in financial time series (prices, volumes, spreads, \
PnL, transaction rates) to trading desk operators and compliance teams. \
Your explanations are concise (3–5 sentences), factual, and actionable. \
Highlight whether the anomaly resembles known market events (flash crash, \
fat-finger, illiquidity) and what immediate risk controls to apply.
"""

_SYSTEM_OPERATIONAL = """\
You are a DevOps and cloud infrastructure expert. \
You explain anomalies detected in operational metrics (CPU, latency, error rate, \
request throughput, queue depth) to on-call engineers. \
Your explanations are concise (3–5 sentences), factual, and actionable. \
Always suggest a runbook step or investigation path as the next action.
"""

SYSTEM_PROMPTS: dict[str, str] = {
    "iot": _SYSTEM_IOT,
    "financial": _SYSTEM_FINANCIAL,
    "operational": _SYSTEM_OPERATIONAL,
    "general": _SYSTEM_GENERAL,
}


# ---------------------------------------------------------------------------
# Human / user prompt
# ---------------------------------------------------------------------------

_HUMAN_TEMPLATE = """\
An anomaly has been detected in the following time series. Explain it clearly.

ANOMALY DETAILS:
- Sensor / Signal:     {sensor_id}
- Location:            {location}
- Unit of measurement: {unit}
- Timestamp:           {timestamp}
- Anomalous value:     {value} {unit}
- Anomaly score:       {score} (0 = normal, 1 = highly anomalous)

CONTEXT:
- Baseline mean (last {baseline_window} points): {baseline_mean} {unit}
- Baseline std dev:    {baseline_std} {unit}
- Deviation:           {deviation_sigma}σ from baseline
- Pattern:             {trend}
- Context window (±10 points): {context_window}

Please provide:
1. A plain-English description of what happened.
2. The most likely root cause(s).
3. Recommended immediate action(s) for the operator.
"""

_SUGGESTIONS_TEMPLATE = """\
Based on the explanation above, list 2–4 concrete, short suggested actions \
(each ≤ 12 words) as a JSON array of strings. \
Return ONLY the JSON array, no other text.
"""


def build_explanation_prompt(domain: str = "general") -> ChatPromptTemplate:
    """Return a ChatPromptTemplate for the main explanation chain."""
    system_content = SYSTEM_PROMPTS.get(domain, _SYSTEM_GENERAL)
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_content),
        HumanMessagePromptTemplate.from_template(_HUMAN_TEMPLATE),
    ])


def build_suggestions_prompt() -> ChatPromptTemplate:
    """Return a prompt that extracts structured action items from an explanation."""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant that extracts actionable steps from anomaly explanations."
        ),
        HumanMessagePromptTemplate.from_template(
            "Given this anomaly explanation:\n\n{explanation}\n\n" + _SUGGESTIONS_TEMPLATE
        ),
    ])
