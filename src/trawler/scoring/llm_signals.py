"""
LLM-powered scoring signals (absurdity, historical significance).

Falls back to heuristics if no API key is configured.
"""

from __future__ import annotations

import json

import anthropic

from trawler.config import get_config

_COMBINED_PROMPT = """\
You are rating a prediction market question on two dimensions.

1. ABSURDITY (1-10): How attention-grabbing is this on social media? A high \
score means someone scrolling TikTok would stop and think "wait, people \
actually bet on that?"

2. SIGNIFICANCE (1-10): Does this relate to a major real-world event \
(elections, wars, economic shifts, scientific breakthroughs, major cultural \
moments)? High scores mean broad historical or cultural weight.

Respond with ONLY a JSON object:
{{"absurdity": <int>, "significance": <int>}}

Market question: {question}"""


def score_market_llm(question: str) -> tuple[float, float]:
    """Score a market for absurdity and significance in a single LLM call.

    Returns (absurdity_normalized, significance_normalized) both in [0, 1].
    """
    cfg = get_config()
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=80,
        messages=[
            {"role": "user", "content": _COMBINED_PROMPT.format(question=question)},
        ],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    # Find the JSON object if there's preamble text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    result = json.loads(text)
    absurdity = int(result.get("absurdity", 5)) / 10.0
    significance = int(result.get("significance", 5)) / 10.0
    return absurdity, significance


def llm_available() -> bool:
    cfg = get_config()
    return bool(cfg.anthropic_api_key)
