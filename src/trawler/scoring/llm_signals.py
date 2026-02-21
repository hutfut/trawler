"""
LLM-powered scoring signals (7 dimensions, batched).

Falls back to heuristics if no API key is configured.
"""

from __future__ import annotations

import json

import anthropic

from trawler.config import get_config

LLM_DIMENSIONS = [
    "absurdity", "significance", "shareability",
    "humor", "relatability", "controversy", "wtf_factor",
]

_BATCH_PROMPT = """\
Rate each resolved prediction market below on 7 dimensions (1-10 each).

1. ABSURDITY: Would someone scrolling TikTok stop and think "wait, people \
actually bet on that?" Consider the question AND the outcome.
2. SIGNIFICANCE: Does this relate to a major real-world event (elections, \
wars, economic shifts, breakthroughs, major cultural moments)?
3. SHAREABILITY: Would someone send this to a friend or post it unprompted? \
The best markets make people say "you have to see this."
4. HUMOR: Is this genuinely funny — not just surprising, but laugh-out-loud \
or absurdly comedic?
5. RELATABILITY: Does this touch something a broad audience cares about? \
Niche crypto or sports spreads score low; pop culture, everyday life score high.
6. CONTROVERSY: Does this topic spark debate or strong opinions? Markets \
where people would argue in the comments score high.
7. WTF_FACTOR: Pure "I can't believe this exists" energy. The market's \
mere existence is the story.

Respond with ONLY a JSON array — one object per market, in order:
[{{"id": 1, "absurdity": <int>, "significance": <int>, "shareability": <int>, \
"humor": <int>, "relatability": <int>, "controversy": <int>, "wtf_factor": <int>}}, ...]

Markets:
{markets_block}"""


def _build_markets_block(markets: list[dict]) -> str:
    lines = []
    for i, m in enumerate(markets, 1):
        vol = m.get("volume", 0)
        lines.append(
            f"[{i}] \"{m['question']}\" | "
            f"Resolved: {m.get('resolution') or 'Unknown'} | "
            f"Volume: ${vol:,.0f}"
        )
    return "\n".join(lines)


def _parse_batch_response(text: str, count: int) -> list[dict]:
    """Extract a JSON array from the LLM response, handling markdown fences."""
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    results = json.loads(text)
    if not isinstance(results, list) or len(results) != count:
        raise ValueError(
            f"Expected {count} results, got {len(results) if isinstance(results, list) else 'non-list'}"
        )
    return results


def score_markets_batch(markets: list[dict]) -> dict[str, dict]:
    """Score multiple markets in a single LLM call.

    Accepts a list of dicts with keys: id, question, resolution, volume.
    Returns a mapping of market_id -> {dimension: normalized_score} for each
    of the 7 LLM dimensions (values in [0, 1]).
    """
    if not markets:
        return {}

    cfg = get_config()
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200 * len(markets),
        messages=[
            {"role": "user", "content": _BATCH_PROMPT.format(
                markets_block=_build_markets_block(markets),
            )},
        ],
    )

    raw = _parse_batch_response(resp.content[0].text.strip(), len(markets))

    scores: dict[str, dict] = {}
    for i, entry in enumerate(raw):
        market_id = markets[i]["id"]
        scores[market_id] = {
            dim: int(entry.get(dim, 5)) / 10.0
            for dim in LLM_DIMENSIONS
        }

    return scores


def score_market_llm(
    question: str,
    resolution: str = "",
    volume: float = 0,
    market_id: str = "",
) -> dict[str, float]:
    """Score a single market via the batch function (thin wrapper).

    Returns a dict of {dimension: normalized_score} for 7 LLM dimensions.
    """
    result = score_markets_batch([{
        "id": market_id or "single",
        "question": question,
        "resolution": resolution,
        "volume": volume,
    }])
    return result.get(market_id or "single", {dim: 0.5 for dim in LLM_DIMENSIONS})


def llm_available() -> bool:
    cfg = get_config()
    return bool(cfg.anthropic_api_key)
