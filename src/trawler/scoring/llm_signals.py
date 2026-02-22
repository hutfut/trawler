"""
LLM-powered scoring signals (7 dimensions, batched).

Falls back to heuristics if no API key is configured.
"""

from __future__ import annotations

import json

import anthropic

from trawler.config import get_config

VALID_DOMAINS = ["Politics", "Pop Culture", "Sports", "Tech/Business", "Wildcard"]

LLM_DIMENSIONS = [
    "absurdity", "significance", "shareability",
    "humor", "relatability", "controversy", "wtf_factor",
]

_BATCH_PROMPT = """\
For each resolved prediction market below, first classify it into exactly one \
domain, then rate it on 7 dimensions (1-10 each).

STEP 1 — CLASSIFY into one domain:
- Politics: elections, legislation, government, geopolitics, world leaders
- Pop Culture: celebrities, music, movies, TV, social media, awards
- Sports: athletic competition, fighting, esports
- Tech/Business: companies, products, crypto, finance, AI, space
- Wildcard: anything that doesn't fit, or pure WTF-energy

STEP 2 — SCORE relative to the domain. All 7 scores should reflect what's \
notable WITHIN that category, not globally. A 7/10 humor in Politics means \
genuinely funny for a political market. A 6/10 controversy in Politics means \
only moderately divisive for politics (where everything is already somewhat \
controversial). An 8/10 shareability in Tech/Business means tech people would \
definitely send it around.

Dimensions:
1. ABSURDITY: "Wait, people actually bet on that?" — relative to the domain.
2. SIGNIFICANCE: How major is this within its domain? A mid-tier election \
market is less significant than a landmark ruling, even though both are Politics.
3. SHAREABILITY: Would someone screenshot this and send it to their group chat? \
IMPORTANT: If the outcome is common knowledge (everyone knows who won the \
election), shareability is LOW. High shareability means the market reveals \
something people DON'T already know, or the bet's mere existence is the story. \
"Interesting news" is NOT shareable — "I can't believe people bet money on THIS" is.
4. HUMOR: Would a 24-year-old scrolling their phone at 10pm actually laugh? \
Not "mildly amusing" — genuinely funny. A nuclear detonation market is not funny. \
A CEO's dating life as a prediction market IS funny.
5. RELATABILITY: Does a broad audience within this domain find the MARKET \
interesting, not just the topic? A market about a well-known event where the \
outcome is obvious in hindsight scores LOW — the audience already knows what \
happened. A market that surfaces something unexpected or niche-but-fascinating \
scores HIGH.
6. CONTROVERSY: Sparks debate among people who follow this domain.
7. WTF_FACTOR: "I can't believe someone made this a BET" energy. This is about \
the absurdity of the WAGER, not the seriousness of the topic. A nuclear \
detonation market is significant, not WTF — anyone would track geopolitical \
risk. A market on whether a CEO gets a girlfriend by August IS WTF. Score the \
bet, not the headline.

Respond with ONLY a JSON array — one object per market, in order:
[{{"id": 1, "domain": "<domain>", "absurdity": <int>, "significance": <int>, \
"shareability": <int>, "humor": <int>, "relatability": <int>, \
"controversy": <int>, "wtf_factor": <int>}}, ...]

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
        dim_scores = {
            dim: int(entry.get(dim, 5)) / 10.0
            for dim in LLM_DIMENSIONS
        }
        raw_domain = str(entry.get("domain", "Wildcard"))
        dim_scores["domain"] = raw_domain if raw_domain in VALID_DOMAINS else "Wildcard"
        scores[market_id] = dim_scores

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
    fallback = {dim: 0.5 for dim in LLM_DIMENSIONS}
    fallback["domain"] = "Wildcard"
    return result.get(market_id or "single", fallback)


def llm_available() -> bool:
    cfg = get_config()
    return bool(cfg.anthropic_api_key)
