"""
Individual scoring signal functions.

Each function takes market data and returns a float in [0, 1].
"""

from __future__ import annotations

import json
import math
from typing import Sequence


def surprise_score(
    resolution: str,
    outcome_prices: list[float],
    outcomes: list[str],
    price_history: list[dict],
) -> float:
    """How unexpected was the outcome?

    Price history always tracks the FIRST outcome (usually "Yes").
    If that first outcome won, surprise = how far the tail was from 1.0.
    If it lost, surprise = how high the tail was (market expected it to win
    but it didn't).
    """
    if not price_history or not outcomes or not resolution:
        return 0.0

    tail_start = max(0, len(price_history) - max(1, len(price_history) // 5))
    tail_prices = [pt["price"] for pt in price_history[tail_start:]]
    if not tail_prices:
        return 0.0

    avg_tail = sum(tail_prices) / len(tail_prices)

    first_outcome_won = resolution.upper() == outcomes[0].upper()

    if first_outcome_won:
        # First outcome won — surprise is how low the market had it
        surprise = 1.0 - avg_tail
    else:
        # First outcome lost — surprise is how high the market had it
        surprise = avg_tail

    return _clamp(surprise)


def narrative_arc_score(price_history: list[dict]) -> float:
    """Did the odds tell an interesting story?

    Measures volatility, direction reversals, and max single-period swing.
    A flat line at 90% that resolves YES is boring.  Wild swings are interesting.
    """
    if len(price_history) < 3:
        return 0.0

    prices = [pt["price"] for pt in price_history]

    # Standard deviation of price changes
    deltas = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    if not deltas:
        return 0.0

    mean_delta = sum(deltas) / len(deltas)
    variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
    stdev = math.sqrt(variance)

    # Direction reversals (sign changes in deltas)
    reversals = sum(
        1 for i in range(len(deltas) - 1) if deltas[i] * deltas[i + 1] < 0
    )
    reversal_ratio = reversals / max(1, len(deltas) - 1)

    # Max single-period swing
    max_swing = max(abs(d) for d in deltas)

    # Overall price range
    price_range = max(prices) - min(prices)

    # Combine into a composite — each component normalized roughly to [0, 1]
    # stdev of 0.15+ is very volatile for a probability market
    stdev_norm = min(stdev / 0.15, 1.0)
    swing_norm = min(max_swing / 0.3, 1.0)
    range_norm = min(price_range / 0.8, 1.0)

    composite = (
        stdev_norm * 0.3
        + reversal_ratio * 0.25
        + swing_norm * 0.25
        + range_norm * 0.2
    )
    return _clamp(composite)


def absurdity_score_fallback(question: str) -> float:
    """Heuristic absurdity scorer used when no LLM is available.

    Looks for signals like question marks, unusual length, keywords that
    suggest novelty.  This is a rough stand-in — the LLM version is better.
    """
    q = question.lower()
    score = 0.0

    novelty_keywords = [
        "celebrity", "meme", "alien", "ufo", "bigfoot", "dye",
        "eat", "tweet", "tiktok", "viral", "bet", "dare",
        "weird", "bizarre", "strange", "crazy", "insane",
        "kanye", "elon", "florida man",
    ]
    hits = sum(1 for kw in novelty_keywords if kw in q)
    score += min(hits * 0.2, 0.6)

    if len(question) > 100:
        score += 0.15

    if "?" in question:
        score += 0.05

    return _clamp(score)


def volume_score(market_volume: float, all_volumes: Sequence[float]) -> float:
    """Sigmoid-capped volume score.

    Reaches ~0.5 at the volume midpoint and flattens above it so that
    mega-markets ($100M+) don't dominate over mid-volume ones ($1M+).
    A market with ANY meaningful volume ($100K+) gets a decent baseline.
    """
    if not all_volumes or market_volume <= 0:
        return 0.0

    log_vol = math.log1p(market_volume)
    log_all = [math.log1p(v) for v in all_volumes if v > 0]
    if not log_all:
        return 0.0

    median_log = sorted(log_all)[len(log_all) // 2]
    # Sigmoid centered on the median, scaled so ±2 stdevs span [0.1, 0.9]
    stdev = (sum((v - median_log) ** 2 for v in log_all) / len(log_all)) ** 0.5
    if stdev < 0.01:
        return 0.5
    z = (log_vol - median_log) / stdev
    sigmoid = 1.0 / (1.0 + math.exp(-z))
    return _clamp(sigmoid)


def significance_score_fallback(question: str) -> float:
    """Heuristic significance scorer used when no LLM is available.

    Looks for keywords suggesting major real-world events.
    """
    q = question.lower()
    score = 0.0

    significance_keywords = [
        "president", "election", "war", "gdp", "fed", "rate",
        "supreme court", "congress", "senate", "parliament",
        "recession", "pandemic", "earthquake", "hurricane",
        "olympics", "world cup", "nobel", "spacex", "nasa",
        "bitcoin", "crypto", "ai ", "artificial intelligence",
    ]
    hits = sum(1 for kw in significance_keywords if kw in q)
    score += min(hits * 0.25, 0.8)

    if not hits:
        score = 0.2  # baseline — most markets are at least somewhat notable

    return _clamp(score)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))
