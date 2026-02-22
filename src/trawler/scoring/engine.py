from __future__ import annotations

import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from trawler.config import get_config
from trawler.db import get_conn
from trawler.scoring.signals import (
    surprise_score,
    narrative_arc_score,
    absurdity_score_fallback,
    volume_score,
    significance_score_fallback,
)
from trawler.scoring.llm_signals import (
    score_markets_batch,
    LLM_DIMENSIONS,
    llm_available,
)

console = Console()

MIN_VOLUME = 500
LLM_BATCH_SIZE = 6


def _load_markets(conn, rescore: bool) -> list[dict]:
    if rescore:
        query = "SELECT * FROM markets"
    else:
        query = """
            SELECT m.* FROM markets m
            LEFT JOIN scores s ON m.id = s.market_id
            WHERE s.market_id IS NULL
        """
    return conn.execute(query).fetchall()


def _load_price_history(conn, market_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT ts, price FROM price_history WHERE market_id = %s ORDER BY ts",
        (market_id,),
    ).fetchall()
    return [{"ts": r["ts"], "price": r["price"]} for r in rows]


def _load_all_volumes(conn) -> list[float]:
    rows = conn.execute("SELECT volume FROM markets WHERE volume > 0").fetchall()
    return [r["volume"] for r in rows]


def _upsert_score(
    conn, market_id: str, components: dict, composite: float, domain: str,
) -> None:
    conn.execute(
        """
        INSERT INTO scores (market_id, surprise, narrative_arc, absurdity,
                            volume_score, volume_surprise, significance,
                            shareability, humor, relatability, controversy,
                            wtf_factor, domain, composite)
        VALUES (%(market_id)s, %(surprise)s, %(narrative_arc)s, %(absurdity)s,
                %(volume_score)s, %(volume_surprise)s, %(significance)s,
                %(shareability)s, %(humor)s, %(relatability)s, %(controversy)s,
                %(wtf_factor)s, %(domain)s, %(composite)s)
        ON CONFLICT (market_id) DO UPDATE SET
            surprise = EXCLUDED.surprise,
            narrative_arc = EXCLUDED.narrative_arc,
            absurdity = EXCLUDED.absurdity,
            volume_score = EXCLUDED.volume_score,
            volume_surprise = EXCLUDED.volume_surprise,
            significance = EXCLUDED.significance,
            shareability = EXCLUDED.shareability,
            humor = EXCLUDED.humor,
            relatability = EXCLUDED.relatability,
            controversy = EXCLUDED.controversy,
            wtf_factor = EXCLUDED.wtf_factor,
            domain = EXCLUDED.domain,
            composite = EXCLUDED.composite,
            scored_at = now()
        """,
        {
            "market_id": market_id,
            "surprise": components["surprise"],
            "narrative_arc": components["narrative_arc"],
            "absurdity": components["absurdity"],
            "volume_score": components["volume_score"],
            "volume_surprise": components["volume_surprise"],
            "significance": components["significance"],
            "shareability": components["shareability"],
            "humor": components["humor"],
            "relatability": components["relatability"],
            "controversy": components["controversy"],
            "wtf_factor": components["wtf_factor"],
            "domain": domain,
            "composite": composite,
        },
    )


_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "Politics": [
        "trump", "biden", "election", "congress", "senate", "president",
        "government", "shutdown", "impeach", "pardon", "war", "nato",
        "zelenskyy", "putin", "vote", "ballot", "democrat", "republican",
        "governor", "mayor", "legislation", "supreme court", "sanctions",
    ],
    "Sports": [
        "nfl", "nba", "mlb", "nhl", "super bowl", "world series",
        "championship", "ufc", "boxing", "tyson", "fight", "playoff",
        "touchdown", "goal", "soccer", "football", "baseball", "basketball",
        "tennis", "olympics", "athlete", "espn", "match",
    ],
    "Pop Culture": [
        "celebrity", "movie", "film", "grammy", "oscar", "emmy", "album",
        "song", "tiktok", "instagram", "kanye", "kardashian", "beyonce",
        "taylor swift", "billie eilish", "stranger things", "netflix",
        "disney", "marvel", "anime", "youtube", "influencer", "viral",
        "award", "halftime", "super bowl halftime", "bad bunny", "drake",
    ],
    "Tech/Business": [
        "tesla", "spacex", "apple", "google", "amazon", "microsoft", "ai",
        "stock", "ipo", "startup", "elon musk", "self-driving", "fsd",
        "openai", "chatgpt", "valuation", "boeing", "nvidia", "meta",
    ],
}


def _heuristic_domain(question: str) -> str:
    """Best-guess domain from keywords when LLM is unavailable."""
    q = question.lower()
    best_domain = "Wildcard"
    best_hits = 0
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in q)
        if hits > best_hits:
            best_hits = hits
            best_domain = domain
    return best_domain


def _heuristic_llm_scores(question: str) -> dict[str, float | str]:
    """Fallback scores for all 7 LLM dimensions + domain using keyword heuristics."""
    absurdity = absurdity_score_fallback(question)
    significance = significance_score_fallback(question)
    return {
        "absurdity": absurdity,
        "significance": significance,
        "shareability": (absurdity + significance) / 2,
        "humor": absurdity * 0.8,
        "relatability": 0.3,
        "controversy": significance * 0.6,
        "wtf_factor": absurdity * 0.9,
        "domain": _heuristic_domain(question),
    }


def _compute_composite(weights, components: dict) -> float:
    llm_part = (
        weights.absurdity * components["absurdity"]
        + weights.significance * components["significance"]
        + weights.shareability * components["shareability"]
        + weights.humor * components["humor"]
        + weights.relatability * components["relatability"]
        + weights.controversy * components["controversy"]
        + weights.wtf_factor * components["wtf_factor"]
    )
    vol_part = (
        weights.volume * components["volume_score"]
        + weights.volume_surprise * components["volume_surprise"]
    )

    if components["narrative_arc"] < 0.05:
        # Deadline bet: redistribute surprise + narrative_arc weight to LLM dims
        llm_base_weight = (
            weights.absurdity + weights.significance + weights.shareability
            + weights.humor + weights.relatability + weights.controversy
            + weights.wtf_factor
        )
        if llm_base_weight > 0:
            scale = 1 + (weights.surprise + weights.narrative_arc) / llm_base_weight
        else:
            scale = 1.0
        return vol_part + llm_part * scale

    return (
        weights.surprise * components["surprise"]
        + weights.narrative_arc * components["narrative_arc"]
        + vol_part + llm_part
    )


def run_scoring(rescore: bool = False) -> None:
    cfg = get_config()
    weights = cfg.scoring_weights
    use_llm = llm_available()

    if use_llm:
        console.print(
            f"[green]LLM scoring enabled[/green] "
            f"(Haiku · batches of {LLM_BATCH_SIZE} · 7 dimensions)."
        )
    else:
        console.print(
            "[yellow]LLM scoring disabled[/yellow] — using heuristic fallbacks. "
            "Set ANTHROPIC_API_KEY for better scoring."
        )

    with get_conn() as conn:
        markets = _load_markets(conn, rescore)
        if not markets:
            console.print("[dim]No unscored markets found.[/dim]")
            return

        pre_filter = len(markets)
        markets = [m for m in markets if (m.get("volume") or 0) >= MIN_VOLUME]
        if pre_filter != len(markets):
            console.print(
                f"[dim]Skipped {pre_filter - len(markets)} markets below "
                f"${MIN_VOLUME} volume[/dim]"
            )

        all_volumes = _load_all_volumes(conn)

        console.print(f"Scoring [cyan]{len(markets)}[/cyan] markets…")

        # Pre-compute math-derived signals for all markets
        math_signals: dict[str, dict] = {}
        for market in markets:
            mid = market["id"]
            resolution = market.get("resolution", "")

            outcomes = market.get("outcomes", [])
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)

            outcome_prices = market.get("outcome_prices", [])
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)
            outcome_prices = [float(p) for p in outcome_prices] if outcome_prices else []

            history = _load_price_history(conn, mid)

            math_signals[mid] = {
                "surprise": surprise_score(resolution, outcome_prices, outcomes, history),
                "narrative_arc": narrative_arc_score(history),
                "volume_score": volume_score(market.get("volume", 0) or 0, all_volumes),
            }

        # LLM scoring in batches
        batches = [markets[i:i + LLM_BATCH_SIZE] for i in range(0, len(markets), LLM_BATCH_SIZE)]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scoring…", total=len(markets))

            for batch in batches:
                llm_scores: dict[str, dict] = {}

                if use_llm:
                    batch_input = [
                        {
                            "id": m["id"],
                            "question": m["question"],
                            "resolution": m.get("resolution", ""),
                            "volume": m.get("volume", 0) or 0,
                        }
                        for m in batch
                    ]
                    try:
                        llm_scores = score_markets_batch(batch_input)
                    except Exception as exc:
                        console.print(
                            f"[yellow]Batch LLM failed ({len(batch)} markets): "
                            f"{exc} — using heuristics[/yellow]"
                        )

                for market in batch:
                    mid = market["id"]
                    question = market["question"]

                    if mid in llm_scores:
                        llm = llm_scores[mid]
                    else:
                        llm = _heuristic_llm_scores(question)

                    domain = str(llm.get("domain", "Wildcard"))

                    llm_dims = {dim: llm.get(dim, 0.5) for dim in LLM_DIMENSIONS}
                    vol_s = math_signals[mid]["volume_score"]
                    weirdness = (
                        llm_dims.get("absurdity", 0.5)
                        + llm_dims.get("wtf_factor", 0.5)
                        + llm_dims.get("humor", 0.5)
                    ) / 3
                    components = {
                        **math_signals[mid],
                        "volume_surprise": vol_s * weirdness,
                        **llm_dims,
                    }

                    composite = _compute_composite(weights, components)
                    _upsert_score(conn, mid, components, composite, domain)

                conn.commit()
                progress.update(task, advance=len(batch))

    console.print("[green]Scoring complete.[/green]")
