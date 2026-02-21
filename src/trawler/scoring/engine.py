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
    score_market_llm,
    llm_available,
)

console = Console()


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


def _upsert_score(conn, market_id: str, components: dict, composite: float) -> None:
    conn.execute(
        """
        INSERT INTO scores (market_id, surprise, narrative_arc, absurdity,
                            volume_score, significance, topical, composite)
        VALUES (%(market_id)s, %(surprise)s, %(narrative_arc)s, %(absurdity)s,
                %(volume_score)s, %(significance)s, %(topical)s, %(composite)s)
        ON CONFLICT (market_id) DO UPDATE SET
            surprise = EXCLUDED.surprise,
            narrative_arc = EXCLUDED.narrative_arc,
            absurdity = EXCLUDED.absurdity,
            volume_score = EXCLUDED.volume_score,
            significance = EXCLUDED.significance,
            topical = EXCLUDED.topical,
            composite = EXCLUDED.composite,
            scored_at = now()
        """,
        {
            "market_id": market_id,
            "surprise": components["surprise"],
            "narrative_arc": components["narrative_arc"],
            "absurdity": components["absurdity"],
            "volume_score": components["volume_score"],
            "significance": components["significance"],
            "topical": components["topical"],
            "composite": composite,
        },
    )


def run_scoring(rescore: bool = False) -> None:
    cfg = get_config()
    weights = cfg.scoring_weights
    use_llm = llm_available()

    if use_llm:
        console.print("[green]LLM scoring enabled[/green] (Haiku · 1 call per market).")
    else:
        console.print(
            "[yellow]LLM scoring disabled[/yellow] — using heuristic fallbacks. "
            "Set ANTHROPIC_API_KEY for better absurdity/significance scoring."
        )

    with get_conn() as conn:
        markets = _load_markets(conn, rescore)
        if not markets:
            console.print("[dim]No unscored markets found.[/dim]")
            return

        all_volumes = _load_all_volumes(conn)

        console.print(f"Scoring [cyan]{len(markets)}[/cyan] markets…")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scoring…", total=len(markets))

            for market in markets:
                market_id = market["id"]
                question = market["question"]
                resolution = market.get("resolution", "")

                outcomes = market.get("outcomes", [])
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)

                outcome_prices = market.get("outcome_prices", [])
                if isinstance(outcome_prices, str):
                    outcome_prices = json.loads(outcome_prices)
                outcome_prices = [float(p) for p in outcome_prices] if outcome_prices else []

                history = _load_price_history(conn, market_id)

                s_surprise = surprise_score(resolution, outcome_prices, outcomes, history)
                s_narrative = narrative_arc_score(history)

                if use_llm:
                    try:
                        s_absurdity, s_significance = score_market_llm(question)
                    except Exception as exc:
                        console.print(
                            f"[yellow]LLM fallback for {market_id}: {exc}[/yellow]"
                        )
                        s_absurdity = absurdity_score_fallback(question)
                        s_significance = significance_score_fallback(question)
                else:
                    s_absurdity = absurdity_score_fallback(question)
                    s_significance = significance_score_fallback(question)

                s_volume = volume_score(market.get("volume", 0) or 0, all_volumes)
                s_topical = 0.0

                components = {
                    "surprise": s_surprise,
                    "narrative_arc": s_narrative,
                    "absurdity": s_absurdity,
                    "volume_score": s_volume,
                    "significance": s_significance,
                    "topical": s_topical,
                }

                composite = (
                    weights.surprise * s_surprise
                    + weights.narrative_arc * s_narrative
                    + weights.absurdity * s_absurdity
                    + weights.volume * s_volume
                    + weights.significance * s_significance
                    + weights.topical * s_topical
                )

                _upsert_score(conn, market_id, components, composite)
                conn.commit()
                progress.update(task, advance=1)

    console.print("[green]Scoring complete.[/green]")
