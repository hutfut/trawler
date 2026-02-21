from __future__ import annotations

import json
import re
from datetime import date

import anthropic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from trawler.config import get_config
from trawler.db import get_conn

console = Console()

DOMAIN_THEMES: dict[str, str] = {
    "Politics": "political predictions and power plays",
    "Pop Culture": "celebrity bets and entertainment predictions",
    "Sports": "athletic upsets and sports predictions nobody saw coming",
    "Tech/Business": "tech predictions and business bets",
    "Wildcard": "the absolute wildest things people bet real money on",
}

MIN_DOMAIN_SIZE = 3

COMPILATION_SYSTEM_PROMPT = """\
You write short, punchy narration scripts for TikTok/YouTube Shorts videos \
about prediction markets. This video's theme is: {domain_theme}.

All markets in this batch belong to the same category, so the video should \
feel cohesive — the audience clicked for {domain_theme}, keep them engaged \
all the way through.

CRITICAL FRAMING RULES:
- The story is the REAL-WORLD EVENT, not the bet. The prediction market is \
the framing device, not the subject. Lead with what happened in the world, \
then reveal what the market predicted and how much money was on the line.
- For deadline-style bets, the interesting angle is that people bet early at \
long odds — NOT that the odds converged to 100% near the deadline (that's \
obvious and boring).
- Use the market description and context provided to ground your narration in \
real-world details. Don't just talk about odds and money — talk about the \
actual event, the drama, the controversy, the human story.

TEMPORAL AWARENESS:
- Today's date is {today}. You will be given the date each market resolved.
- Frame events with appropriate temporal distance. Something that resolved a \
year ago should sound like a retrospective, not breaking news.
- Never make past events sound like they just happened.

TONE: conversational, slightly incredulous, like you're telling a friend \
something unbelievable. Not overly formal, not cringe. Think "Daily Dose of \
Internet" energy — calm but hooked.

STRUCTURE RULES:
- Each segment is 10-20 seconds when read aloud (~30-60 words).
- Start each segment with a hook about the real-world event, NOT about the bet.
- The intro MUST be unique and specific — reference the most attention-grabbing \
market in the batch. NEVER use generic openers like "people bet millions on \
wild things."
- Never give financial advice or encourage betting.
- Do NOT use hashtags, emojis, or platform-specific jargon in the script."""

COMPILATION_USER_PROMPT = """\
Generate a compilation narration script for a short-form video. The video will \
cover {count} resolved prediction markets.

Here are the markets:

{markets_block}

Write:
1. A 1-sentence intro hook — specific to the most compelling market in this \
batch, NOT a generic opener
2. One narration segment per market (30-60 words each) — lead with the \
real-world story, then the market angle
3. A 1-sentence outro/call-to-action (follow for more, etc.)

Return your response as JSON with this structure:
{{
  "intro": "...",
  "segments": [
    {{"market_id": "...", "narration": "..."}},
    ...
  ],
  "outro": "..."
}}"""


def _format_market_block(market: dict, history: list[dict], score: dict) -> str:
    outcomes = market.get("outcomes", [])
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)

    prices = [h["price"] for h in history]
    price_summary = ""
    if prices:
        price_min = min(prices)
        price_max = max(prices)
        price_start = prices[0]
        price_end = prices[-1]
        price_summary = (
            f"Odds range: {price_start:.0%} → {price_end:.0%} "
            f"(low: {price_min:.0%}, high: {price_max:.0%})"
        )

    volume_str = f"${market.get('volume', 0):,.0f}" if market.get("volume") else "unknown"

    description = market.get("description", "") or ""
    if len(description) > 300:
        description = description[:300] + "…"

    lines = [
        f"Market ID: {market['id']}",
        f"Event: {market.get('event_title', '')}",
        f"Question: {market['question']}",
        f"Outcomes: {', '.join(outcomes)}",
        f"Resolution: {market.get('resolution', 'unknown')}",
        f"Resolved on: {market.get('closed_time', 'unknown')}",
        f"Volume: {volume_str}",
    ]
    if description:
        lines.append(f"Context: {description}")
    if price_summary:
        lines.append(price_summary)

    return "\n".join(lines)


_STRIP_WORDS = re.compile(
    r"\b(?:will|the|a|an|be|in|on|by|before|after|from|to|of|for|and|or"
    r"|january|february|march|april|may|june|july|august|september"
    r"|october|november|december"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
)


def _normalize_question(q: str) -> str:
    """Strip numbers, dates, months, filler words, and punctuation to cluster near-duplicates."""
    q = q.lower()
    q = re.sub(r"\d[\d,.\-/]*", "", q)
    q = _STRIP_WORDS.sub("", q)
    q = re.sub(r"[^a-z\s]", "", q)
    return re.sub(r"\s+", " ", q).strip()


def _semantic_dedup(markets: list[dict]) -> list[dict]:
    """Keep only the highest-composite market per normalized question cluster."""
    clusters: dict[str, dict] = {}
    for m in markets:
        key = _normalize_question(m["question"])
        if key not in clusters or m.get("composite", 0) > clusters[key].get("composite", 0):
            clusters[key] = m
    return sorted(clusters.values(), key=lambda m: m.get("composite", 0), reverse=True)


def _load_top_markets_by_domain(
    conn, per_domain: int,
) -> dict[str, list[dict]]:
    """Load top-scored markets grouped by domain, one per event, deduped."""
    pool_size = per_domain * 5
    rows = conn.execute(
        """
        SELECT * FROM (
            SELECT DISTINCT ON (m.event_id)
                   m.*, e.title AS event_title,
                   s.surprise, s.narrative_arc, s.absurdity,
                   s.volume_score, s.significance, s.shareability,
                   s.humor, s.relatability, s.controversy,
                   s.wtf_factor, s.domain, s.composite
            FROM markets m
            JOIN events e ON m.event_id = e.id
            JOIN scores s ON m.id = s.market_id
            WHERE m.volume >= 500
              AND s.domain IS NOT NULL
              AND (s.narrative_arc > 0.02 OR s.surprise > 0.5
                   OR s.absurdity > 0.3 OR s.shareability > 0.5
                   OR s.humor > 0.5 OR s.wtf_factor > 0.5)
            ORDER BY m.event_id, s.composite DESC
        ) deduped
        ORDER BY composite DESC
        LIMIT %s
        """,
        (pool_size,),
    ).fetchall()

    by_domain: dict[str, list[dict]] = {}
    for row in rows:
        domain = row.get("domain") or "Wildcard"
        by_domain.setdefault(domain, []).append(row)

    for domain in list(by_domain):
        by_domain[domain] = _semantic_dedup(by_domain[domain])[:per_domain]

    wildcard = by_domain.pop("Wildcard", [])
    for domain in list(by_domain):
        if len(by_domain[domain]) < MIN_DOMAIN_SIZE:
            wildcard.extend(by_domain.pop(domain))

    if wildcard:
        wildcard = _semantic_dedup(wildcard)[:per_domain]
        by_domain["Wildcard"] = wildcard

    return by_domain


def _load_price_history(conn, market_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT ts, price FROM price_history WHERE market_id = %s ORDER BY ts",
        (market_id,),
    ).fetchall()
    return [{"ts": str(r["ts"]), "price": r["price"]} for r in rows]


def _already_scripted(conn, market_ids: list[str]) -> bool:
    """Check if a script already exists covering exactly these market IDs."""
    ids_json = json.dumps(sorted(market_ids))
    row = conn.execute(
        """
        SELECT 1 FROM scripts
        WHERE market_ids::jsonb = %s::jsonb
        LIMIT 1
        """,
        (ids_json,),
    ).fetchone()
    return row is not None


def _generate_compilation_script(
    markets_with_context: list[tuple[dict, list[dict], dict]],
    domain: str,
) -> dict:
    """Call the LLM to produce a themed compilation script."""
    cfg = get_config()
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    domain_theme = DOMAIN_THEMES.get(domain, DOMAIN_THEMES["Wildcard"])

    markets_block = "\n\n---\n\n".join(
        _format_market_block(m, h, s) for m, h, s in markets_with_context
    )

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        system=COMPILATION_SYSTEM_PROMPT.format(
            today=date.today().isoformat(),
            domain_theme=domain_theme,
        ),
        messages=[
            {
                "role": "user",
                "content": COMPILATION_USER_PROMPT.format(
                    count=len(markets_with_context),
                    markets_block=markets_block,
                ),
            }
        ],
    )

    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    return json.loads(text)


def run_generation(
    top_n: int = 20,
    group_size: int = 4,
    domain_filter: str | None = None,
) -> None:
    cfg = get_config()
    if not cfg.anthropic_api_key:
        console.print(
            "[red]ANTHROPIC_API_KEY is required for script generation.[/red]"
        )
        raise SystemExit(1)

    with get_conn() as conn:
        by_domain = _load_top_markets_by_domain(conn, per_domain=top_n)

        if domain_filter:
            by_domain = {
                k: v for k, v in by_domain.items()
                if k.lower() == domain_filter.lower()
            }

        if not by_domain:
            console.print("[dim]No scored markets found. Run 'trawler score' first.[/dim]")
            return

        total_markets = sum(len(v) for v in by_domain.values())
        console.print(
            f"[cyan]{total_markets}[/cyan] markets across "
            f"[cyan]{len(by_domain)}[/cyan] domains: "
            + ", ".join(f"{d} ({len(ms)})" for d, ms in by_domain.items())
        )

        # Build (domain, group) pairs for all domains
        all_groups: list[tuple[str, list[dict]]] = []
        for domain, markets in by_domain.items():
            for i in range(0, len(markets), group_size):
                all_groups.append((domain, markets[i : i + group_size]))

        generated = 0
        skipped = 0
        used_ids: set[str] = set()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating…", total=len(all_groups))

            for domain, group in all_groups:
                # Cross-script dedup: drop markets already used
                group = [m for m in group if m["id"] not in used_ids]
                if not group:
                    progress.update(task, advance=1)
                    continue

                market_ids = [m["id"] for m in group]

                if _already_scripted(conn, market_ids):
                    skipped += 1
                    progress.update(task, advance=1)
                    continue

                markets_with_context = []
                for m in group:
                    history = _load_price_history(conn, m["id"])
                    score_dict = {
                        "surprise": m.get("surprise", 0),
                        "narrative_arc": m.get("narrative_arc", 0),
                    }
                    markets_with_context.append((m, history, score_dict))

                try:
                    result = _generate_compilation_script(
                        markets_with_context, domain,
                    )

                    script_lines = [f"INTRO: {result.get('intro', '')}"]
                    for seg in result.get("segments", []):
                        script_lines.append(f"\nSEGMENT ({seg.get('market_id', '?')}):")
                        script_lines.append(seg.get("narration", ""))
                    script_lines.append(f"\nOUTRO: {result.get('outro', '')}")
                    script_text = "\n".join(script_lines)

                    conn.execute(
                        """
                        INSERT INTO scripts (market_ids, format, domain, script_text)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (json.dumps(sorted(market_ids)), "compilation",
                         domain, script_text),
                    )
                    conn.commit()
                    generated += 1
                    used_ids.update(market_ids)

                except Exception as exc:
                    console.print(
                        f"[yellow]Warning: {domain} script generation failed: {exc}[/yellow]"
                    )

                progress.update(task, advance=1)

    console.print(
        f"[green]Done.[/green] Generated [cyan]{generated}[/cyan] scripts"
        f" ([dim]{skipped} skipped — already exist[/dim])."
    )
