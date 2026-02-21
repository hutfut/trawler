from __future__ import annotations

import json

import anthropic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from trawler.config import get_config
from trawler.db import get_conn

console = Console()

COMPILATION_SYSTEM_PROMPT = """\
You write short, punchy narration scripts for TikTok/YouTube Shorts videos \
about prediction markets. The channel covers resolved prediction markets — \
surprising outcomes, wild odds swings, and things people actually bet real \
money on.

Tone: conversational, slightly incredulous, like you're telling a friend \
something unbelievable. Not overly formal, not cringe. Think "Daily Dose of \
Internet" energy — calm but hooked.

Rules:
- Each segment is 10-20 seconds when read aloud (~30-60 words).
- Start each segment with a hook that makes the viewer want to keep watching.
- Reference the actual odds movements when they're dramatic.
- End each segment with the outcome — don't bury the lede, but build a beat \
  of tension first.
- Never give financial advice or encourage betting. Frame everything as \
  "look at this interesting thing that happened."
- Do NOT use hashtags, emojis, or platform-specific jargon in the script itself."""

COMPILATION_USER_PROMPT = """\
Generate a compilation narration script for a short-form video. The video will \
cover {count} resolved prediction markets.

Here are the markets:

{markets_block}

Write:
1. A 1-sentence intro hook for the overall video (something like "People bet \
real money on some wild things — here's what happened this week")
2. One narration segment per market (30-60 words each)
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

    lines = [
        f"Market ID: {market['id']}",
        f"Question: {market['question']}",
        f"Outcomes: {', '.join(outcomes)}",
        f"Resolution: {market.get('resolution', 'unknown')}",
        f"Volume: {volume_str}",
    ]
    if price_summary:
        lines.append(price_summary)
    lines.append(f"Surprise score: {score.get('surprise', 0):.2f}/1.0")
    lines.append(f"Narrative arc score: {score.get('narrative_arc', 0):.2f}/1.0")

    return "\n".join(lines)


def _load_top_markets(conn, top_n: int) -> list[dict]:
    return conn.execute(
        """
        SELECT m.*, s.surprise, s.narrative_arc, s.absurdity,
               s.volume_score, s.significance, s.composite
        FROM markets m
        JOIN scores s ON m.id = s.market_id
        ORDER BY s.composite DESC
        LIMIT %s
        """,
        (top_n,),
    ).fetchall()


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
) -> dict:
    """Call the LLM to produce a compilation script."""
    cfg = get_config()
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    markets_block = "\n\n---\n\n".join(
        _format_market_block(m, h, s) for m, h, s in markets_with_context
    )

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        system=COMPILATION_SYSTEM_PROMPT,
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
    # Handle markdown-wrapped JSON
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove opening fence
        text = text.rsplit("```", 1)[0]  # remove closing fence

    return json.loads(text)


def run_generation(top_n: int = 20, group_size: int = 4) -> None:
    cfg = get_config()
    if not cfg.anthropic_api_key:
        console.print(
            "[red]ANTHROPIC_API_KEY is required for script generation.[/red]"
        )
        raise SystemExit(1)

    with get_conn() as conn:
        markets = _load_top_markets(conn, top_n)
        if not markets:
            console.print("[dim]No scored markets found. Run 'trawler score' first.[/dim]")
            return

        console.print(
            f"Generating scripts from top [cyan]{len(markets)}[/cyan] scored markets "
            f"(groups of {group_size})…"
        )

        # Split into groups
        groups: list[list[dict]] = []
        for i in range(0, len(markets), group_size):
            groups.append(markets[i : i + group_size])

        generated = 0
        skipped = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating…", total=len(groups))

            for group in groups:
                market_ids = [m["id"] for m in group]

                if _already_scripted(conn, market_ids):
                    skipped += 1
                    progress.update(task, advance=1)
                    continue

                # Build context for each market in the group
                markets_with_context = []
                for m in group:
                    history = _load_price_history(conn, m["id"])
                    score_dict = {
                        "surprise": m.get("surprise", 0),
                        "narrative_arc": m.get("narrative_arc", 0),
                    }
                    markets_with_context.append((m, history, score_dict))

                try:
                    result = _generate_compilation_script(markets_with_context)

                    # Format the full script text for storage
                    script_lines = [f"INTRO: {result.get('intro', '')}"]
                    for seg in result.get("segments", []):
                        script_lines.append(f"\nSEGMENT ({seg.get('market_id', '?')}):")
                        script_lines.append(seg.get("narration", ""))
                    script_lines.append(f"\nOUTRO: {result.get('outro', '')}")
                    script_text = "\n".join(script_lines)

                    conn.execute(
                        """
                        INSERT INTO scripts (market_ids, format, script_text)
                        VALUES (%s, %s, %s)
                        """,
                        (json.dumps(sorted(market_ids)), "compilation", script_text),
                    )
                    conn.commit()
                    generated += 1

                except Exception as exc:
                    console.print(f"[yellow]Warning: script generation failed: {exc}[/yellow]")

                progress.update(task, advance=1)

    console.print(
        f"[green]Done.[/green] Generated [cyan]{generated}[/cyan] scripts"
        f" ([dim]{skipped} skipped — already exist[/dim])."
    )
