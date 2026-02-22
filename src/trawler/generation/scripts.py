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
MIN_DOMAIN_COMPOSITE = 0.55

STANDARDIZED_OUTRO = "Follow for more."

COMPILATION_SYSTEM_PROMPT = """\
You write narration scripts for short-form video (TikTok / YouTube Shorts). \
The channel covers {domain_theme}. The audience is scrolling at 10pm after \
dinner — you have ONE SECOND before they swipe. Every sentence must earn \
the next.

VOICE:
- Full meme-page energy. You are showing a friend something wild on your phone.
- React to the absurdity. Be opinionated. Say what you actually think.
- "People bet THREE HUNDRED THOUSAND DOLLARS on whether she would throw a \
punch on reality TV" — NOT "A prediction market was created for a reality \
television altercation."
- If the audience would not know who someone is, explain in one clause: \
"Ansem — a crypto influencer turned amateur boxer —"

USING ODDS DATA:
Each market is labeled either "Swing market" or "Deadline bet."
- **Swing markets** had dramatic odds movement. Use the odds journey as the \
hook: "The market opened at 12% and climbed to 94%" shows what the crowd \
believed and when they changed their minds.
- **Deadline bets** resolved because a deadline passed — the odds just drifted \
to the obvious conclusion. Do NOT narrate the odds for these. Instead, the \
angle is: why did this bet exist? What does it say about culture that people \
wagered real money on this? The absurdity of the bet's existence IS the story.
- If a market says "Odds data: Not available," ignore odds entirely. Focus on \
the story, the people, and the cultural context.

DOLLAR AMOUNTS — HARD LIMIT:
- At most ONE segment per script may open with a dollar figure as the first \
sentence. The rest MUST open differently.
- NEVER follow a dollar amount with "That's what people bet on..." or \
"That's what people wagered on..." — this pattern is BANNED. If you use a \
number, land the context immediately in the same sentence: \
"Eighty million dollars on whether Eleven dies in Stranger Things" — NOT \
"Eighty million dollars. That's what people bet on a character's survival."
- If the volume is under $1M, do NOT lead with it. Lead with the story.
- If the dollar amount is unremarkable for the topic, skip it entirely. \
"$43K on Sam Altman TIME Person of the Year" adds nothing.

CONTEXT:
- Use the market description, event title, and context provided. If there was \
a court case, a rivalry, a controversy, competing candidates — work it in. \
Do not narrate in a vacuum.
- If a person is involved, mention who else was in contention or what the \
real-world stakes were beyond the bet.
- If the outcome is common knowledge (everyone knows who won the election), \
the angle is the MARKET DYNAMICS — early odds, crowd psychology, when the \
money moved — NOT the outcome itself.

TEMPORAL AWARENESS:
- Today is {today}. Frame past events with appropriate distance.

STRUCTURAL VARIETY:
- Each segment MUST use a different opening structure. Examples: a rhetorical \
question, a short declarative fragment, a temporal hook ("Back in 2024..."), \
a number-first hook ("Sixteen million dollars..."), a person-first hook \
("Jake Paul — the YouTuber everyone loves to hate —"). If two consecutive \
segments open the same way, rewrite one.

HARD RULES:
- Each segment: 50-80 words. This is NOT a suggestion — count them. If a \
segment exceeds 80 words, cut it down. Brevity is impact.
- NO intro. Jump straight into the first market.
- NO outro. (We add one separately.)
- Every segment MUST end on a quotable line — an observation, a zinger, an \
absurd comparison, or a callback. NEVER end a segment with "the market \
resolved," "the bet died," "nothing happened," or any variant of "and then \
it was over." The last sentence is the screenshot moment.
- If your angle is "nothing happened" or "the outcome was obvious," you've \
chosen the wrong angle. Every market is here because something about it is \
wild — find that angle or the segment fails.
- NEVER be congratulatory to winners or admonishing to losers. No "bettors \
lost big" or "the smart money won." Focus on the story, not scorekeeping.
- NEVER repeat phrasing across segments. Vary structure and vocabulary.
- NEVER say "the market resolved Yes/No." Describe what happened in the world.
- No financial advice, no hashtags, no emojis."""

COMPILATION_USER_PROMPT = """\
Write narration segments for a short-form video covering {count} resolved \
prediction markets. Jump straight into the first market — no intro, no setup.

Markets:

{markets_block}

Return ONLY a JSON object:
{{
  "segments": [
    {{"market_id": "...", "narration": "..."}},
    ...
  ]
}}"""


def _format_market_block(market: dict, history: list[dict], score: dict) -> str:
    outcomes = market.get("outcomes", [])
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)

    prices = [h["price"] for h in history]
    narrative_arc = score.get("narrative_arc", 0) or 0

    if prices:
        price_min = min(prices)
        price_max = max(prices)
        price_start = prices[0]
        price_end = prices[-1]
        price_summary = (
            f"Odds range: {price_start:.0%} → {price_end:.0%} "
            f"(low: {price_min:.0%}, high: {price_max:.0%})"
        )
        if narrative_arc >= 0.1:
            bet_type = "Swing market (use the odds journey)"
        else:
            bet_type = "Deadline bet (odds uninstructive — focus on the story)"
    else:
        price_summary = "Odds data: Not available — focus on the story and cultural context."
        bet_type = "Deadline bet (odds uninstructive — focus on the story)"

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
        f"Type: {bet_type}",
    ]
    if description:
        lines.append(f"Context: {description}")
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


_ENTITY_PATTERNS = re.compile(
    r"\b("
    r"trump|biden|harris|obama|clinton|pelosi|mcconnell|desantis|pence|"
    r"vance|walz|newsom|rfk|vivek|haley|tucker carlson|bernie|aoc|"
    r"xi jinping|putin|zelenskyy|zelensky|modi|macron|trudeau|netanyahu|"
    r"starmer|milei|bolsonaro|"
    r"elon musk|zuckerberg|bezos|gates|altman|"
    r"kanye|kardashian|taylor swift|billie eilish|bad bunny|drake|"
    r"beyonce|rihanna|travis scott|doja cat|"
    r"jake paul|mike tyson|logan paul|messi|lebron|mahomes|ohtani|"
    r"tesla|spacex|openai|chatgpt|tiktok|boeing|"
    r"pope leo|pope francis|"
    r"stranger things|super bowl|world cup|"
    r"epstein|diddy|p\.? ?diddy|sean combs|tiktok ban|"
    r"mrbeast|mr\\.? ?beast|gene hackman|nuclear"
    r")\b",
    re.IGNORECASE,
)


def _extract_entities(question: str) -> set[str]:
    """Pull recognizable entity names from a market question."""
    return {m.lower() for m in _ENTITY_PATTERNS.findall(question)}


def _semantic_dedup(markets: list[dict]) -> list[dict]:
    """Keep only the highest-composite market per normalized question cluster."""
    clusters: dict[str, dict] = {}
    for m in markets:
        key = _normalize_question(m["question"])
        if key not in clusters or m.get("composite", 0) > clusters[key].get("composite", 0):
            clusters[key] = m
    return sorted(clusters.values(), key=lambda m: m.get("composite", 0), reverse=True)


def _entity_dedup(
    markets: list[dict], used_entities: set[str], max_per_entity: int = 1,
) -> list[dict]:
    """Filter markets so each named entity appears at most max_per_entity times.

    Updates used_entities in place with the entities that are selected.
    Markets are assumed to be pre-sorted by composite (highest first).
    """
    entity_counts: dict[str, int] = {}
    for ent in used_entities:
        entity_counts[ent] = max_per_entity

    result = []
    for m in markets:
        entities = _extract_entities(m["question"])
        blocked = any(entity_counts.get(e, 0) >= max_per_entity for e in entities)
        if blocked and entities:
            continue
        result.append(m)
        for e in entities:
            entity_counts[e] = entity_counts.get(e, 0) + 1
            used_entities.add(e)

    return result


def _load_top_markets_by_domain(
    conn, per_domain: int,
) -> dict[str, list[dict]]:
    """Load top-scored markets per domain, one per event, deduped.

    Uses a windowed query to fetch the top markets within each domain
    independently so no single domain monopolises the pool.
    """
    pool_per = per_domain * 3
    rows = conn.execute(
        """
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY domain ORDER BY composite DESC
            ) AS domain_rank
            FROM (
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
                  AND s.composite >= %s
                  AND m.resolution IS NOT NULL
                  AND m.resolution != ''
                  AND CASE
                      WHEN s.domain = 'Politics'
                      THEN (s.humor + s.wtf_factor) / 2 >= 0.55
                      ELSE TRUE
                  END
                ORDER BY m.event_id, s.composite DESC
            ) event_deduped
        ) ranked
        WHERE domain_rank <= %s
        ORDER BY domain, composite DESC
        """,
        (MIN_DOMAIN_COMPOSITE, pool_per),
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

        all_groups: list[tuple[str, list[dict]]] = []
        for domain, markets in by_domain.items():
            for i in range(0, len(markets), group_size):
                group = markets[i : i + group_size]
                if len(group) >= 3:
                    all_groups.append((domain, group))

        generated = 0
        skipped = 0
        used_ids: set[str] = set()
        used_entities: set[str] = set()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating…", total=len(all_groups))

            for domain, group in all_groups:
                group = [m for m in group if m["id"] not in used_ids]
                group = _entity_dedup(group, used_entities)
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

                    script_lines = []
                    for seg in result.get("segments", []):
                        script_lines.append(f"SEGMENT ({seg.get('market_id', '?')}):")
                        script_lines.append(seg.get("narration", ""))
                        script_lines.append("")
                    script_lines.append(f"OUTRO: {STANDARDIZED_OUTRO}")
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
