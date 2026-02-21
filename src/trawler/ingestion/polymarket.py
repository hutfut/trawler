from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from trawler.db import get_conn

console = Console()

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

PAGE_SIZE = 100
PRICE_HISTORY_FIDELITY = 60  # minutes


def _fetch_closed_events_page(
    client: httpx.Client,
    order: str,
    ascending: bool,
    limit: int,
    label: str,
) -> list[dict]:
    """Paginate through closed events with a given ordering."""
    events: list[dict] = []
    offset = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Fetching {label}…", total=limit)

        while len(events) < limit:
            batch_limit = min(PAGE_SIZE, limit - len(events))
            resp = client.get(
                f"{GAMMA_BASE}/events",
                params={
                    "closed": "true",
                    "order": order,
                    "ascending": str(ascending).lower(),
                    "limit": batch_limit,
                    "offset": offset,
                },
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break

            events.extend(batch)
            offset += len(batch)
            progress.update(task, completed=len(events))

    return events[:limit]


def _fetch_closed_events(client: httpx.Client, limit: int) -> list[dict]:
    """Fetch a diverse pool of closed events using multiple orderings.

    Pulls high-volume events for coverage, plus low-liquidity events to
    catch niche/weird markets that scoring might rank highly on absurdity.
    """
    half = max(limit // 2, 1)

    by_volume = _fetch_closed_events_page(
        client, order="volume", ascending=False, limit=half,
        label="top volume events",
    )
    by_niche = _fetch_closed_events_page(
        client, order="liquidity", ascending=True, limit=half,
        label="niche/low-liquidity events",
    )

    seen_ids: set[str] = set()
    merged: list[dict] = []
    for event in by_volume + by_niche:
        eid = str(event.get("id", ""))
        if eid not in seen_ids:
            seen_ids.add(eid)
            merged.append(event)

    console.print(
        f"[dim]Merged {len(by_volume)} by volume + {len(by_niche)} niche "
        f"→ {len(merged)} unique events[/dim]"
    )
    return merged[:limit]


def _fetch_price_history(
    client: httpx.Client, asset_id: str
) -> list[dict]:
    """Fetch full price history for a single market asset."""
    resp = client.get(
        f"{CLOB_BASE}/prices-history",
        params={
            "market": asset_id,
            "interval": "max",
            "fidelity": PRICE_HISTORY_FIDELITY,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("history", [])


def _upsert_event(conn, event: dict) -> None:
    conn.execute(
        """
        INSERT INTO events (id, slug, title, description, tags, start_date, end_date)
        VALUES (%(id)s, %(slug)s, %(title)s, %(description)s, %(tags)s, %(start_date)s, %(end_date)s)
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            tags = EXCLUDED.tags
        """,
        {
            "id": str(event.get("id", "")),
            "slug": event.get("slug", ""),
            "title": event.get("title", "Untitled"),
            "description": event.get("description", ""),
            "tags": json.dumps(event.get("tags", [])),
            "start_date": event.get("startDate"),
            "end_date": event.get("endDate"),
        },
    )


def _parse_json_field(value, default=None):
    """Parse a field that may be a JSON string or already a Python object."""
    if default is None:
        default = []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def _derive_resolution(outcomes: list, outcome_prices: list) -> str:
    """Derive the winning outcome from outcomePrices.

    Polymarket sets the winning outcome's price to "1" (or 1.0) and losers to
    "0" on resolution. Returns the winning outcome string, or "" if unresolved.
    """
    if not outcomes or not outcome_prices or len(outcomes) != len(outcome_prices):
        return ""
    try:
        prices = [float(p) for p in outcome_prices]
    except (ValueError, TypeError):
        return ""
    max_price = max(prices)
    if max_price < 0.95:
        return ""
    winner_idx = prices.index(max_price)
    return outcomes[winner_idx]


def _upsert_market(conn, market: dict, event_id: str) -> None:
    outcomes = _parse_json_field(market.get("outcomes", []))
    outcome_prices = _parse_json_field(market.get("outcomePrices", []))
    clobtokens = _parse_json_field(market.get("clobTokenIds", []))

    resolution = _derive_resolution(outcomes, outcome_prices)

    conn.execute(
        """
        INSERT INTO markets (id, event_id, question, outcomes, outcome_prices,
                             volume, volume_num, liquidity, closed_time, resolution, asset_ids)
        VALUES (%(id)s, %(event_id)s, %(question)s, %(outcomes)s, %(outcome_prices)s,
                %(volume)s, %(volume_num)s, %(liquidity)s, %(closed_time)s, %(resolution)s, %(asset_ids)s)
        ON CONFLICT (id) DO UPDATE SET
            outcome_prices = EXCLUDED.outcome_prices,
            volume = EXCLUDED.volume,
            resolution = EXCLUDED.resolution
        """,
        {
            "id": str(market.get("id", "")),
            "event_id": event_id,
            "question": market.get("question", market.get("title", "Untitled")),
            "outcomes": json.dumps(outcomes),
            "outcome_prices": json.dumps(outcome_prices),
            "volume": float(market.get("volume", 0) or 0),
            "volume_num": float(market.get("volumeNum", 0) or 0),
            "liquidity": float(market.get("liquidity", 0) or 0),
            "closed_time": market.get("closedTime") or market.get("endDate"),
            "resolution": resolution,
            "asset_ids": json.dumps(clobtokens),
        },
    )


def _market_has_price_history(conn, market_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM price_history WHERE market_id = %s LIMIT 1",
        (market_id,),
    ).fetchone()
    return row is not None


def _insert_price_history(conn, market_id: str, history: list[dict]) -> int:
    if not history:
        return 0
    rows = [
        (market_id, datetime.fromtimestamp(pt["t"], tz=timezone.utc), pt["p"])
        for pt in history
        if "t" in pt and "p" in pt
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO price_history (market_id, ts, price)
            VALUES (%s, %s, %s)
            ON CONFLICT (market_id, ts) DO NOTHING
            """,
            rows,
        )
    return len(rows)


def run_ingest(limit: int = 500) -> None:
    console.print(f"[bold]Ingesting up to {limit} resolved events from Polymarket…[/bold]")

    with httpx.Client(timeout=30) as client:
        events = _fetch_closed_events(client, limit=limit)

    console.print(f"Fetched [cyan]{len(events)}[/cyan] closed events.")

    # Collect all markets from events
    all_markets: list[tuple[str, dict]] = []
    for event in events:
        event_id = str(event.get("id", ""))
        for market in event.get("markets", []):
            all_markets.append((event_id, market))

    console.print(f"Found [cyan]{len(all_markets)}[/cyan] markets across those events.")

    # Upsert events and markets
    with get_conn() as conn:
        for event in events:
            _upsert_event(conn, event)
        for event_id, market in all_markets:
            _upsert_market(conn, market, event_id)
        conn.commit()

    console.print("[green]Events and markets saved.[/green]")

    # Fetch price history for each market (idempotent — skip if already fetched)
    with get_conn() as conn, httpx.Client(timeout=30) as client:
        markets_needing_history = []
        for event_id, market in all_markets:
            market_id = str(market.get("id", ""))
            if not _market_has_price_history(conn, market_id):
                markets_needing_history.append(market)

        console.print(
            f"Fetching price history for [cyan]{len(markets_needing_history)}[/cyan] markets "
            f"([dim]{len(all_markets) - len(markets_needing_history)} already cached[/dim])…"
        )

        fetched = 0
        errors = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Price history…", total=len(markets_needing_history))

            for market in markets_needing_history:
                market_id = str(market.get("id", ""))
                clobtokens = market.get("clobTokenIds", [])
                if isinstance(clobtokens, str):
                    try:
                        clobtokens = json.loads(clobtokens)
                    except json.JSONDecodeError:
                        clobtokens = []

                # Fetch history for the first outcome token (the "Yes" side)
                asset_id = clobtokens[0] if clobtokens else None
                if not asset_id:
                    progress.update(task, advance=1)
                    continue

                try:
                    history = _fetch_price_history(client, asset_id)
                    count = _insert_price_history(conn, market_id, history)
                    if count > 0:
                        fetched += 1
                    conn.commit()
                except httpx.HTTPStatusError as exc:
                    errors += 1
                    if errors <= 3:
                        console.print(
                            f"[yellow]Warning: {exc.response.status_code} for market {market_id}[/yellow]"
                        )
                except Exception as exc:
                    errors += 1
                    if errors <= 3:
                        console.print(f"[yellow]Warning: {exc} for market {market_id}[/yellow]")

                progress.update(task, advance=1)

                # Be polite to the API
                time.sleep(0.1)

    console.print(
        f"[green]Done.[/green] Price history fetched for [cyan]{fetched}[/cyan] markets"
        f" ([yellow]{errors} errors[/yellow])."
    )


def run_backfill_resolutions() -> None:
    """Backfill resolution for existing markets using their stored outcomePrices."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, outcomes, outcome_prices FROM markets WHERE resolution = '' OR resolution IS NULL"
        ).fetchall()

        if not rows:
            console.print("[dim]All markets already have resolutions.[/dim]")
            return

        console.print(f"Backfilling resolution for [cyan]{len(rows)}[/cyan] markets…")
        updated = 0
        for row in rows:
            outcomes = row["outcomes"]
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            prices = row["outcome_prices"]
            if isinstance(prices, str):
                prices = json.loads(prices)

            resolution = _derive_resolution(outcomes, prices)
            if resolution:
                conn.execute(
                    "UPDATE markets SET resolution = %s WHERE id = %s",
                    (resolution, row["id"]),
                )
                updated += 1

        conn.commit()
        console.print(
            f"[green]Done.[/green] Updated [cyan]{updated}[/cyan] markets "
            f"([dim]{len(rows) - updated} could not be resolved[/dim])."
        )
