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
VOLUME_FLOOR = 500  # skip markets below this dollar volume

NOVELTY_TAG_IDS = [
    596,     # Culture
    286,     # Celebrities
    53,      # Movies
    100,     # Music
    315,     # Entertainment
    1401,    # Tech
    107,     # Business
    102846,  # Best of 2025
]
SPORTS_TAG_ID = 1
CRYPTO_PRICES_TAG_ID = 1312


def _fetch_closed_events_page(
    client: httpx.Client,
    limit: int,
    label: str,
    *,
    order: str = "volume",
    ascending: bool = False,
    tag_id: int | None = None,
    exclude_tag_id: int | None = None,
) -> list[dict]:
    """Paginate through closed events with a given ordering and optional tag filters."""
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
            params: dict = {
                "closed": "true",
                "order": order,
                "ascending": str(ascending).lower(),
                "limit": batch_limit,
                "offset": offset,
            }
            if tag_id is not None:
                params["tag_id"] = tag_id
            if exclude_tag_id is not None:
                params["exclude_tag_id"] = exclude_tag_id

            resp = client.get(f"{GAMMA_BASE}/events", params=params)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break

            events.extend(batch)
            offset += len(batch)
            progress.update(task, completed=len(events))

    return events[:limit]


def _dedup_events(event_lists: list[tuple[str, list[dict]]]) -> list[dict]:
    """Merge multiple event lists, deduplicating by event ID."""
    seen_ids: set[str] = set()
    merged: list[dict] = []
    bucket_counts: list[str] = []

    for label, events in event_lists:
        added = 0
        for event in events:
            eid = str(event.get("id", ""))
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                merged.append(event)
                added += 1
        bucket_counts.append(f"{added} from {label}")

    console.print(f"[dim]Merged: {', '.join(bucket_counts)} → {len(merged)} unique events[/dim]")
    return merged


def _event_has_tag(event: dict, tag_id: int) -> bool:
    tags = event.get("tags", [])
    if isinstance(tags, str):
        tags = json.loads(tags)
    return any(
        (isinstance(t, dict) and str(t.get("id", "")) == str(tag_id))
        for t in (tags or [])
    )


_TAG_LABELS = {
    596: "Culture", 286: "Celebrities", 53: "Movies", 100: "Music",
    315: "Entertainment", 1401: "Tech", 107: "Business", 102846: "Best of 2025",
}


def _fetch_closed_events(client: httpx.Client, limit: int) -> list[dict]:
    """Fetch a diverse pool of closed events using multiple strategies.

    Bucket A (25%): competitive ordering — contested markets with dramatic odds
    Bucket B (25%): high-volume, excluding sports and crypto prices
    Bucket C (50%): tag-targeted fetches for human-appeal categories
    """
    bucket_a_size = max(limit // 4, 1)
    bucket_b_size = max(limit // 4, 1)
    bucket_c_per_tag = max(limit // (2 * len(NOVELTY_TAG_IDS)), 1)

    buckets: list[tuple[str, list[dict]]] = []

    buckets.append(("competitive", _fetch_closed_events_page(
        client, limit=bucket_a_size, label="competitive events",
        order="competitive", ascending=False,
    )))

    volume_raw = _fetch_closed_events_page(
        client, limit=bucket_b_size + 50, label="high-volume non-sports",
        order="volume", ascending=False, exclude_tag_id=SPORTS_TAG_ID,
    )
    volume_filtered = [
        e for e in volume_raw
        if not _event_has_tag(e, CRYPTO_PRICES_TAG_ID)
    ][:bucket_b_size]
    buckets.append(("volume (no sports/crypto)", volume_filtered))

    for tag_id in NOVELTY_TAG_IDS:
        tag_label = _TAG_LABELS.get(tag_id, str(tag_id))
        buckets.append((tag_label, _fetch_closed_events_page(
            client, limit=bucket_c_per_tag, label=f"tag: {tag_label}",
            order="volume", ascending=False, tag_id=tag_id,
        )))

    merged = _dedup_events(buckets)
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
        INSERT INTO markets (id, event_id, question, description, outcomes, outcome_prices,
                             volume, volume_num, liquidity, closed_time, resolution, asset_ids)
        VALUES (%(id)s, %(event_id)s, %(question)s, %(description)s, %(outcomes)s, %(outcome_prices)s,
                %(volume)s, %(volume_num)s, %(liquidity)s, %(closed_time)s, %(resolution)s, %(asset_ids)s)
        ON CONFLICT (id) DO UPDATE SET
            outcome_prices = EXCLUDED.outcome_prices,
            volume = EXCLUDED.volume,
            resolution = EXCLUDED.resolution,
            description = EXCLUDED.description
        """,
        {
            "id": str(market.get("id", "")),
            "event_id": event_id,
            "question": market.get("question", market.get("title", "Untitled")),
            "description": market.get("description", ""),
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

    all_markets: list[tuple[str, dict]] = []
    skipped_low_vol = 0
    for event in events:
        event_id = str(event.get("id", ""))
        for market in event.get("markets", []):
            vol = float(market.get("volume", 0) or 0)
            if vol < VOLUME_FLOOR:
                skipped_low_vol += 1
                continue
            all_markets.append((event_id, market))

    console.print(
        f"Found [cyan]{len(all_markets)}[/cyan] markets across those events"
        f" ([dim]{skipped_low_vol} skipped below ${VOLUME_FLOOR} volume[/dim])."
    )

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
