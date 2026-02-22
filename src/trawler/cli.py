from __future__ import annotations

import typer
from rich.console import Console

from trawler.db import init_db

app = typer.Typer(help="Trawler — retrospective prediction market content pipeline")
console = Console()


@app.command()
def init():
    """Initialize the database schema."""
    init_db()
    console.print("[green]Database initialized.[/green]")


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Drop all tables and reinitialize. Use for clean iteration loops."""
    from trawler.db import reset_db

    if not yes:
        typer.confirm("This will delete ALL data. Continue?", abort=True)

    reset_db()
    console.print("[green]Database reset — all tables dropped and recreated.[/green]")


@app.command()
def ingest(
    limit: int = typer.Option(500, help="Max number of resolved events to fetch"),
):
    """Pull resolved markets and price history from Polymarket."""
    from trawler.ingestion.polymarket import run_ingest

    init_db()
    run_ingest(limit=limit)


@app.command()
def backfill():
    """Backfill resolution data for existing markets."""
    from trawler.ingestion.polymarket import run_backfill_resolutions

    run_backfill_resolutions()


@app.command(name="backfill-history")
def backfill_history():
    """Fetch price history for all markets missing it in the DB."""
    from trawler.ingestion.polymarket import run_backfill_history

    run_backfill_history()


@app.command()
def score(
    rescore: bool = typer.Option(False, help="Re-score already scored markets"),
):
    """Compute virality scores for ingested markets."""
    from trawler.scoring.engine import run_scoring

    run_scoring(rescore=rescore)


@app.command()
def generate(
    top: int = typer.Option(20, help="Top markets per domain to consider"),
    group_size: int = typer.Option(4, help="Markets per compilation script"),
    domain: str = typer.Option("", help="Generate for a single domain only (e.g. 'Politics')"),
):
    """Generate themed narration scripts grouped by domain."""
    from trawler.generation.scripts import run_generation

    run_generation(
        top_n=top, group_size=group_size,
        domain_filter=domain or None,
    )


@app.command()
def review(
    limit: int = typer.Option(10, help="Number of scripts to show"),
    export: bool = typer.Option(False, help="Export to export/review-YYYY-MM-DD-HH-MM.md"),
    domain: str = typer.Option("", help="Filter to a single domain (e.g. 'Politics')"),
):
    """Display generated scripts for human review."""
    from datetime import datetime
    from pathlib import Path

    from trawler.generation.review import run_review

    export_path = ""
    if export:
        Path("export").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
        export_path = f"export/review-{ts}.md"

    run_review(limit=limit, export_path=export_path, domain_filter=domain)


if __name__ == "__main__":
    app()
