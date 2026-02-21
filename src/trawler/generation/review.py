from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trawler.db import get_conn

console = Console()


def _load_scripts(conn, limit: int) -> list[dict]:
    return conn.execute(
        """
        SELECT s.id, s.market_ids, s.format, s.script_text, s.created_at
        FROM scripts s
        ORDER BY s.created_at DESC
        LIMIT %s
        """,
        (limit,),
    ).fetchall()


def _load_markets_for_script(conn, market_ids: list[str]) -> list[dict]:
    if not market_ids:
        return []
    placeholders = ",".join(["%s"] * len(market_ids))
    return conn.execute(
        f"""
        SELECT m.id, m.question, m.resolution, m.volume,
               sc.composite, sc.surprise, sc.narrative_arc, sc.absurdity,
               sc.volume_score, sc.significance
        FROM markets m
        LEFT JOIN scores sc ON m.id = sc.market_id
        WHERE m.id IN ({placeholders})
        ORDER BY sc.composite DESC
        """,
        tuple(market_ids),
    ).fetchall()


def _render_script_to_console(script: dict, markets: list[dict]) -> None:
    """Render a single script with its market context to the terminal."""
    header = (
        f"Script #{script['id']}  |  "
        f"Format: {script['format']}  |  "
        f"Created: {script['created_at']}"
    )
    console.rule(f"[bold cyan]{header}[/bold cyan]")

    # Market summary table
    table = Table(title="Markets in this script", show_lines=True)
    table.add_column("Question", style="white", max_width=50)
    table.add_column("Resolution", style="green")
    table.add_column("Volume", style="yellow", justify="right")
    table.add_column("Composite", style="magenta", justify="right")
    table.add_column("Surprise", justify="right")
    table.add_column("Arc", justify="right")
    table.add_column("Absurd", justify="right")

    for m in markets:
        table.add_row(
            m["question"][:50],
            str(m.get("resolution", "")),
            f"${m.get('volume', 0):,.0f}" if m.get("volume") else "—",
            f"{m.get('composite', 0):.3f}" if m.get("composite") is not None else "—",
            f"{m.get('surprise', 0):.2f}" if m.get("surprise") is not None else "—",
            f"{m.get('narrative_arc', 0):.2f}" if m.get("narrative_arc") is not None else "—",
            f"{m.get('absurdity', 0):.2f}" if m.get("absurdity") is not None else "—",
        )

    console.print(table)
    console.print()

    # Script text
    console.print(Panel(
        script["script_text"],
        title="[bold]Narration Script[/bold]",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()


def _render_script_to_markdown(script: dict, markets: list[dict]) -> str:
    """Render a single script as a markdown section."""
    lines = [
        f"## Script #{script['id']}",
        f"**Format:** {script['format']}  ",
        f"**Created:** {script['created_at']}",
        "",
        "### Markets",
        "",
        "| Question | Resolution | Volume | Composite | Surprise | Arc | Absurd |",
        "|----------|------------|--------|-----------|----------|-----|--------|",
    ]

    for m in markets:
        q = m["question"][:50].replace("|", "\\|")
        vol = f"${m.get('volume', 0):,.0f}" if m.get("volume") else "—"
        lines.append(
            f"| {q} "
            f"| {m.get('resolution', '')} "
            f"| {vol} "
            f"| {m.get('composite', 0):.3f} "
            f"| {m.get('surprise', 0):.2f} "
            f"| {m.get('narrative_arc', 0):.2f} "
            f"| {m.get('absurdity', 0):.2f} |"
        )

    lines.extend([
        "",
        "### Script",
        "",
        "```",
        script["script_text"],
        "```",
        "",
        "---",
        "",
    ])

    return "\n".join(lines)


def run_review(limit: int = 10, export_path: str = "") -> None:
    with get_conn() as conn:
        scripts = _load_scripts(conn, limit)
        if not scripts:
            console.print("[dim]No scripts found. Run 'trawler generate' first.[/dim]")
            return

        console.print(f"Showing [cyan]{len(scripts)}[/cyan] most recent scripts.\n")

        md_parts: list[str] = []

        for script in scripts:
            market_ids = script.get("market_ids", [])
            if isinstance(market_ids, str):
                market_ids = json.loads(market_ids)

            markets = _load_markets_for_script(conn, market_ids)

            _render_script_to_console(script, markets)

            if export_path:
                md_parts.append(_render_script_to_markdown(script, markets))

    if export_path:
        md_content = "# Trawler Script Review\n\n" + "\n".join(md_parts)
        Path(export_path).write_text(md_content)
        console.print(f"[green]Exported to {export_path}[/green]")
