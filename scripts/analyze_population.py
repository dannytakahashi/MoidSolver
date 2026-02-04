#!/usr/bin/env python3
"""Analyze population statistics from hand history database."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from moid.db import get_connection
from moid.analysis import PopulationAnalyzer, compute_stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze population statistics from hand database"
    )
    parser.add_argument(
        "-d", "--database",
        default="hands.db",
        help="Database file path (default: hands.db)",
    )
    parser.add_argument(
        "-p", "--position",
        choices=["UTG", "UTG1", "CO", "BTN", "SB", "BB"],
        help="Filter by position",
    )
    parser.add_argument(
        "--min-stack",
        type=float,
        help="Minimum stack size in BBs",
    )
    parser.add_argument(
        "--max-stack",
        type=float,
        help="Maximum stack size in BBs",
    )
    parser.add_argument(
        "--exploits",
        action="store_true",
        help="Show exploitation recommendations",
    )
    parser.add_argument(
        "--by-position",
        action="store_true",
        help="Show breakdown by position",
    )
    parser.add_argument(
        "--by-stack",
        action="store_true",
        help="Show breakdown by stack depth",
    )

    args = parser.parse_args()
    console = Console()

    # Open database
    db_path = Path(args.database)
    if not db_path.exists():
        console.print(f"[red]Database not found: {db_path}[/]")
        console.print("Run import_hands.py first to create the database.")
        return 1

    conn = get_connection(db_path)

    # Compute stats
    console.print(f"[bold]Analyzing database:[/] {db_path}")
    console.print()

    if args.position or args.min_stack or args.max_stack:
        # Filtered stats
        stats = compute_stats(
            conn,
            position=args.position,
            min_stack=args.min_stack,
            max_stack=args.max_stack,
        )
        _display_stats(console, stats, title="Filtered Statistics")
    else:
        # Full population analysis
        analyzer = PopulationAnalyzer(conn)
        pop_stats = analyzer.analyze()

        _display_stats(console, pop_stats.overall, title="Overall Population Statistics")

        if args.by_position:
            console.print()
            _display_position_breakdown(console, pop_stats.by_position)

        if args.by_stack:
            console.print()
            _display_stack_breakdown(console, pop_stats)

        if args.exploits:
            console.print()
            _display_exploits(console, analyzer)

    conn.close()
    return 0


def _display_stats(console: Console, stats, title: str) -> None:
    """Display player stats in a formatted table."""
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Notes", style="dim")

    # Sample size
    table.add_row("Hands", f"{stats.hands:,}", "sample size")

    # Preflop
    table.add_row("", "", "")
    table.add_row("[bold]Preflop[/]", "", "")
    table.add_row("VPIP", f"{stats.vpip:.1f}%", _get_vpip_note(stats.vpip))
    table.add_row("PFR", f"{stats.pfr:.1f}%", _get_pfr_note(stats.pfr))
    table.add_row("3-Bet", f"{stats.three_bet:.1f}%", _get_3bet_note(stats.three_bet))
    table.add_row("Fold to 3-Bet", f"{stats.fold_to_3bet:.1f}%", "")

    # Postflop
    table.add_row("", "", "")
    table.add_row("[bold]Postflop[/]", "", "")
    table.add_row("C-Bet", f"{stats.cbet:.1f}%", "")
    table.add_row("Fold to C-Bet", f"{stats.fold_to_cbet:.1f}%", _get_fold_cbet_note(stats.fold_to_cbet))

    # Aggression
    table.add_row("", "", "")
    table.add_row("[bold]Aggression[/]", "", "")
    table.add_row("AF", f"{stats.af:.2f}", _get_af_note(stats.af))
    table.add_row("AFq", f"{stats.afq:.1f}%", "")

    # Showdown
    table.add_row("", "", "")
    table.add_row("[bold]Showdown[/]", "", "")
    table.add_row("WTSD", f"{stats.wtsd:.1f}%", _get_wtsd_note(stats.wtsd))
    table.add_row("W$SD", f"{stats.wsd:.1f}%", "")

    console.print(table)


def _display_position_breakdown(console: Console, by_position: dict) -> None:
    """Display stats breakdown by position."""
    table = Table(title="Statistics by Position", show_header=True, header_style="bold")

    table.add_column("Position", style="cyan")
    table.add_column("Hands", justify="right")
    table.add_column("VPIP", justify="right")
    table.add_column("PFR", justify="right")
    table.add_column("3-Bet", justify="right")
    table.add_column("C-Bet", justify="right")
    table.add_column("AF", justify="right")

    for pos in ["UTG", "UTG1", "CO", "BTN", "SB", "BB"]:
        if pos in by_position:
            stats = by_position[pos]
            table.add_row(
                pos,
                f"{stats.hands:,}",
                f"{stats.vpip:.1f}%",
                f"{stats.pfr:.1f}%",
                f"{stats.three_bet:.1f}%",
                f"{stats.cbet:.1f}%",
                f"{stats.af:.2f}",
            )

    console.print(table)


def _display_stack_breakdown(console: Console, pop_stats) -> None:
    """Display stats breakdown by stack depth."""
    table = Table(title="Statistics by Stack Depth", show_header=True, header_style="bold")

    table.add_column("Stack", style="cyan")
    table.add_column("Hands", justify="right")
    table.add_column("VPIP", justify="right")
    table.add_column("PFR", justify="right")
    table.add_column("AF", justify="right")

    stacks = [
        ("Short (<50bb)", pop_stats.short_stack),
        ("Medium (50-100bb)", pop_stats.medium_stack),
        ("Deep (>100bb)", pop_stats.deep_stack),
    ]

    for name, stats in stacks:
        table.add_row(
            name,
            f"{stats.hands:,}",
            f"{stats.vpip:.1f}%",
            f"{stats.pfr:.1f}%",
            f"{stats.af:.2f}",
        )

    console.print(table)


def _display_exploits(console: Console, analyzer: PopulationAnalyzer) -> None:
    """Display exploitation recommendations."""
    exploits = analyzer.get_exploits()

    if not exploits:
        console.print("[yellow]No significant exploits identified.[/]")
        return

    panel_content = "\n".join(f"• {e}" for e in exploits)
    panel = Panel(
        panel_content,
        title="[bold]Exploitation Recommendations[/]",
        border_style="green",
    )
    console.print(panel)


def _get_vpip_note(vpip: float) -> str:
    if vpip > 40:
        return "[red]very loose[/]"
    elif vpip > 30:
        return "[yellow]loose[/]"
    elif vpip < 18:
        return "[blue]tight[/]"
    return "normal"


def _get_pfr_note(pfr: float) -> str:
    if pfr < 10:
        return "[red]passive[/]"
    elif pfr > 25:
        return "[yellow]aggressive[/]"
    return "normal"


def _get_3bet_note(three_bet: float) -> str:
    if three_bet < 4:
        return "[red]low (exploitable)[/]"
    elif three_bet > 10:
        return "[yellow]high[/]"
    return "normal"


def _get_fold_cbet_note(fold_cbet: float) -> str:
    if fold_cbet > 60:
        return "[red]overfolds (exploit!)[/]"
    elif fold_cbet < 40:
        return "[yellow]calls too much[/]"
    return "normal"


def _get_af_note(af: float) -> str:
    if af < 1.5:
        return "[red]passive[/]"
    elif af > 4:
        return "[yellow]aggressive[/]"
    return "normal"


def _get_wtsd_note(wtsd: float) -> str:
    if wtsd > 35:
        return "[yellow]station (don't bluff)[/]"
    elif wtsd < 22:
        return "[blue]folds too much[/]"
    return "normal"


if __name__ == "__main__":
    sys.exit(main())
