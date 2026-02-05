#!/usr/bin/env python3
"""Personal poker stats dashboard.

Analyze your own play, compare to GTO benchmarks, and identify leaks.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from moid.db import get_connection
from moid.analysis.hero import HeroAnalyzer, Leak
from moid.analysis.spots import SpotAnalyzer, SpotType
from moid.analysis.flagger import HandFlagger
from moid.analysis.benchmarks import GTO_BENCHMARKS


def main():
    parser = argparse.ArgumentParser(
        description="Personal poker stats dashboard - analyze your play"
    )
    parser.add_argument(
        "-d", "--database",
        default="hands.db",
        help="Database file path (default: hands.db)",
    )
    parser.add_argument(
        "-p", "--position",
        choices=["UTG", "UTG1", "CO", "BTN", "SB", "BB"],
        help="Show stats for specific position",
    )
    parser.add_argument(
        "--leaks",
        action="store_true",
        help="Show detailed leak analysis",
    )
    parser.add_argument(
        "--spots",
        action="store_true",
        help="Show spot-by-spot analysis",
    )
    parser.add_argument(
        "--flagged",
        action="store_true",
        help="Show hands flagged for review",
    )
    parser.add_argument(
        "--flagged-limit",
        type=int,
        default=10,
        help="Number of flagged hands to show (default: 10)",
    )
    parser.add_argument(
        "--no-gto",
        action="store_true",
        help="Hide GTO comparison columns",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show additional detail",
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

    # Header
    console.print()
    console.print(Panel.fit(
        "[bold blue]Personal Poker Stats Dashboard[/]\n"
        f"Database: {db_path}",
        border_style="blue",
    ))
    console.print()

    # Initialize analyzers
    hero_analyzer = HeroAnalyzer(conn)
    hero_stats = hero_analyzer.analyze()

    # Check sample size
    if hero_stats.overall.hands < 100:
        console.print(f"[yellow]Warning: Only {hero_stats.overall.hands} hands in database.[/]")
        console.print("[dim]Stats may not be reliable with small sample size.[/]")
        console.print()

    # Display based on options
    if args.position:
        _display_position_stats(console, hero_stats, args.position, not args.no_gto)
    else:
        _display_overall_stats(console, hero_stats, not args.no_gto)
        console.print()
        _display_position_breakdown(console, hero_stats, not args.no_gto)

    # Leaks
    if args.leaks or (not args.spots and not args.flagged):
        console.print()
        _display_leaks(console, hero_stats.leaks, args.verbose)

    # Strengths
    if hero_stats.strengths and args.verbose:
        console.print()
        _display_strengths(console, hero_stats.strengths)

    # Spot analysis
    if args.spots:
        console.print()
        spot_analyzer = SpotAnalyzer(conn)
        _display_spot_analysis(console, spot_analyzer)

    # Flagged hands
    if args.flagged:
        console.print()
        flagger = HandFlagger(conn)
        _display_flagged_hands(console, flagger, args.flagged_limit)

    conn.close()
    return 0


def _display_overall_stats(
    console: Console, hero_stats, show_gto: bool
) -> None:
    """Display overall hero statistics."""
    stats = hero_stats.overall

    table = Table(
        title="Overall Statistics",
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
    )
    table.add_column("Statistic", style="cyan")
    table.add_column("Your Value", justify="right")
    if show_gto:
        table.add_column("GTO Range", justify="right", style="dim")
        table.add_column("Status", justify="center")

    # Sample size
    if show_gto:
        table.add_row("Hands", f"{stats.hands:,}", "", _sample_size_indicator(stats.hands))
    else:
        table.add_row("Hands", f"{stats.hands:,}")

    # Preflop stats
    _add_stat_row(table, "VPIP", stats.vpip, GTO_BENCHMARKS.vpip_range, show_gto)
    _add_stat_row(table, "PFR", stats.pfr, GTO_BENCHMARKS.pfr_range, show_gto)

    vpip_pfr_gap = stats.vpip - stats.pfr
    if show_gto:
        gap_status = "[green]OK[/]" if vpip_pfr_gap <= 8 else "[red]High[/]"
        table.add_row("VPIP-PFR Gap", f"{vpip_pfr_gap:.1f}", "3-6", gap_status)
    else:
        table.add_row("VPIP-PFR Gap", f"{vpip_pfr_gap:.1f}")

    _add_stat_row(table, "3-Bet", stats.three_bet, GTO_BENCHMARKS.three_bet_range, show_gto)
    _add_stat_row(table, "Fold to 3-Bet", stats.fold_to_3bet, (45.0, 55.0), show_gto)

    # Postflop stats
    _add_stat_row(table, "C-Bet", stats.cbet, GTO_BENCHMARKS.cbet_range, show_gto)
    _add_stat_row(table, "Fold to C-Bet", stats.fold_to_cbet, GTO_BENCHMARKS.fold_to_cbet_range, show_gto)

    # Aggression
    if show_gto:
        af_status = _range_status(stats.af, GTO_BENCHMARKS.af_range)
        table.add_row("AF", f"{stats.af:.2f}", f"{GTO_BENCHMARKS.af_range[0]:.1f}-{GTO_BENCHMARKS.af_range[1]:.1f}", af_status)
    else:
        table.add_row("AF", f"{stats.af:.2f}")

    table.add_row("AFq", f"{stats.afq:.1f}%")

    # Showdown
    _add_stat_row(table, "WTSD", stats.wtsd, GTO_BENCHMARKS.wtsd_range, show_gto)
    if show_gto:
        table.add_row("W$SD", f"{stats.wsd:.1f}%", "45-55", _range_status(stats.wsd, (45.0, 55.0)))
    else:
        table.add_row("W$SD", f"{stats.wsd:.1f}%")

    console.print(table)


def _add_stat_row(
    table: Table,
    name: str,
    value: float,
    gto_range: tuple[float, float],
    show_gto: bool,
) -> None:
    """Add a stat row with GTO comparison."""
    if show_gto:
        status = _range_status(value, gto_range)
        table.add_row(
            name,
            f"{value:.1f}%",
            f"{gto_range[0]:.0f}-{gto_range[1]:.0f}",
            status,
        )
    else:
        table.add_row(name, f"{value:.1f}%")


def _range_status(value: float, gto_range: tuple[float, float]) -> str:
    """Get status indicator for a value vs range."""
    low, high = gto_range
    margin = (high - low) * 0.3  # 30% margin outside range

    if low <= value <= high:
        return "[green]OK[/]"
    elif value < low - margin or value > high + margin:
        return "[red]!!![/]"
    elif value < low or value > high:
        return "[yellow]![/]"
    return "[green]OK[/]"


def _sample_size_indicator(hands: int) -> str:
    """Get sample size indicator."""
    if hands >= 5000:
        return "[green]Reliable[/]"
    elif hands >= 1000:
        return "[yellow]OK[/]"
    elif hands >= 500:
        return "[yellow]Small[/]"
    else:
        return "[red]Very Small[/]"


def _display_position_breakdown(
    console: Console, hero_stats, show_gto: bool
) -> None:
    """Display stats breakdown by position."""
    table = Table(
        title="Statistics by Position",
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
    )

    table.add_column("Position", style="cyan")
    table.add_column("Hands", justify="right")
    table.add_column("VPIP", justify="right")
    table.add_column("PFR", justify="right")
    if show_gto:
        table.add_column("Target RFI", justify="right", style="dim")
    table.add_column("3-Bet", justify="right")
    table.add_column("C-Bet", justify="right")
    table.add_column("AF", justify="right")

    for pos in ["UTG", "UTG1", "CO", "BTN", "SB", "BB"]:
        if pos not in hero_stats.by_position:
            continue

        stats = hero_stats.by_position[pos]
        bench = GTO_BENCHMARKS.by_position.get(pos)

        row = [
            pos,
            f"{stats.hands:,}",
            f"{stats.vpip:.1f}%",
            _format_with_deviation(stats.pfr, bench.rfi if bench else None),
        ]

        if show_gto:
            row.append(f"{bench.rfi:.0f}%" if bench else "-")

        row.extend([
            f"{stats.three_bet:.1f}%",
            f"{stats.cbet:.1f}%",
            f"{stats.af:.2f}",
        ])

        table.add_row(*row)

    console.print(table)


def _format_with_deviation(value: float, target: float = None) -> str:
    """Format value with color based on deviation from target."""
    if target is None:
        return f"{value:.1f}%"

    diff = value - target
    if abs(diff) <= 5:
        return f"[green]{value:.1f}%[/]"
    elif abs(diff) <= 10:
        return f"[yellow]{value:.1f}%[/]"
    else:
        return f"[red]{value:.1f}%[/]"


def _display_position_stats(
    console: Console, hero_stats, position: str, show_gto: bool
) -> None:
    """Display detailed stats for a specific position."""
    if position not in hero_stats.by_position:
        console.print(f"[yellow]No hands found for position {position}[/]")
        return

    stats = hero_stats.by_position[position]
    bench = GTO_BENCHMARKS.by_position.get(position)

    table = Table(
        title=f"{position} Statistics",
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
    )
    table.add_column("Statistic", style="cyan")
    table.add_column("Your Value", justify="right")
    if show_gto and bench:
        table.add_column("GTO Target", justify="right", style="dim")
        table.add_column("Deviation", justify="right")

    table.add_row("Hands", f"{stats.hands:,}", "", "")

    if show_gto and bench:
        _add_position_stat_row(table, "VPIP", stats.vpip, None)
        _add_position_stat_row(table, "PFR (RFI)", stats.pfr, bench.rfi)
        _add_position_stat_row(table, "3-Bet", stats.three_bet, bench.three_bet_vs_open)
        _add_position_stat_row(table, "Fold to 3-Bet", stats.fold_to_3bet, bench.fold_vs_3bet)
        _add_position_stat_row(table, "C-Bet Flop", stats.cbet, bench.cbet_flop)
        _add_position_stat_row(table, "Fold to C-Bet", stats.fold_to_cbet, bench.fold_vs_cbet)
    else:
        table.add_row("VPIP", f"{stats.vpip:.1f}%")
        table.add_row("PFR", f"{stats.pfr:.1f}%")
        table.add_row("3-Bet", f"{stats.three_bet:.1f}%")
        table.add_row("Fold to 3-Bet", f"{stats.fold_to_3bet:.1f}%")
        table.add_row("C-Bet", f"{stats.cbet:.1f}%")
        table.add_row("Fold to C-Bet", f"{stats.fold_to_cbet:.1f}%")

    table.add_row("AF", f"{stats.af:.2f}", "", "")
    table.add_row("WTSD", f"{stats.wtsd:.1f}%", "", "")
    table.add_row("W$SD", f"{stats.wsd:.1f}%", "", "")

    console.print(table)


def _add_position_stat_row(
    table: Table, name: str, value: float, target: float = None
) -> None:
    """Add row with deviation from target."""
    if target is None:
        table.add_row(name, f"{value:.1f}%", "-", "-")
        return

    diff = value - target
    if abs(diff) <= 5:
        color = "green"
    elif abs(diff) <= 10:
        color = "yellow"
    else:
        color = "red"

    diff_str = f"[{color}]{diff:+.1f}%[/{color}]"
    table.add_row(name, f"{value:.1f}%", f"{target:.0f}%", diff_str)


def _display_leaks(
    console: Console, leaks: list[Leak], verbose: bool
) -> None:
    """Display identified leaks."""
    if not leaks:
        console.print(Panel(
            "[green]No significant leaks identified![/]\n"
            "Your stats are within acceptable ranges.",
            title="Leak Analysis",
            border_style="green",
        ))
        return

    major = [l for l in leaks if l.severity == "major"]
    moderate = [l for l in leaks if l.severity == "moderate"]
    minor = [l for l in leaks if l.severity == "minor"]

    content_parts = []

    if major:
        content_parts.append("[bold red]Major Leaks:[/]")
        for leak in major:
            pos = f"[{leak.position}] " if leak.position else ""
            content_parts.append(f"  [red]!!!![/] {pos}{leak.description}")
            if verbose:
                content_parts.append(f"      [dim]Suggestion: {leak.suggestion}[/]")

    if moderate:
        content_parts.append("")
        content_parts.append("[bold yellow]Moderate Leaks:[/]")
        for leak in moderate:
            pos = f"[{leak.position}] " if leak.position else ""
            content_parts.append(f"  [yellow]!![/] {pos}{leak.description}")
            if verbose:
                content_parts.append(f"      [dim]Suggestion: {leak.suggestion}[/]")

    if minor and verbose:
        content_parts.append("")
        content_parts.append("[bold blue]Minor Leaks:[/]")
        for leak in minor:
            pos = f"[{leak.position}] " if leak.position else ""
            content_parts.append(f"  [blue]![/] {pos}{leak.description}")

    console.print(Panel(
        "\n".join(content_parts),
        title=f"Leak Analysis ({len(leaks)} issues found)",
        border_style="red" if major else "yellow",
    ))


def _display_strengths(console: Console, strengths: list[str]) -> None:
    """Display areas where hero is playing well."""
    content = "\n".join(f"  [green]+[/] {s}" for s in strengths)
    console.print(Panel(
        content,
        title="Strengths",
        border_style="green",
    ))


def _display_spot_analysis(console: Console, analyzer: SpotAnalyzer) -> None:
    """Display spot-by-spot analysis."""
    spots = analyzer.get_spot_summary()

    table = Table(
        title="Spot Analysis",
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
    )
    table.add_column("Spot", style="cyan")
    table.add_column("Opportunities", justify="right")
    table.add_column("Fold %", justify="right")
    table.add_column("Call %", justify="right")
    table.add_column("Bet/Raise %", justify="right")
    table.add_column("Deviation", justify="left")

    for spot_name, stats in spots.items():
        if stats.opportunities == 0:
            continue

        deviation = stats.deviation_summary
        if "optimal" in deviation.lower():
            deviation = f"[green]{deviation}[/]"
        elif "too much" in deviation.lower():
            deviation = f"[red]{deviation}[/]"
        elif "not enough" in deviation.lower():
            deviation = f"[yellow]{deviation}[/]"

        table.add_row(
            spot_name.replace("_", " ").title(),
            str(stats.opportunities),
            f"{stats.fold_pct:.1f}%",
            f"{stats.call_pct:.1f}%",
            f"{stats.bet_pct:.1f}%",
            deviation,
        )

    console.print(table)


def _display_flagged_hands(
    console: Console, flagger: HandFlagger, limit: int
) -> None:
    """Display hands flagged for review."""
    flagged = flagger.flag_hands(limit=limit)

    if not flagged:
        console.print("[dim]No hands flagged for review.[/]")
        return

    table = Table(
        title=f"Hands Flagged for Review ({len(flagged)} hands)",
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
    )
    table.add_column("Priority", justify="center", width=4)
    table.add_column("Hand ID", style="cyan")
    table.add_column("Position")
    table.add_column("Pot", justify="right")
    table.add_column("Result", justify="right")
    table.add_column("Flags")
    table.add_column("Summary", style="dim")

    for hand in flagged:
        priority_colors = {1: "red", 2: "yellow", 3: "blue"}
        priority_str = f"[{priority_colors[hand.priority]}]P{hand.priority}[/{priority_colors[hand.priority]}]"

        result_str = f"{hand.hero_result:+.1f}bb"
        if hand.hero_result < 0:
            result_str = f"[red]{result_str}[/]"
        else:
            result_str = f"[green]{result_str}[/]"

        flags_str = ", ".join(f.name.replace("_", " ").lower() for f in hand.flags[:2])

        table.add_row(
            priority_str,
            hand.hand_id[:12] + "..." if len(hand.hand_id) > 15 else hand.hand_id,
            hand.hero_position,
            f"{hand.pot_size:.1f}bb",
            result_str,
            flags_str,
            hand.summary[:40] + "..." if len(hand.summary) > 40 else hand.summary,
        )

    console.print(table)

    console.print()
    console.print("[dim]Use 'review_hand.py <hand_id>' to analyze specific hands.[/]")


if __name__ == "__main__":
    sys.exit(main())
