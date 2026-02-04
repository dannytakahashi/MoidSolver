#!/usr/bin/env python3
"""Solve a specific poker spot."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from moid.game.cards import Card
from moid.game.tree import GameTree
from moid.solver.cfr import CFRSolver, CFRConfig
from moid.solver.best_response import ExploitativeSolver
from moid.analysis.stats import PlayerStats
from moid.db import get_connection
from moid.analysis import compute_stats
from moid.viz import RangeDisplay


def main():
    parser = argparse.ArgumentParser(
        description="Solve a poker spot using CFR or exploitative solver"
    )
    parser.add_argument(
        "-b", "--board",
        required=True,
        help="Board cards (e.g., 'AsKhTd' or 'As Kh Td')",
    )
    parser.add_argument(
        "-p", "--pot",
        type=float,
        default=6.5,
        help="Starting pot in BBs (default: 6.5)",
    )
    parser.add_argument(
        "-s", "--stack",
        type=float,
        default=100,
        help="Effective stack in BBs (default: 100)",
    )
    parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=1000,
        help="Number of CFR iterations (default: 1000)",
    )
    parser.add_argument(
        "--bet-sizes",
        default="0.33,0.5,0.75,1.0",
        help="Bet sizes as pot fractions (default: 0.33,0.5,0.75,1.0)",
    )
    parser.add_argument(
        "-d", "--database",
        help="Use population stats from database for exploitative solving",
    )
    parser.add_argument(
        "--mode",
        choices=["nash", "exploitative"],
        default="nash",
        help="Solving mode (default: nash)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Save strategy to file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    console = Console()

    # Parse board
    board_str = args.board.replace(" ", "")
    board = []
    for i in range(0, len(board_str), 2):
        if i + 2 <= len(board_str):
            card = Card.from_string(board_str[i:i+2])
            board.append(card)

    if len(board) < 3:
        console.print("[red]Board must have at least 3 cards[/]")
        return 1

    board_display = " ".join(str(c) for c in board)
    console.print(f"[bold]Board:[/] {board_display}")
    console.print(f"[bold]Pot:[/] {args.pot} BB")
    console.print(f"[bold]Stack:[/] {args.stack} BB")
    console.print()

    # Parse bet sizes
    bet_sizes = [float(x) for x in args.bet_sizes.split(",")]
    console.print(f"[bold]Bet sizes:[/] {', '.join(f'{s*100:.0f}%' for s in bet_sizes)}")

    # Choose solving mode
    if args.mode == "exploitative" or args.database:
        # Load population stats
        if args.database:
            db_path = Path(args.database)
            if not db_path.exists():
                console.print(f"[red]Database not found: {db_path}[/]")
                return 1
            conn = get_connection(db_path)
            stats = compute_stats(conn)
            conn.close()
        else:
            # Use default micro-stakes population stats
            stats = PlayerStats(
                hands=10000,
                vpip=35.0,
                pfr=12.0,
                three_bet=4.0,
                fold_to_3bet=60.0,
                cbet=55.0,
                fold_to_cbet=55.0,
                af=1.5,
                wtsd=30.0,
                wsd=48.0,
            )

        console.print()
        console.print("[bold]Mode:[/] Exploitative (against population)")
        _display_opponent_stats(console, stats)

        console.print()
        console.print("[bold]Computing exploitative strategy...[/]")

        solver = ExploitativeSolver(
            stats=stats,
            board=board,
            starting_pot=args.pot,
            effective_stack=args.stack,
        )
        profile = solver.solve(bet_sizes=bet_sizes)

        console.print("[green]Done![/]")

        # Show recommendations
        _display_recommendations(console, solver, board)

    else:
        # Nash equilibrium via CFR
        console.print()
        console.print("[bold]Mode:[/] Nash equilibrium (CFR)")

        # Build game tree
        street = 1 if len(board) == 3 else 2 if len(board) == 4 else 3
        tree = GameTree(
            starting_pot=args.pot,
            effective_stack=args.stack,
            starting_street=street,
        )
        tree.build(bet_sizes)

        node_counts = tree.count_nodes()
        console.print(f"Game tree: {node_counts['total']} nodes ({node_counts['terminal']} terminal)")

        config = CFRConfig(
            num_iterations=args.iterations,
            use_monte_carlo=True,
        )
        solver = CFRSolver(tree, board, config)

        # Run solver with progress
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running CFR ({args.iterations} iterations)...")

            def callback(iteration, exploitability):
                progress.update(task, description=f"CFR iteration {iteration}, exploit={exploitability:.4f}")

            profile = solver.solve(callback=callback)

        console.print("[green]Converged![/]")

    # Display strategy
    console.print()
    _display_strategy_summary(console, profile)

    # Save if requested
    if args.output:
        profile.save(args.output)
        console.print(f"\n[bold]Strategy saved to:[/] {args.output}")

    return 0


def _display_opponent_stats(console: Console, stats: PlayerStats) -> None:
    """Display opponent stats summary."""
    table = Table(title="Opponent Model", show_header=False)
    table.add_column("Stat", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("VPIP", f"{stats.vpip:.1f}%")
    table.add_row("PFR", f"{stats.pfr:.1f}%")
    table.add_row("C-Bet", f"{stats.cbet:.1f}%")
    table.add_row("Fold to C-Bet", f"{stats.fold_to_cbet:.1f}%")
    table.add_row("AF", f"{stats.af:.2f}")

    console.print(table)


def _display_recommendations(console: Console, solver: ExploitativeSolver, board) -> None:
    """Display exploitation recommendations."""
    console.print()

    recommendations = []

    # Check various situations
    for strength in ["strong", "medium", "weak", "air"]:
        for facing_bet in [False, True]:
            action, explanation = solver.get_recommended_action(
                strength, facing_bet
            )
            situation = "facing bet" if facing_bet else "first to act"
            recommendations.append(f"[cyan]{strength.upper()}[/] ({situation}): {action.upper()} - {explanation}")

    panel = Panel(
        "\n".join(recommendations[:4]),  # Show first 4
        title="[bold]Exploitative Recommendations[/]",
        border_style="green",
    )
    console.print(panel)


def _display_strategy_summary(console: Console, profile) -> None:
    """Display strategy summary."""
    table = Table(title="Strategy Summary", show_header=True)
    table.add_column("Information Set", style="cyan")
    table.add_column("Actions", justify="left")

    # Show first few info sets
    count = 0
    for key, strategy in profile.strategies.items():
        if count >= 10:
            break

        actions_str = ", ".join(
            f"{a.action_type.name}:{p:.0%}"
            for a, p in zip(strategy.actions, strategy.probabilities)
            if p > 0.01
        )

        # Truncate key for display
        display_key = key[:40] + "..." if len(key) > 40 else key

        table.add_row(display_key, actions_str)
        count += 1

    console.print(table)
    console.print(f"\n[dim]Total information sets: {profile.num_info_sets()}[/]")


if __name__ == "__main__":
    sys.exit(main())
