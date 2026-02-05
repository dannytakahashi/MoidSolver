#!/usr/bin/env python3
"""Review specific hands with detailed analysis.

Analyze a specific hand, see the full action sequence,
compare to solver recommendations, and understand where you deviated.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from moid.db import get_connection, HandRepository
from moid.parser.models import Street, ActionType
from moid.analysis.benchmarks import classify_board_texture, GTO_BENCHMARKS


def main():
    parser = argparse.ArgumentParser(
        description="Review and analyze specific poker hands"
    )
    parser.add_argument(
        "hand_id",
        nargs="?",
        help="Hand ID to review (partial match supported)",
    )
    parser.add_argument(
        "-d", "--database",
        default="hands.db",
        help="Database file path (default: hands.db)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent hands",
    )
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=10,
        help="Number of hands to list (default: 10)",
    )
    parser.add_argument(
        "--losses",
        action="store_true",
        help="Only show hands where hero lost",
    )
    parser.add_argument(
        "--big-pots",
        action="store_true",
        help="Only show large pots (30bb+)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show additional analysis",
    )

    args = parser.parse_args()
    console = Console()

    # Open database
    db_path = Path(args.database)
    if not db_path.exists():
        console.print(f"[red]Database not found: {db_path}[/]")
        return 1

    conn = get_connection(db_path)
    repo = HandRepository(conn)

    if args.list or not args.hand_id:
        _list_hands(console, conn, args.num, args.losses, args.big_pots)
    else:
        _review_hand(console, conn, repo, args.hand_id, args.verbose)

    conn.close()
    return 0


def _list_hands(
    console: Console,
    conn,
    limit: int,
    losses_only: bool,
    big_pots_only: bool,
) -> None:
    """List recent hands."""
    query = """
        SELECT
            h.hand_id,
            h.timestamp,
            h.bb,
            h.board,
            h.total_pot,
            p.position,
            p.hole_cards,
            p.result
        FROM hands h
        JOIN players p ON h.id = p.hand_id
        WHERE p.is_hero = 1
    """

    if losses_only:
        query += " AND p.result < 0"

    if big_pots_only:
        query += " AND h.total_pot >= 30"

    query += " ORDER BY h.timestamp DESC LIMIT ?"

    cursor = conn.execute(query, (limit,))

    table = Table(
        title="Recent Hands",
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
    )
    table.add_column("Hand ID", style="cyan")
    table.add_column("Position")
    table.add_column("Cards")
    table.add_column("Board")
    table.add_column("Pot", justify="right")
    table.add_column("Result", justify="right")
    table.add_column("Time")

    for row in cursor:
        hand_id = row["hand_id"]
        if len(hand_id) > 15:
            hand_id = hand_id[:12] + "..."

        cards = row["hole_cards"] or "-"
        if cards != "-":
            cards = cards.replace(" ", "")

        board = row["board"] or "-"
        if board != "-":
            board = board.replace(" ", " ")
            if len(board) > 14:
                board = board[:14]

        result = row["result"] or 0
        result_str = f"{result:+.1f}bb"
        if result < 0:
            result_str = f"[red]{result_str}[/]"
        elif result > 0:
            result_str = f"[green]{result_str}[/]"

        timestamp = row["timestamp"][:16]  # Trim to minutes

        table.add_row(
            hand_id,
            row["position"],
            cards,
            board,
            f"{row['total_pot']:.1f}bb",
            result_str,
            timestamp,
        )

    console.print(table)
    console.print()
    console.print("[dim]Use 'review_hand.py <hand_id>' to analyze a specific hand.[/]")


def _review_hand(
    console: Console,
    conn,
    repo: HandRepository,
    hand_id: str,
    verbose: bool,
) -> None:
    """Review a specific hand in detail."""
    # Try to find the hand
    hand = repo.get_hand(hand_id)

    if not hand:
        # Try partial match
        cursor = conn.execute(
            "SELECT hand_id FROM hands WHERE hand_id LIKE ? LIMIT 5",
            (f"%{hand_id}%",),
        )
        matches = [row["hand_id"] for row in cursor]

        if not matches:
            console.print(f"[red]Hand not found: {hand_id}[/]")
            return

        if len(matches) == 1:
            hand = repo.get_hand(matches[0])
        else:
            console.print("[yellow]Multiple matches found:[/]")
            for match in matches:
                console.print(f"  {match}")
            console.print("\n[dim]Please specify a more complete hand ID.[/]")
            return

    # Display hand header
    console.print()
    _display_hand_header(console, hand)

    # Display players
    console.print()
    _display_players(console, hand)

    # Display action by street
    console.print()
    _display_actions(console, hand)

    # Display analysis
    if verbose:
        console.print()
        _display_analysis(console, hand)


def _display_hand_header(console: Console, hand) -> None:
    """Display hand header with basic info."""
    board_str = " ".join(hand.board) if hand.board else "No board"
    texture = classify_board_texture(hand.board) if len(hand.board) >= 3 else ""

    hero = next((p for p in hand.players if p.is_hero), None)
    hero_info = ""
    if hero:
        cards = f"{hero.hole_cards[0]}{hero.hole_cards[1]}" if hero.hole_cards else "unknown"
        result = hero.result or 0
        result_str = f"{result:+.1f}bb"
        if result < 0:
            result_str = f"[red]{result_str}[/]"
        elif result > 0:
            result_str = f"[green]{result_str}[/]"
        hero_info = f"\nHero: {hero.position.name} with {cards} | Result: {result_str}"

    content = f"""[bold]Hand:[/] {hand.hand_id}
[bold]Stakes:[/] ${hand.stakes[0]}/{hand.stakes[1]}
[bold]Board:[/] {board_str} [dim]({texture})[/]
[bold]Pot:[/] {hand.total_pot:.1f}bb{hero_info}"""

    console.print(Panel(content, title="Hand Summary", border_style="blue"))


def _display_players(console: Console, hand) -> None:
    """Display player information."""
    table = Table(
        title="Players",
        show_header=True,
        header_style="bold",
        box=box.SIMPLE,
    )
    table.add_column("Position")
    table.add_column("Stack", justify="right")
    table.add_column("Cards")
    table.add_column("Result", justify="right")
    table.add_column("")

    # Sort by position
    sorted_players = sorted(hand.players, key=lambda p: p.position.value)

    # Check if this is a chop (multiple winners)
    num_winners = len(hand.winners)
    is_chop = num_winners > 1

    for player in sorted_players:
        cards = ""
        if player.hole_cards:
            cards = f"{player.hole_cards[0]}{player.hole_cards[1]}"
        elif player.showed_cards:
            cards = "[shown]"

        result_str = ""
        if player.result:
            result_str = f"{player.result:+.1f}bb"
            if player.result < 0:
                result_str = f"[red]{result_str}[/]"
            elif player.result > 0:
                result_str = f"[green]{result_str}[/]"

        indicator = ""
        if player.is_hero and player.position in hand.winners:
            if is_chop:
                indicator = "[bold cyan](Hero, Chop)[/]"
            else:
                indicator = "[bold cyan](Hero, Winner)[/]"
        elif player.is_hero:
            indicator = "[bold cyan](Hero)[/]"
        elif player.position in hand.winners:
            if is_chop:
                indicator = "[yellow](Chop)[/]"
            else:
                indicator = "[green](Winner)[/]"

        table.add_row(
            player.position.name,
            f"{player.stack:.1f}bb",
            cards,
            result_str,
            indicator,
        )

    console.print(table)


def _display_actions(console: Console, hand) -> None:
    """Display action sequence by street with pot sizes."""
    streets = [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]

    # Calculate pot at each street
    pot = 0.0
    street_pots = {}

    # Track pot through actions
    current_street = Street.PREFLOP
    for action in hand.actions:
        if action.street != current_street:
            street_pots[current_street] = pot
            current_street = action.street
        if action.amount > 0:
            pot += action.amount
    street_pots[current_street] = pot

    # Track if we went all-in (to show remaining board cards)
    all_in_street = None
    for action in hand.actions:
        if action.is_all_in or action.action_type == ActionType.ALL_IN:
            all_in_street = action.street
            break

    hero = next((p for p in hand.players if p.is_hero), None)

    for street in streets:
        actions = [a for a in hand.actions if a.street == street]

        # Build street name and board cards separately to avoid markup conflicts
        if street == Street.PREFLOP:
            street_name = "Preflop"
            board_cards = ""
        elif street == Street.FLOP and len(hand.board) >= 3:
            street_name = "Flop:"
            board_cards = " ".join(hand.board[:3])
        elif street == Street.TURN and len(hand.board) >= 4:
            street_name = "Turn:"
            board_cards = hand.board[3]
        elif street == Street.RIVER and len(hand.board) >= 5:
            street_name = "River:"
            board_cards = hand.board[4]
        else:
            # No board for this street
            if street == Street.FLOP:
                street_name = "Flop"
            elif street == Street.TURN:
                street_name = "Turn"
            elif street == Street.RIVER:
                street_name = "River"
            else:
                continue
            board_cards = ""

        # Get pot size at start of this street
        prev_street_idx = streets.index(street) - 1
        if prev_street_idx >= 0:
            pot_at_start = street_pots.get(streets[prev_street_idx], 0)
        else:
            pot_at_start = 0

        # For preflop, start with blinds
        if street == Street.PREFLOP:
            pot_at_start = 1.5  # SB + BB

        # Build the header line with proper markup
        if board_cards:
            header_line = f"[bold cyan]{street_name}[/bold cyan] [bold white]{board_cards}[/bold white] [dim](pot: {pot_at_start:.1f}bb)[/dim]"
        else:
            header_line = f"[bold cyan]{street_name}[/bold cyan] [dim](pot: {pot_at_start:.1f}bb)[/dim]"

        # Show street even if no actions (all-in on earlier street)
        if not actions:
            # Only show if we have board cards for this street and went all-in earlier
            if all_in_street and streets.index(street) > streets.index(all_in_street):
                if (street == Street.FLOP and len(hand.board) >= 3) or \
                   (street == Street.TURN and len(hand.board) >= 4) or \
                   (street == Street.RIVER and len(hand.board) >= 5):
                    console.print(f"\n{header_line}")
                    console.print("─" * 50)
                    console.print("  [dim]All-in, no further action[/dim]")
            continue

        # Skip blinds-only preflop display
        if street == Street.PREFLOP:
            non_blind = [
                a for a in actions
                if a.action_type not in (ActionType.POST_SB, ActionType.POST_BB, ActionType.POST_ANTE)
            ]
            if not non_blind:
                continue

        # Show street header with pot
        console.print(f"\n{header_line}")
        console.print("─" * 50)

        for action in actions:
            # Skip posting blinds in display
            if action.action_type in (ActionType.POST_SB, ActionType.POST_BB, ActionType.POST_ANTE):
                continue

            is_hero = hero and action.position == hero.position
            pos_str = action.position.name
            if is_hero:
                pos_str = f"[bold cyan]{pos_str}[/]"

            action_str = _format_action(action)

            console.print(f"  {pos_str}: {action_str}")

    # Show final result
    console.print(f"\n[bold]Final pot:[/] {hand.total_pot:.1f}bb")
    if hand.winners:
        winner_names = [w.name for w in hand.winners]
        if len(winner_names) > 1:
            console.print(f"[yellow]Chop between: {', '.join(winner_names)}[/]")
        else:
            console.print(f"[green]Winner: {winner_names[0]}[/]")


def _format_action(action) -> str:
    """Format an action for display."""
    action_colors = {
        ActionType.FOLD: "red",
        ActionType.CHECK: "dim",
        ActionType.CALL: "blue",
        ActionType.BET: "green",
        ActionType.RAISE: "green",
        ActionType.ALL_IN: "yellow bold",
    }

    color = action_colors.get(action.action_type, "white")
    name = action.action_type.name

    if action.amount > 0:
        if action.action_type == ActionType.ALL_IN:
            return f"[{color}]ALL-IN {action.amount:.1f}bb[/{color}]"
        elif action.action_type in (ActionType.BET, ActionType.RAISE):
            return f"[{color}]{name} {action.amount:.1f}bb[/{color}]"
        else:
            return f"[{color}]{name} {action.amount:.1f}bb[/{color}]"

    return f"[{color}]{name}[/{color}]"


def _display_analysis(console: Console, hand) -> None:
    """Display analysis and suggestions for the hand."""
    hero = next((p for p in hand.players if p.is_hero), None)
    if not hero:
        return

    analysis_points = []

    # Analyze preflop
    preflop_analysis = _analyze_preflop(hand, hero)
    if preflop_analysis:
        analysis_points.extend(preflop_analysis)

    # Analyze postflop
    postflop_analysis = _analyze_postflop(hand, hero)
    if postflop_analysis:
        analysis_points.extend(postflop_analysis)

    if analysis_points:
        content = "\n".join(f"• {point}" for point in analysis_points)
        console.print(Panel(content, title="Analysis", border_style="cyan"))
    else:
        console.print("[dim]No specific analysis notes for this hand.[/]")


def _analyze_preflop(hand, hero) -> list[str]:
    """Analyze preflop play."""
    points = []

    preflop_actions = hand.get_actions(street=Street.PREFLOP, position=hero.position)

    # Filter out blinds
    preflop_actions = [
        a for a in preflop_actions
        if a.action_type not in (ActionType.POST_SB, ActionType.POST_BB, ActionType.POST_ANTE)
    ]

    if not preflop_actions:
        return points

    first_action = preflop_actions[0]
    position = hero.position.name

    # Get benchmark for position
    bench = GTO_BENCHMARKS.by_position.get(position)

    # Check for limp
    if first_action.action_type == ActionType.CALL:
        # Check if this was a limp or a call
        prior_raises = [
            a for a in hand.get_actions(street=Street.PREFLOP)
            if a.action_type in (ActionType.RAISE, ActionType.BET)
            and a.position != hero.position
        ]
        if not prior_raises:
            points.append(
                f"[yellow]Limped from {position}.[/] Consider raising or folding - "
                "limping forfeits initiative and invites multiway pots."
            )
        else:
            # Cold call vs raise
            if bench:
                points.append(
                    f"Cold called from {position}. GTO cold-call rate: ~{bench.call_vs_open:.0f}%. "
                    "Ensure hand is in calling range."
                )

    # Check for fold
    if first_action.action_type == ActionType.FOLD:
        if bench and bench.rfi > 20:
            points.append(
                f"Folded from {position}. This position should open ~{bench.rfi:.0f}% of hands."
            )

    # Check for 3-bet pot
    raises = [
        a for a in hand.get_actions(street=Street.PREFLOP)
        if a.action_type == ActionType.RAISE
    ]
    if len(raises) >= 2:
        hero_raised = any(a.position == hero.position for a in raises)
        if hero_raised:
            points.append("Played a 3-bet pot. Review sizing and range construction.")

    return points


def _analyze_postflop(hand, hero) -> list[str]:
    """Analyze postflop play."""
    points = []

    if not hand.board:
        return points

    board_texture = classify_board_texture(hand.board)

    # Check if hero was the preflop aggressor
    pf_raises = [
        a for a in hand.get_actions(street=Street.PREFLOP)
        if a.action_type in (ActionType.RAISE, ActionType.BET)
    ]
    was_pfr = any(a.position == hero.position for a in pf_raises)

    # Flop analysis
    flop_actions = hand.get_actions(street=Street.FLOP, position=hero.position)
    if flop_actions:
        first_flop = flop_actions[0]

        if was_pfr:
            if first_flop.action_type == ActionType.CHECK:
                if board_texture == "dry":
                    points.append(
                        f"Checked back on [cyan]dry[/] flop as PFR. "
                        "Consider c-betting more often on favorable textures."
                    )
            elif first_flop.action_type in (ActionType.BET, ActionType.RAISE):
                if board_texture == "wet":
                    points.append(
                        f"C-bet on [yellow]wet[/] board. "
                        "This texture favors calling ranges - be selective."
                    )
        else:
            # Faced c-bet or action
            if first_flop.action_type == ActionType.FOLD:
                points.append(
                    "Folded on flop. Review whether hand had enough equity to continue."
                )

    # Turn/river analysis
    for street in [Street.TURN, Street.RIVER]:
        street_actions = hand.get_actions(street=street, position=hero.position)
        if street_actions:
            first_action = street_actions[0]

            if first_action.action_type == ActionType.FOLD and first_action.amount > 0:
                # Big fold
                if hand.total_pot > 30:
                    points.append(
                        f"Made a fold on {street.name.lower()} in a {hand.total_pot:.0f}bb pot. "
                        "Review pot odds and opponent's likely range."
                    )

    # Check for went to showdown
    if hero.showed_cards:
        if hero in hand.winners or hero.position in hand.winners:
            points.append(
                "[green]Won at showdown.[/] Review if value was maximized."
            )
        else:
            points.append(
                "[red]Lost at showdown.[/] Review hand reading and calling decisions."
            )

    return points


if __name__ == "__main__":
    sys.exit(main())
