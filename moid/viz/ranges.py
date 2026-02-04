"""Range display and visualization utilities."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from rich.console import Console
from rich.table import Table
from rich.style import Style
from rich.text import Text

from moid.solver.strategy import StrategyProfile


# Standard hand matrix layout (13x13)
RANKS = "AKQJT98765432"

# Pre-computed hand matrix positions
# Pairs on diagonal, suited above, offsuit below
HAND_MATRIX = []
for i, r1 in enumerate(RANKS):
    row = []
    for j, r2 in enumerate(RANKS):
        if i == j:
            row.append(f"{r1}{r2}")  # Pair
        elif i < j:
            row.append(f"{r1}{r2}s")  # Suited (above diagonal)
        else:
            row.append(f"{r2}{r1}o")  # Offsuit (below diagonal)
    HAND_MATRIX.append(row)


@dataclass
class RangeData:
    """Data for a single hand in the range."""
    hand: str
    frequency: float = 0.0  # 0-1, how often we play this hand
    action_freqs: dict[str, float] = field(default_factory=dict)  # Action breakdown
    ev: float = 0.0  # Expected value

    @property
    def color_value(self) -> float:
        """Value for coloring (0-1)."""
        return self.frequency


class RangeDisplay:
    """
    Display poker ranges as 13x13 matrix.

    Supports both terminal (rich) and graphical (matplotlib) output.
    """

    def __init__(self):
        self.console = Console()
        self.range_data: dict[str, RangeData] = {}

        # Initialize all hands with zero frequency
        for row in HAND_MATRIX:
            for hand in row:
                self.range_data[hand] = RangeData(hand=hand)

    def set_frequency(self, hand: str, frequency: float) -> None:
        """Set frequency for a hand."""
        if hand in self.range_data:
            self.range_data[hand].frequency = frequency

    def set_action_breakdown(
        self,
        hand: str,
        action_freqs: dict[str, float],
    ) -> None:
        """Set action frequency breakdown for a hand."""
        if hand in self.range_data:
            self.range_data[hand].action_freqs = action_freqs

    def load_from_range_string(self, range_str: str) -> None:
        """
        Load range from string notation.

        Examples:
            "AA,KK,QQ" - specific hands at 100%
            "AKs:0.5,AQs:0.75" - hands with frequencies
            "TT+" - pair range
        """
        for part in range_str.split(","):
            part = part.strip()
            if not part:
                continue

            # Check for frequency
            if ":" in part:
                hand, freq = part.split(":")
                frequency = float(freq)
            else:
                hand = part
                frequency = 1.0

            # Handle range notation
            hands = self._expand_range(hand)
            for h in hands:
                self.set_frequency(h, frequency)

    def _expand_range(self, notation: str) -> list[str]:
        """Expand range notation to list of hands."""
        hands = []

        # Pair plus: "TT+"
        if len(notation) == 3 and notation[2] == "+" and notation[0] == notation[1]:
            start_idx = RANKS.index(notation[0])
            for i in range(start_idx + 1):
                r = RANKS[i]
                hands.append(f"{r}{r}")
            return hands

        # Suited plus: "ATs+"
        if len(notation) == 4 and notation[3] == "+":
            r1 = notation[0]
            r2 = notation[1]
            suited = notation[2] == "s"

            r1_idx = RANKS.index(r1)
            r2_idx = RANKS.index(r2)

            for i in range(r1_idx + 1, r2_idx + 1):
                h = f"{r1}{RANKS[i]}{'s' if suited else 'o'}"
                hands.append(h)
            return hands

        # Single hand
        return [notation]

    def display_terminal(
        self,
        title: str = "Range",
        show_ev: bool = False,
    ) -> None:
        """Display range in terminal using rich."""
        table = Table(title=title, show_header=True, header_style="bold")

        # Add column headers
        table.add_column("", style="bold")
        for rank in RANKS:
            table.add_column(rank, justify="center")

        # Add rows
        for i, rank in enumerate(RANKS):
            row = [rank]
            for j in range(13):
                hand = HAND_MATRIX[i][j]
                data = self.range_data[hand]

                # Color based on frequency
                if data.frequency > 0.8:
                    style = Style(bgcolor="green", color="white")
                elif data.frequency > 0.5:
                    style = Style(bgcolor="yellow", color="black")
                elif data.frequency > 0.2:
                    style = Style(bgcolor="orange3", color="black")
                elif data.frequency > 0:
                    style = Style(bgcolor="red", color="white")
                else:
                    style = Style(bgcolor="grey30", color="grey50")

                # Format cell content
                if show_ev and data.ev != 0:
                    cell = f"{data.frequency*100:.0f}"
                else:
                    cell = f"{data.frequency*100:.0f}" if data.frequency > 0 else ""

                text = Text(cell.center(3), style=style)
                row.append(text)

            table.add_row(*row)

        self.console.print(table)

    def display_action_breakdown(
        self,
        title: str = "Strategy",
    ) -> None:
        """Display range with action breakdown colors."""
        table = Table(title=title, show_header=True, header_style="bold")

        table.add_column("", style="bold")
        for rank in RANKS:
            table.add_column(rank, justify="center")

        # Color scheme for actions
        action_colors = {
            "fold": "red",
            "check": "grey50",
            "call": "blue",
            "bet": "green",
            "raise": "yellow",
        }

        for i, rank in enumerate(RANKS):
            row = [rank]
            for j in range(13):
                hand = HAND_MATRIX[i][j]
                data = self.range_data[hand]

                if not data.action_freqs:
                    row.append(Text("", style=Style(bgcolor="grey30")))
                    continue

                # Get dominant action
                dominant = max(data.action_freqs, key=data.action_freqs.get)
                freq = data.action_freqs[dominant]

                color = action_colors.get(dominant, "white")
                style = Style(bgcolor=color, color="white" if color != "yellow" else "black")

                cell = f"{freq*100:.0f}"
                row.append(Text(cell.center(3), style=style))

            table.add_row(*row)

        self.console.print(table)

        # Legend
        self.console.print("\nLegend: ", end="")
        for action, color in action_colors.items():
            self.console.print(f"[{color}]{action}[/] ", end="")
        self.console.print()

    def plot(
        self,
        title: str = "Range",
        figsize: tuple[int, int] = (10, 10),
        cmap: str = "RdYlGn",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot range as heatmap using matplotlib.

        Args:
            title: Plot title
            figsize: Figure size
            cmap: Colormap name
            save_path: Optional path to save figure
        """
        if not HAS_MATPLOTLIB:
            self.console.print("[red]matplotlib not available. Use terminal display.[/]")
            return

        # Build frequency matrix
        freq_matrix = np.zeros((13, 13))
        for i in range(13):
            for j in range(13):
                hand = HAND_MATRIX[i][j]
                freq_matrix[i, j] = self.range_data[hand].frequency

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        im = ax.imshow(freq_matrix, cmap=cmap, vmin=0, vmax=1)

        # Add labels
        ax.set_xticks(np.arange(13))
        ax.set_yticks(np.arange(13))
        ax.set_xticklabels(list(RANKS))
        ax.set_yticklabels(list(RANKS))

        # Add hand labels in cells
        for i in range(13):
            for j in range(13):
                hand = HAND_MATRIX[i][j]
                freq = freq_matrix[i, j]

                # Choose text color based on background
                text_color = "white" if freq < 0.5 else "black"

                ax.text(j, i, hand, ha="center", va="center",
                       color=text_color, fontsize=8)

        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Frequency")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()


def display_range(
    hands: list[str],
    frequencies: Optional[list[float]] = None,
    title: str = "Range",
) -> None:
    """
    Convenience function to display a range.

    Args:
        hands: List of hands in range
        frequencies: Optional frequencies (default 1.0)
        title: Display title
    """
    display = RangeDisplay()

    if frequencies is None:
        frequencies = [1.0] * len(hands)

    for hand, freq in zip(hands, frequencies):
        display.set_frequency(hand, freq)

    display.display_terminal(title=title)


def display_strategy(
    profile: StrategyProfile,
    info_set_pattern: str = "",
    title: str = "Strategy",
) -> None:
    """
    Display strategy from a StrategyProfile.

    Args:
        profile: Strategy profile from solver
        info_set_pattern: Filter info sets by pattern
        title: Display title
    """
    display = RangeDisplay()

    # Group strategies by hand bucket
    for key, strategy in profile.strategies.items():
        if info_set_pattern and info_set_pattern not in key:
            continue

        # Extract hand bucket from key
        try:
            parts = key.split("|")
            bucket = int(parts[1].replace("B", ""))
        except (IndexError, ValueError):
            continue

        # Map bucket to hand range (simplified)
        # In full implementation, would use actual abstraction mapping
        bucket_hands = _bucket_to_hands(bucket)

        # Get action breakdown
        action_freqs = {}
        for action, prob in zip(strategy.actions, strategy.probabilities):
            action_name = action.action_type.name.lower()
            action_freqs[action_name] = prob

        for hand in bucket_hands:
            display.set_frequency(hand, 1.0)
            display.set_action_breakdown(hand, action_freqs)

    display.display_action_breakdown(title=title)


def _bucket_to_hands(bucket: int, num_buckets: int = 8) -> list[str]:
    """Map equity bucket to approximate hands."""
    # Simplified mapping - full implementation would use actual abstraction
    bucket_hands = {
        0: ["72o", "83o", "93o", "T2o"],
        1: ["32s", "43s", "54s", "65s"],
        2: ["K2o", "Q3o", "J4o", "T5o"],
        3: ["K2s", "Q3s", "J4s", "T5s"],
        4: ["KTo", "QJo", "JTo"],
        5: ["ATo", "KJo", "QTs"],
        6: ["AQo", "AJs", "KQs"],
        7: ["AA", "KK", "QQ", "AKs"],
    }
    return bucket_hands.get(bucket, [])
