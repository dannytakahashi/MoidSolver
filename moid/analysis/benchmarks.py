"""GTO benchmarks and practical microstakes adjustments.

This module provides reference frequencies for optimal poker play
and practical adjustments for microstakes environments.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PositionBenchmarks:
    """GTO benchmarks for a specific position."""

    # Preflop opening
    rfi: float = 0.0  # Raise first in %
    rfi_range: str = ""  # Example: "22+, A2s+, K9s+, Q9s+, J9s+, T8s+, 97s+, ..."

    # Facing raises
    fold_vs_open: float = 0.0  # Fold % when facing an open
    call_vs_open: float = 0.0  # Call % when facing an open (cold call)
    three_bet_vs_open: float = 0.0  # 3-bet % when facing an open

    # Facing 3-bet (after we opened)
    fold_vs_3bet: float = 0.0
    call_vs_3bet: float = 0.0
    four_bet_vs_3bet: float = 0.0

    # Postflop (as preflop aggressor)
    cbet_flop: float = 0.0  # C-bet frequency on flop
    cbet_turn: float = 0.0  # C-bet turn after c-betting flop
    cbet_river: float = 0.0  # C-bet river after c-betting turn

    # Defense vs c-bet
    fold_vs_cbet: float = 0.0
    call_vs_cbet: float = 0.0
    raise_vs_cbet: float = 0.0


@dataclass
class BoardTextureBenchmarks:
    """C-bet benchmarks by board texture."""

    # Dry boards (e.g., K72r, A83r)
    cbet_dry: float = 70.0
    sizing_dry: float = 0.33  # Pot fraction

    # Wet boards (e.g., JT9ss, 876ss)
    cbet_wet: float = 45.0
    sizing_wet: float = 0.66

    # Paired boards (e.g., 883, TT2)
    cbet_paired: float = 50.0
    sizing_paired: float = 0.50

    # High card boards (e.g., AKQ, AQJ)
    cbet_broadway: float = 60.0
    sizing_broadway: float = 0.33

    # Low boards (e.g., 732r, 542r)
    cbet_low: float = 75.0
    sizing_low: float = 0.33


@dataclass
class GTOBenchmarks:
    """
    Complete GTO benchmarks for 6-max NLHE.

    Values are approximate GTO frequencies from solver outputs.
    Reality varies by opponent, stack depth, and game dynamics.
    """

    # Position-specific benchmarks
    by_position: dict[str, PositionBenchmarks] = field(default_factory=dict)

    # Board texture benchmarks
    board_textures: BoardTextureBenchmarks = field(
        default_factory=BoardTextureBenchmarks
    )

    # General stat ranges (across all positions)
    vpip_range: tuple[float, float] = (22.0, 28.0)  # Typical GTO range
    pfr_range: tuple[float, float] = (18.0, 24.0)
    three_bet_range: tuple[float, float] = (6.0, 10.0)
    cbet_range: tuple[float, float] = (50.0, 70.0)
    fold_to_cbet_range: tuple[float, float] = (40.0, 50.0)
    af_range: tuple[float, float] = (2.0, 3.5)
    wtsd_range: tuple[float, float] = (24.0, 30.0)

    def __post_init__(self):
        """Initialize position benchmarks if not provided."""
        if not self.by_position:
            self.by_position = self._get_default_benchmarks()

    @staticmethod
    def _get_default_benchmarks() -> dict[str, PositionBenchmarks]:
        """Get default GTO benchmarks by position."""
        return {
            "UTG": PositionBenchmarks(
                rfi=15.0,
                rfi_range="77+, ATs+, KQs, AJo+, KQo",
                fold_vs_open=0.0,  # First to act, N/A
                call_vs_open=0.0,
                three_bet_vs_open=0.0,
                fold_vs_3bet=55.0,
                call_vs_3bet=30.0,
                four_bet_vs_3bet=15.0,
                cbet_flop=65.0,
                cbet_turn=55.0,
                cbet_river=45.0,
                fold_vs_cbet=45.0,
                call_vs_cbet=45.0,
                raise_vs_cbet=10.0,
            ),
            "UTG1": PositionBenchmarks(
                rfi=18.0,
                rfi_range="66+, A9s+, KJs+, QJs, ATo+, KQo",
                fold_vs_open=75.0,
                call_vs_open=15.0,
                three_bet_vs_open=10.0,
                fold_vs_3bet=52.0,
                call_vs_3bet=32.0,
                four_bet_vs_3bet=16.0,
                cbet_flop=65.0,
                cbet_turn=55.0,
                cbet_river=45.0,
                fold_vs_cbet=45.0,
                call_vs_cbet=45.0,
                raise_vs_cbet=10.0,
            ),
            "CO": PositionBenchmarks(
                rfi=27.0,
                rfi_range="55+, A2s+, K9s+, Q9s+, J9s+, T8s+, 97s+, 86s+, 75s+, "
                "64s+, 54s, ATo+, KTo+, QTo+, JTo",
                fold_vs_open=68.0,
                call_vs_open=18.0,
                three_bet_vs_open=14.0,
                fold_vs_3bet=50.0,
                call_vs_3bet=33.0,
                four_bet_vs_3bet=17.0,
                cbet_flop=60.0,
                cbet_turn=50.0,
                cbet_river=45.0,
                fold_vs_cbet=42.0,
                call_vs_cbet=48.0,
                raise_vs_cbet=10.0,
            ),
            "BTN": PositionBenchmarks(
                rfi=45.0,
                rfi_range="22+, A2s+, K2s+, Q4s+, J6s+, T6s+, 95s+, 85s+, 74s+, "
                "64s+, 53s+, 43s, A2o+, K5o+, Q8o+, J8o+, T8o+, 97o+, 87o",
                fold_vs_open=55.0,
                call_vs_open=25.0,
                three_bet_vs_open=20.0,
                fold_vs_3bet=48.0,
                call_vs_3bet=35.0,
                four_bet_vs_3bet=17.0,
                cbet_flop=55.0,
                cbet_turn=45.0,
                cbet_river=40.0,
                fold_vs_cbet=40.0,
                call_vs_cbet=50.0,
                raise_vs_cbet=10.0,
            ),
            "SB": PositionBenchmarks(
                rfi=40.0,  # Open-limp or raise
                rfi_range="22+, A2s+, K2s+, Q2s+, J4s+, T6s+, 95s+, 85s+, 74s+, "
                "64s+, 53s+, 43s, A2o+, K4o+, Q7o+, J8o+, T8o+, 97o+, 87o",
                fold_vs_open=55.0,
                call_vs_open=20.0,
                three_bet_vs_open=25.0,  # Higher 3-bet % from SB
                fold_vs_3bet=45.0,
                call_vs_3bet=35.0,
                four_bet_vs_3bet=20.0,
                cbet_flop=50.0,  # OOP, more careful
                cbet_turn=40.0,
                cbet_river=35.0,
                fold_vs_cbet=45.0,
                call_vs_cbet=45.0,
                raise_vs_cbet=10.0,
            ),
            "BB": PositionBenchmarks(
                rfi=0.0,  # BB doesn't RFI, they defend
                rfi_range="",
                fold_vs_open=50.0,  # Wide defense due to pot odds
                call_vs_open=40.0,
                three_bet_vs_open=10.0,  # Lower 3-bet, more calling
                fold_vs_3bet=40.0,
                call_vs_3bet=45.0,
                four_bet_vs_3bet=15.0,
                cbet_flop=40.0,  # BB is usually facing c-bet, not making it
                cbet_turn=35.0,
                cbet_river=30.0,
                fold_vs_cbet=45.0,  # Key stat for BB
                call_vs_cbet=45.0,
                raise_vs_cbet=10.0,
            ),
        }

    def get_benchmark(self, position: str, stat: str) -> Optional[float]:
        """Get a specific benchmark value."""
        if position not in self.by_position:
            return None

        pos_bench = self.by_position[position]
        return getattr(pos_bench, stat, None)

    def get_overall_range(self, stat: str) -> Optional[tuple[float, float]]:
        """Get the expected range for a stat across all positions."""
        ranges = {
            "vpip": self.vpip_range,
            "pfr": self.pfr_range,
            "three_bet": self.three_bet_range,
            "cbet": self.cbet_range,
            "fold_to_cbet": self.fold_to_cbet_range,
            "af": self.af_range,
            "wtsd": self.wtsd_range,
        }
        return ranges.get(stat)


@dataclass
class MicrostakesAdjustments:
    """
    Practical adjustments for microstakes play.

    These deviate from GTO to exploit common population tendencies:
    - Players don't 3-bet enough, so we can open wider
    - Players call too much, so we value bet thinner but bluff less
    - Players don't fold to c-bets enough, so we need stronger hands
    """

    # Preflop adjustments (vs GTO)
    open_adjustment: dict[str, float] = field(default_factory=dict)

    # Postflop adjustments
    cbet_adjustment: dict[str, float] = field(default_factory=dict)
    bluff_frequency_multiplier: float = 0.7  # Reduce bluffs by 30%
    value_bet_threshold_adjustment: float = -5.0  # Value bet 5% thinner

    # Tendencies to exploit
    exploits: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default adjustments."""
        if not self.open_adjustment:
            self.open_adjustment = {
                # Open tighter from EP (players don't 3-bet enough,
                # but they also don't fold enough postflop)
                "UTG": -3.0,  # 15% -> 12%
                "UTG1": -2.0,  # 18% -> 16%
                "CO": +2.0,  # 27% -> 29%, steal more
                "BTN": +5.0,  # 45% -> 50%, steal more
                "SB": -5.0,  # 40% -> 35%, too often called by BB
            }

        if not self.cbet_adjustment:
            self.cbet_adjustment = {
                # Adjust c-bet frequencies based on board texture
                "dry": +10.0,  # More c-betting on dry boards
                "wet": -10.0,  # Less c-betting on wet boards
                "paired": +5.0,  # Slightly more on paired
            }

        if not self.exploits:
            self.exploits = [
                "Value bet thinner - population calls too wide",
                "Reduce bluff frequency - population calls too often",
                "Open wider on BTN/CO - low 3-bet frequency",
                "Size up value bets - population is inelastic",
                "C-bet more on dry boards - population overfolds",
                "Don't hero call - population is underbluffing",
                "3-bet less for value, more for steal",
            ]

    def get_adjusted_rfi(self, position: str, gto_rfi: float) -> float:
        """Get adjusted RFI for microstakes."""
        adjustment = self.open_adjustment.get(position, 0.0)
        return max(0.0, min(100.0, gto_rfi + adjustment))

    def get_adjusted_cbet(self, board_type: str, gto_cbet: float) -> float:
        """Get adjusted c-bet frequency for microstakes."""
        adjustment = self.cbet_adjustment.get(board_type, 0.0)
        return max(0.0, min(100.0, gto_cbet + adjustment))


# Singleton instances for easy access
GTO_BENCHMARKS = GTOBenchmarks()
MICROSTAKES_ADJUSTMENTS = MicrostakesAdjustments()


def get_benchmark(position: str, stat: str) -> Optional[float]:
    """Convenience function to get a GTO benchmark."""
    return GTO_BENCHMARKS.get_benchmark(position, stat)


def get_adjusted_target(
    position: str, stat: str, for_microstakes: bool = True
) -> Optional[float]:
    """
    Get target frequency for a stat.

    Args:
        position: Player position
        stat: Stat name (e.g., "rfi", "cbet_flop")
        for_microstakes: Whether to apply microstakes adjustments

    Returns:
        Target frequency or None if not found
    """
    gto_value = get_benchmark(position, stat)
    if gto_value is None:
        return None

    if not for_microstakes:
        return gto_value

    # Apply microstakes adjustments
    if stat == "rfi":
        return MICROSTAKES_ADJUSTMENTS.get_adjusted_rfi(position, gto_value)

    # No specific adjustment, return GTO value
    return gto_value


def classify_board_texture(board: list[str]) -> str:
    """
    Classify board texture for c-bet decision.

    Args:
        board: List of board cards (e.g., ["As", "Kh", "7d"])

    Returns:
        Board texture: "dry", "wet", "paired", or "broadway"
    """
    if len(board) < 3:
        return "unknown"

    # Parse cards
    ranks = []
    suits = []
    for card in board[:3]:
        if len(card) == 2:
            rank_char = card[0].upper()
            rank_map = {
                "A": 14,
                "K": 13,
                "Q": 12,
                "J": 11,
                "T": 10,
                "9": 9,
                "8": 8,
                "7": 7,
                "6": 6,
                "5": 5,
                "4": 4,
                "3": 3,
                "2": 2,
            }
            ranks.append(rank_map.get(rank_char, 0))
            suits.append(card[1].lower())

    # Check for paired board
    if len(ranks) != len(set(ranks)):
        return "paired"

    # Check for broadway (all cards T+)
    if all(r >= 10 for r in ranks):
        return "broadway"

    # Check for flush draw (2+ same suit)
    from collections import Counter

    suit_counts = Counter(suits)
    has_flush_draw = any(c >= 2 for c in suit_counts.values())

    # Check for straight draw potential (connected)
    sorted_ranks = sorted(ranks)
    gaps = [sorted_ranks[i + 1] - sorted_ranks[i] for i in range(2)]
    is_connected = max(gaps) <= 2 and sum(gaps) <= 4

    if has_flush_draw and is_connected:
        return "wet"
    elif has_flush_draw or is_connected:
        return "semi_wet"
    else:
        return "dry"
