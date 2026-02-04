"""Population-level analysis for microstakes tendencies."""

import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from .stats import PlayerStats, compute_stats


@dataclass
class PopulationStats:
    """
    Aggregate population statistics with segmentation.

    Tracks stats across different stack depths and positions
    to identify exploitable population tendencies.
    """
    # Overall stats
    overall: PlayerStats = field(default_factory=PlayerStats)

    # By stack depth (in BBs)
    short_stack: PlayerStats = field(default_factory=PlayerStats)   # < 50bb
    medium_stack: PlayerStats = field(default_factory=PlayerStats)  # 50-100bb
    deep_stack: PlayerStats = field(default_factory=PlayerStats)    # > 100bb

    # By position
    by_position: dict[str, PlayerStats] = field(default_factory=dict)

    # Position vs position matchups (e.g., "BTN_vs_BB")
    matchups: dict[str, PlayerStats] = field(default_factory=dict)

    def get_tendency(self, stat_name: str) -> str:
        """
        Describe population tendency for a given stat.

        Returns a string description of whether population
        is above/below typical ranges.
        """
        value = getattr(self.overall, stat_name, None)
        if value is None:
            return "unknown"

        # Typical ranges for microstakes
        typical_ranges = {
            "vpip": (25, 35),      # Typical fish: 40+
            "pfr": (15, 22),       # Typical fish: < 10
            "three_bet": (5, 9),   # Typical fish: < 4
            "cbet": (55, 70),
            "fold_to_cbet": (40, 55),
            "af": (1.5, 3.0),
            "wtsd": (25, 35),
        }

        if stat_name not in typical_ranges:
            return "n/a"

        low, high = typical_ranges[stat_name]

        if value < low:
            return "low (exploitable)"
        elif value > high:
            return "high (exploitable)"
        else:
            return "normal"


class PopulationAnalyzer:
    """
    Analyze population tendencies from hand history database.

    Designed for anonymous player pools (Ignition/Bovada) where
    we can't track individual players but can identify aggregate
    population weaknesses.
    """

    def __init__(self, conn: sqlite3.Connection):
        """
        Initialize analyzer with database connection.

        Args:
            conn: SQLite connection to hand history database
        """
        self.conn = conn

    def analyze(self) -> PopulationStats:
        """
        Perform full population analysis.

        Returns:
            PopulationStats with all computed statistics
        """
        pop_stats = PopulationStats()

        # Overall stats
        pop_stats.overall = compute_stats(self.conn)

        # Stack depth segmentation
        pop_stats.short_stack = compute_stats(self.conn, max_stack=50)
        pop_stats.medium_stack = compute_stats(self.conn, min_stack=50, max_stack=100)
        pop_stats.deep_stack = compute_stats(self.conn, min_stack=100)

        # Position stats
        for pos in ["UTG", "UTG1", "CO", "BTN", "SB", "BB"]:
            pop_stats.by_position[pos] = compute_stats(self.conn, position=pos)

        # Key matchups
        pop_stats.matchups = self._analyze_matchups()

        return pop_stats

    def _analyze_matchups(self) -> dict[str, PlayerStats]:
        """Analyze specific position vs position matchups."""
        matchups = {}

        # BTN vs BB (most common HU postflop spot)
        matchups["BTN_vs_BB"] = self._analyze_matchup("BTN", "BB")

        # CO vs BTN
        matchups["CO_vs_BTN"] = self._analyze_matchup("CO", "BTN")

        # SB vs BB
        matchups["SB_vs_BB"] = self._analyze_matchup("SB", "BB")

        return matchups

    def _analyze_matchup(self, pos1: str, pos2: str) -> PlayerStats:
        """
        Analyze stats for a specific position matchup.

        Returns stats for pos1 when heads-up against pos2.
        """
        # This is a simplified version - full implementation would
        # filter for hands that were HU between these specific positions
        return compute_stats(self.conn, position=pos1)

    def get_exploits(self) -> list[str]:
        """
        Identify exploitable population tendencies.

        Returns:
            List of exploitation recommendations
        """
        exploits = []
        stats = self.analyze()

        # Check for common micro-stakes leaks
        if stats.overall.fold_to_cbet > 55:
            exploits.append(
                f"Population folds to c-bet {stats.overall.fold_to_cbet:.1f}% - "
                "increase c-bet frequency"
            )

        if stats.overall.fold_to_3bet > 65:
            exploits.append(
                f"Population folds to 3-bet {stats.overall.fold_to_3bet:.1f}% - "
                "widen 3-bet bluffing range"
            )

        if stats.overall.vpip > 40:
            exploits.append(
                f"Population VPIP is {stats.overall.vpip:.1f}% - "
                "tighten up and value bet wider"
            )

        if stats.overall.three_bet < 5:
            exploits.append(
                f"Population 3-bets only {stats.overall.three_bet:.1f}% - "
                "open wider in late position"
            )

        if stats.overall.af < 1.5:
            exploits.append(
                f"Population aggression factor is {stats.overall.af:.2f} - "
                "respect their bets/raises more"
            )

        if stats.overall.wtsd > 35:
            exploits.append(
                f"Population WTSD is {stats.overall.wtsd:.1f}% - "
                "reduce bluff frequency on later streets"
            )

        # Position-specific exploits
        bb_stats = stats.by_position.get("BB")
        if bb_stats and bb_stats.fold_to_cbet > 60:
            exploits.append(
                f"BB folds to c-bet {bb_stats.fold_to_cbet:.1f}% - "
                "c-bet more aggressively in position vs BB"
            )

        btn_stats = stats.by_position.get("BTN")
        if btn_stats and btn_stats.vpip > 45:
            exploits.append(
                f"BTN VPIP is {btn_stats.vpip:.1f}% - "
                "3-bet wider from blinds vs BTN opens"
            )

        return exploits

    def get_position_summary(self) -> dict[str, dict[str, float]]:
        """
        Get summary stats for each position.

        Returns:
            Dict mapping position to key stats
        """
        stats = self.analyze()
        summary = {}

        for pos, pos_stats in stats.by_position.items():
            summary[pos] = {
                "hands": pos_stats.hands,
                "vpip": pos_stats.vpip,
                "pfr": pos_stats.pfr,
                "3bet": pos_stats.three_bet,
                "cbet": pos_stats.cbet,
                "af": pos_stats.af,
            }

        return summary

    def get_stack_depth_summary(self) -> dict[str, dict[str, float]]:
        """
        Get summary stats by stack depth.

        Returns:
            Dict mapping stack category to key stats
        """
        stats = self.analyze()

        return {
            "short (<50bb)": {
                "hands": stats.short_stack.hands,
                "vpip": stats.short_stack.vpip,
                "pfr": stats.short_stack.pfr,
                "af": stats.short_stack.af,
            },
            "medium (50-100bb)": {
                "hands": stats.medium_stack.hands,
                "vpip": stats.medium_stack.vpip,
                "pfr": stats.medium_stack.pfr,
                "af": stats.medium_stack.af,
            },
            "deep (>100bb)": {
                "hands": stats.deep_stack.hands,
                "vpip": stats.deep_stack.vpip,
                "pfr": stats.deep_stack.pfr,
                "af": stats.deep_stack.af,
            },
        }
