"""Hero-focused analysis for personal study.

This module provides analysis specifically for the hero (you),
comparing your play against GTO benchmarks and identifying leaks.
"""

import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from .stats import PlayerStats
from .benchmarks import (
    GTOBenchmarks,
    GTO_BENCHMARKS,
    MICROSTAKES_ADJUSTMENTS,
)


@dataclass
class Leak:
    """A identified leak in hero's play."""

    category: str  # "preflop", "postflop", "tendencies"
    position: Optional[str]  # Position where leak occurs, or None for overall
    stat: str  # Stat name
    hero_value: float  # Hero's actual value
    optimal_value: float  # GTO or adjusted target
    deviation: float  # How far off (positive = too high, negative = too low)
    severity: str  # "minor", "moderate", "major"
    description: str  # Human-readable description
    suggestion: str  # How to fix it

    def __repr__(self) -> str:
        pos = f"[{self.position}] " if self.position else ""
        return f"Leak({pos}{self.stat}: {self.hero_value:.1f}% vs {self.optimal_value:.1f}%)"


@dataclass
class HeroStats:
    """Hero's complete statistics with comparisons."""

    # Overall stats
    overall: PlayerStats = field(default_factory=PlayerStats)

    # Stats by position
    by_position: dict[str, PlayerStats] = field(default_factory=dict)

    # Identified leaks
    leaks: list[Leak] = field(default_factory=list)

    # Strengths (areas where hero is doing well)
    strengths: list[str] = field(default_factory=list)

    @property
    def num_leaks(self) -> int:
        return len(self.leaks)

    @property
    def major_leaks(self) -> list[Leak]:
        return [l for l in self.leaks if l.severity == "major"]

    @property
    def moderate_leaks(self) -> list[Leak]:
        return [l for l in self.leaks if l.severity == "moderate"]


class HeroAnalyzer:
    """
    Analyze hero's play from hand history database.

    Focuses on identifying leaks by comparing hero's tendencies
    to GTO benchmarks and practical microstakes targets.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        benchmarks: Optional[GTOBenchmarks] = None,
        use_microstakes_adjustments: bool = True,
    ):
        """
        Initialize hero analyzer.

        Args:
            conn: Database connection
            benchmarks: GTO benchmarks (default: use standard)
            use_microstakes_adjustments: Apply practical adjustments
        """
        self.conn = conn
        self.benchmarks = benchmarks or GTO_BENCHMARKS
        self.use_adjustments = use_microstakes_adjustments

    def analyze(self) -> HeroStats:
        """
        Perform complete hero analysis.

        Returns:
            HeroStats with stats, leaks, and recommendations
        """
        hero_stats = HeroStats()

        # Compute overall and position-specific stats
        hero_stats.overall = self._compute_hero_stats()
        hero_stats.by_position = self._compute_position_stats()

        # Identify leaks
        hero_stats.leaks = self._identify_leaks(hero_stats)

        # Identify strengths
        hero_stats.strengths = self._identify_strengths(hero_stats)

        return hero_stats

    def _compute_hero_stats(
        self,
        position: Optional[str] = None,
        min_stack: Optional[float] = None,
        max_stack: Optional[float] = None,
    ) -> PlayerStats:
        """Compute stats for hero only."""
        stats = PlayerStats()

        # Build filters for hero only
        filters = ["p.is_hero = 1"]
        params = []

        if position:
            filters.append("p.position = ?")
            params.append(position)
        if min_stack is not None:
            filters.append("p.stack >= ?")
            params.append(min_stack)
        if max_stack is not None:
            filters.append("p.stack <= ?")
            params.append(max_stack)

        where_clause = " AND ".join(filters)

        # Count hands
        stats.hands = self._count_hero_hands(where_clause, params)
        if stats.hands == 0:
            return stats

        # Compute each stat
        vpip_pfr = self._compute_vpip_pfr(where_clause, params)
        stats.vpip = vpip_pfr["vpip"]
        stats.pfr = vpip_pfr["pfr"]

        three_bet = self._compute_3bet_stats(where_clause, params)
        stats.three_bet = three_bet["three_bet"]
        stats.fold_to_3bet = three_bet["fold_to_3bet"]

        cbet = self._compute_cbet_stats(where_clause, params)
        stats.cbet = cbet["cbet"]
        stats.fold_to_cbet = cbet["fold_to_cbet"]

        aggression = self._compute_aggression(where_clause, params)
        stats.af = aggression["af"]
        stats.afq = aggression["afq"]

        showdown = self._compute_showdown_stats(where_clause, params)
        stats.wtsd = showdown["wtsd"]
        stats.wsd = showdown["wsd"]

        return stats

    def _compute_position_stats(self) -> dict[str, PlayerStats]:
        """Compute hero stats broken down by position."""
        positions = ["UTG", "UTG1", "CO", "BTN", "SB", "BB"]
        result = {}

        for pos in positions:
            pos_stats = self._compute_hero_stats(position=pos)
            if pos_stats.hands > 0:
                result[pos] = pos_stats

        return result

    def _count_hero_hands(self, where_clause: str, params: list) -> int:
        """Count hero's hands."""
        query = f"""
            SELECT COUNT(DISTINCT h.id)
            FROM hands h
            JOIN players p ON h.id = p.hand_id
            WHERE {where_clause}
        """
        cursor = self.conn.execute(query, params)
        return cursor.fetchone()[0]

    def _compute_vpip_pfr(
        self, where_clause: str, params: list
    ) -> dict[str, float]:
        """Compute hero VPIP and PFR."""
        query = f"""
            SELECT
                COUNT(DISTINCT h.id) as total_hands,
                COUNT(DISTINCT CASE
                    WHEN a.street = 'PREFLOP'
                    AND a.action_type IN ('CALL', 'RAISE', 'BET', 'ALL_IN')
                    THEN h.id
                END) as vpip_hands,
                COUNT(DISTINCT CASE
                    WHEN a.street = 'PREFLOP'
                    AND a.action_type IN ('RAISE', 'BET', 'ALL_IN')
                    THEN h.id
                END) as pfr_hands
            FROM hands h
            JOIN players p ON h.id = p.hand_id
            LEFT JOIN actions a ON h.id = a.hand_id AND a.position = p.position
            WHERE {where_clause}
        """
        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()

        total = row[0] or 1
        return {
            "vpip": (row[1] or 0) / total * 100,
            "pfr": (row[2] or 0) / total * 100,
        }

    def _compute_3bet_stats(
        self, where_clause: str, params: list
    ) -> dict[str, float]:
        """Compute hero 3-bet stats."""
        query = f"""
            WITH hero_hands AS (
                SELECT DISTINCT h.id as hand_id, p.position
                FROM hands h
                JOIN players p ON h.id = p.hand_id
                WHERE {where_clause}
            ),
            facing_raise AS (
                SELECT hh.hand_id, hh.position,
                       MAX(CASE WHEN a2.action_type = 'RAISE' THEN 1 ELSE 0 END) as reraised
                FROM hero_hands hh
                JOIN actions a1 ON hh.hand_id = a1.hand_id
                    AND a1.street = 'PREFLOP'
                    AND a1.action_type = 'RAISE'
                    AND a1.position != hh.position
                LEFT JOIN actions a2 ON hh.hand_id = a2.hand_id
                    AND a2.position = hh.position
                    AND a2.street = 'PREFLOP'
                    AND a2.action_type = 'RAISE'
                GROUP BY hh.hand_id, hh.position
            )
            SELECT
                COUNT(*) as opportunities,
                SUM(reraised) as three_bets
            FROM facing_raise
        """
        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()

        opportunities = row[0] or 1
        three_bet = (row[1] or 0) / opportunities * 100 if opportunities > 0 else 0

        # Fold to 3-bet
        fold_query = f"""
            WITH hero_raises AS (
                SELECT DISTINCT h.id as hand_id, p.position
                FROM hands h
                JOIN players p ON h.id = p.hand_id
                JOIN actions a ON h.id = a.hand_id
                    AND a.position = p.position
                    AND a.street = 'PREFLOP'
                    AND a.action_type = 'RAISE'
                WHERE {where_clause}
            ),
            faced_3bet AS (
                SELECT hr.hand_id, hr.position,
                       MAX(CASE WHEN a2.action_type = 'FOLD' THEN 1 ELSE 0 END) as folded
                FROM hero_raises hr
                JOIN actions a1 ON hr.hand_id = a1.hand_id
                    AND a1.street = 'PREFLOP'
                    AND a1.action_type = 'RAISE'
                    AND a1.position != hr.position
                LEFT JOIN actions a2 ON hr.hand_id = a2.hand_id
                    AND a2.position = hr.position
                    AND a2.street = 'PREFLOP'
                GROUP BY hr.hand_id, hr.position
            )
            SELECT
                COUNT(*) as opportunities,
                SUM(folded) as folds
            FROM faced_3bet
        """
        cursor = self.conn.execute(fold_query, params)
        row = cursor.fetchone()

        fold_opps = row[0] or 1
        fold_to_3bet = (row[1] or 0) / fold_opps * 100 if fold_opps > 0 else 0

        return {"three_bet": three_bet, "fold_to_3bet": fold_to_3bet}

    def _compute_cbet_stats(
        self, where_clause: str, params: list
    ) -> dict[str, float]:
        """Compute hero c-bet stats."""
        query = f"""
            WITH hero_pfr AS (
                SELECT h.id as hand_id, p.position
                FROM hands h
                JOIN players p ON h.id = p.hand_id
                JOIN actions a ON h.id = a.hand_id
                    AND a.position = p.position
                    AND a.street = 'PREFLOP'
                    AND a.action_type IN ('RAISE', 'BET')
                WHERE {where_clause}
                AND h.board IS NOT NULL
                AND h.board != ''
            ),
            cbet_opps AS (
                SELECT hp.hand_id, hp.position,
                       MAX(CASE
                           WHEN a.action_type IN ('BET', 'RAISE', 'ALL_IN') THEN 1
                           ELSE 0
                       END) as cbetted
                FROM hero_pfr hp
                LEFT JOIN actions a ON hp.hand_id = a.hand_id
                    AND a.position = hp.position
                    AND a.street = 'FLOP'
                GROUP BY hp.hand_id, hp.position
            )
            SELECT
                COUNT(*) as opportunities,
                SUM(cbetted) as cbets
            FROM cbet_opps
        """
        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()

        cbet_opps = row[0] or 1
        cbet = (row[1] or 0) / cbet_opps * 100 if cbet_opps > 0 else 0

        # Fold to c-bet
        fold_query = f"""
            WITH hero_called_pf AS (
                SELECT h.id as hand_id, p.position
                FROM hands h
                JOIN players p ON h.id = p.hand_id
                JOIN actions a ON h.id = a.hand_id
                    AND a.position = p.position
                    AND a.street = 'PREFLOP'
                    AND a.action_type = 'CALL'
                WHERE {where_clause}
                AND h.board IS NOT NULL
            ),
            faced_cbet AS (
                SELECT hcp.hand_id, hcp.position,
                       MAX(CASE WHEN a2.action_type = 'FOLD' THEN 1 ELSE 0 END) as folded
                FROM hero_called_pf hcp
                JOIN actions a1 ON hcp.hand_id = a1.hand_id
                    AND a1.street = 'FLOP'
                    AND a1.action_type IN ('BET', 'RAISE')
                    AND a1.position != hcp.position
                LEFT JOIN actions a2 ON hcp.hand_id = a2.hand_id
                    AND a2.position = hcp.position
                    AND a2.street = 'FLOP'
                GROUP BY hcp.hand_id, hcp.position
            )
            SELECT
                COUNT(*) as opportunities,
                SUM(folded) as folds
            FROM faced_cbet
        """
        cursor = self.conn.execute(fold_query, params)
        row = cursor.fetchone()

        fold_opps = row[0] or 1
        fold_to_cbet = (row[1] or 0) / fold_opps * 100 if fold_opps > 0 else 0

        return {"cbet": cbet, "fold_to_cbet": fold_to_cbet}

    def _compute_aggression(
        self, where_clause: str, params: list
    ) -> dict[str, float]:
        """Compute hero aggression factor."""
        query = f"""
            SELECT
                SUM(CASE WHEN a.action_type IN ('BET', 'RAISE', 'ALL_IN') THEN 1 ELSE 0 END) as aggressive,
                SUM(CASE WHEN a.action_type = 'CALL' THEN 1 ELSE 0 END) as calls,
                SUM(CASE WHEN a.action_type = 'FOLD' THEN 1 ELSE 0 END) as folds,
                SUM(CASE WHEN a.action_type = 'CHECK' THEN 1 ELSE 0 END) as checks
            FROM hands h
            JOIN players p ON h.id = p.hand_id
            JOIN actions a ON h.id = a.hand_id AND a.position = p.position
            WHERE {where_clause}
            AND a.street != 'PREFLOP'
        """
        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()

        aggressive = row[0] or 0
        calls = row[1] or 1
        folds = row[2] or 0
        checks = row[3] or 0

        af = aggressive / calls if calls > 0 else 0
        total_actions = aggressive + calls + folds + checks
        afq = aggressive / total_actions * 100 if total_actions > 0 else 0

        return {"af": af, "afq": afq}

    def _compute_showdown_stats(
        self, where_clause: str, params: list
    ) -> dict[str, float]:
        """Compute hero showdown stats.

        WTSD = went to showdown / saw flop
        W$SD = won at showdown / went to showdown

        Uses hands.went_to_showdown flag instead of players.showed_cards
        to avoid selection bias (players can muck losing hands without showing).
        """
        query = f"""
            WITH saw_flop AS (
                SELECT DISTINCT h.id as hand_id, p.position,
                       h.went_to_showdown, p.is_winner
                FROM hands h
                JOIN players p ON h.id = p.hand_id
                WHERE {where_clause}
                AND h.board IS NOT NULL
                AND h.board != ''
                AND NOT EXISTS (
                    SELECT 1 FROM actions a
                    WHERE a.hand_id = h.id
                    AND a.position = p.position
                    AND a.street = 'PREFLOP'
                    AND a.action_type = 'FOLD'
                )
            )
            SELECT
                COUNT(*) as saw_flop,
                SUM(CASE WHEN went_to_showdown = 1 THEN 1 ELSE 0 END) as showdowns,
                SUM(CASE WHEN went_to_showdown = 1 AND is_winner = 1 THEN 1 ELSE 0 END) as won_sd
            FROM saw_flop
        """
        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()

        saw_flop = row[0] or 1
        showdowns = row[1] or 0
        won_sd = row[2] or 0

        wtsd = showdowns / saw_flop * 100 if saw_flop > 0 else 0
        wsd = won_sd / showdowns * 100 if showdowns > 0 else 0

        return {"wtsd": wtsd, "wsd": wsd}

    def _identify_leaks(self, hero_stats: HeroStats) -> list[Leak]:
        """Identify leaks in hero's play."""
        leaks = []

        # Check overall stats
        leaks.extend(self._check_overall_leaks(hero_stats.overall))

        # Check position-specific leaks
        for position, stats in hero_stats.by_position.items():
            leaks.extend(self._check_position_leaks(position, stats))

        # Sort by severity
        severity_order = {"major": 0, "moderate": 1, "minor": 2}
        leaks.sort(key=lambda l: severity_order.get(l.severity, 3))

        return leaks

    def _check_overall_leaks(self, stats: PlayerStats) -> list[Leak]:
        """Check for leaks in overall stats."""
        leaks = []

        # VPIP check
        vpip_range = self.benchmarks.vpip_range
        if stats.vpip < vpip_range[0] - 5:
            leaks.append(
                Leak(
                    category="preflop",
                    position=None,
                    stat="vpip",
                    hero_value=stats.vpip,
                    optimal_value=vpip_range[0],
                    deviation=stats.vpip - vpip_range[0],
                    severity="moderate" if stats.vpip < vpip_range[0] - 10 else "minor",
                    description=f"VPIP is {vpip_range[0] - stats.vpip:.1f}% below optimal range",
                    suggestion="Consider opening wider in late position and defending blinds more",
                )
            )
        elif stats.vpip > vpip_range[1] + 8:
            leaks.append(
                Leak(
                    category="preflop",
                    position=None,
                    stat="vpip",
                    hero_value=stats.vpip,
                    optimal_value=vpip_range[1],
                    deviation=stats.vpip - vpip_range[1],
                    severity="major" if stats.vpip > vpip_range[1] + 15 else "moderate",
                    description=f"VPIP is {stats.vpip - vpip_range[1]:.1f}% above optimal range",
                    suggestion="Tighten preflop ranges, especially from early position",
                )
            )

        # PFR check
        pfr_range = self.benchmarks.pfr_range
        if stats.pfr < pfr_range[0] - 5:
            leaks.append(
                Leak(
                    category="preflop",
                    position=None,
                    stat="pfr",
                    hero_value=stats.pfr,
                    optimal_value=pfr_range[0],
                    deviation=stats.pfr - pfr_range[0],
                    severity="moderate",
                    description=f"PFR is {pfr_range[0] - stats.pfr:.1f}% below optimal",
                    suggestion="Raise more preflop instead of limping or calling",
                )
            )

        # VPIP-PFR gap
        gap = stats.vpip - stats.pfr
        if gap > 8:
            leaks.append(
                Leak(
                    category="preflop",
                    position=None,
                    stat="vpip_pfr_gap",
                    hero_value=gap,
                    optimal_value=5.0,
                    deviation=gap - 5.0,
                    severity="major" if gap > 12 else "moderate",
                    description=f"Gap between VPIP and PFR is {gap:.1f}% (too passive)",
                    suggestion="Reduce cold-calling and limp frequency; raise or fold",
                )
            )

        # 3-bet check
        three_bet_range = self.benchmarks.three_bet_range
        if stats.three_bet < three_bet_range[0] - 2:
            leaks.append(
                Leak(
                    category="preflop",
                    position=None,
                    stat="three_bet",
                    hero_value=stats.three_bet,
                    optimal_value=three_bet_range[0],
                    deviation=stats.three_bet - three_bet_range[0],
                    severity="moderate",
                    description=f"3-bet frequency is {three_bet_range[0] - stats.three_bet:.1f}% below optimal",
                    suggestion="Add more 3-bets with value hands and select bluffs",
                )
            )

        # Fold to 3-bet check
        if stats.fold_to_3bet > 65:
            leaks.append(
                Leak(
                    category="preflop",
                    position=None,
                    stat="fold_to_3bet",
                    hero_value=stats.fold_to_3bet,
                    optimal_value=50.0,
                    deviation=stats.fold_to_3bet - 50.0,
                    severity="major" if stats.fold_to_3bet > 75 else "moderate",
                    description=f"Folding to 3-bets too often ({stats.fold_to_3bet:.1f}%)",
                    suggestion="Defend more vs 3-bets, especially in position with suited hands",
                )
            )
        elif stats.fold_to_3bet < 40:
            leaks.append(
                Leak(
                    category="preflop",
                    position=None,
                    stat="fold_to_3bet",
                    hero_value=stats.fold_to_3bet,
                    optimal_value=50.0,
                    deviation=stats.fold_to_3bet - 50.0,
                    severity="moderate",
                    description=f"Not folding to 3-bets enough ({stats.fold_to_3bet:.1f}%)",
                    suggestion="Fold more marginal opens vs 3-bets, especially OOP",
                )
            )

        # C-bet check
        cbet_range = self.benchmarks.cbet_range
        if stats.cbet < cbet_range[0] - 10:
            leaks.append(
                Leak(
                    category="postflop",
                    position=None,
                    stat="cbet",
                    hero_value=stats.cbet,
                    optimal_value=cbet_range[0],
                    deviation=stats.cbet - cbet_range[0],
                    severity="moderate",
                    description=f"C-bet frequency is low ({stats.cbet:.1f}%)",
                    suggestion="C-bet more on dry, favorable boards",
                )
            )
        elif stats.cbet > cbet_range[1] + 15:
            leaks.append(
                Leak(
                    category="postflop",
                    position=None,
                    stat="cbet",
                    hero_value=stats.cbet,
                    optimal_value=cbet_range[1],
                    deviation=stats.cbet - cbet_range[1],
                    severity="moderate",
                    description=f"C-bet frequency is high ({stats.cbet:.1f}%)",
                    suggestion="Check back more on wet, unfavorable boards",
                )
            )

        # Fold to c-bet check
        fold_cbet_range = self.benchmarks.fold_to_cbet_range
        if stats.fold_to_cbet > fold_cbet_range[1] + 10:
            leaks.append(
                Leak(
                    category="postflop",
                    position=None,
                    stat="fold_to_cbet",
                    hero_value=stats.fold_to_cbet,
                    optimal_value=fold_cbet_range[1],
                    deviation=stats.fold_to_cbet - fold_cbet_range[1],
                    severity="major" if stats.fold_to_cbet > 60 else "moderate",
                    description=f"Folding to c-bets too often ({stats.fold_to_cbet:.1f}%)",
                    suggestion="Defend more floats and raise more as bluffs",
                )
            )

        # Aggression check
        af_range = self.benchmarks.af_range
        if stats.af < af_range[0]:
            leaks.append(
                Leak(
                    category="tendencies",
                    position=None,
                    stat="af",
                    hero_value=stats.af,
                    optimal_value=af_range[0],
                    deviation=stats.af - af_range[0],
                    severity="moderate" if stats.af < 1.5 else "minor",
                    description=f"Aggression factor is low ({stats.af:.2f})",
                    suggestion="Bet and raise more postflop instead of calling",
                )
            )
        elif stats.af > af_range[1] + 1:
            leaks.append(
                Leak(
                    category="tendencies",
                    position=None,
                    stat="af",
                    hero_value=stats.af,
                    optimal_value=af_range[1],
                    deviation=stats.af - af_range[1],
                    severity="minor",
                    description=f"Aggression factor is very high ({stats.af:.2f})",
                    suggestion="Consider more calls with medium-strength hands",
                )
            )

        # WTSD check
        wtsd_range = self.benchmarks.wtsd_range
        if stats.wtsd > wtsd_range[1] + 5:
            leaks.append(
                Leak(
                    category="tendencies",
                    position=None,
                    stat="wtsd",
                    hero_value=stats.wtsd,
                    optimal_value=wtsd_range[1],
                    deviation=stats.wtsd - wtsd_range[1],
                    severity="moderate" if stats.wtsd > 35 else "minor",
                    description=f"Going to showdown too often ({stats.wtsd:.1f}%)",
                    suggestion="Fold more on later streets with weak holdings",
                )
            )
        elif stats.wtsd < wtsd_range[0] - 5:
            leaks.append(
                Leak(
                    category="tendencies",
                    position=None,
                    stat="wtsd",
                    hero_value=stats.wtsd,
                    optimal_value=wtsd_range[0],
                    deviation=stats.wtsd - wtsd_range[0],
                    severity="minor",
                    description=f"Not reaching showdown enough ({stats.wtsd:.1f}%)",
                    suggestion="Call down more with medium-strength hands",
                )
            )

        return leaks

    def _check_position_leaks(
        self, position: str, stats: PlayerStats
    ) -> list[Leak]:
        """Check for leaks in a specific position."""
        leaks = []

        if stats.hands < 50:
            # Not enough sample for position-specific leaks
            return leaks

        pos_bench = self.benchmarks.by_position.get(position)
        if not pos_bench:
            return leaks

        # RFI check (for non-BB positions)
        if position != "BB":
            target_rfi = pos_bench.rfi
            if self.use_adjustments:
                target_rfi = MICROSTAKES_ADJUSTMENTS.get_adjusted_rfi(
                    position, target_rfi
                )

            # Use PFR as proxy for RFI in aggregate stats
            if stats.pfr < target_rfi - 8:
                leaks.append(
                    Leak(
                        category="preflop",
                        position=position,
                        stat="rfi",
                        hero_value=stats.pfr,
                        optimal_value=target_rfi,
                        deviation=stats.pfr - target_rfi,
                        severity="moderate",
                        description=f"Opening too tight from {position}",
                        suggestion=f"Target ~{target_rfi:.0f}% open from {position}",
                    )
                )
            elif stats.pfr > target_rfi + 10:
                leaks.append(
                    Leak(
                        category="preflop",
                        position=position,
                        stat="rfi",
                        hero_value=stats.pfr,
                        optimal_value=target_rfi,
                        deviation=stats.pfr - target_rfi,
                        severity="minor",
                        description=f"Opening too wide from {position}",
                        suggestion=f"Target ~{target_rfi:.0f}% open from {position}",
                    )
                )

        # BB defense check
        if position == "BB":
            # BB should defend roughly 50% vs BTN opens in GTO
            if stats.vpip < 35:
                leaks.append(
                    Leak(
                        category="preflop",
                        position="BB",
                        stat="bb_defense",
                        hero_value=stats.vpip,
                        optimal_value=45.0,
                        deviation=stats.vpip - 45.0,
                        severity="major" if stats.vpip < 30 else "moderate",
                        description="Not defending BB enough",
                        suggestion="Defend ~45-55% vs late position opens",
                    )
                )

        return leaks

    def _identify_strengths(self, hero_stats: HeroStats) -> list[str]:
        """Identify areas where hero is playing well."""
        strengths = []
        stats = hero_stats.overall

        # Check for balanced stats
        vpip_range = self.benchmarks.vpip_range
        if vpip_range[0] <= stats.vpip <= vpip_range[1]:
            strengths.append(f"VPIP is well-balanced at {stats.vpip:.1f}%")

        pfr_range = self.benchmarks.pfr_range
        if pfr_range[0] <= stats.pfr <= pfr_range[1]:
            strengths.append(f"PFR is balanced at {stats.pfr:.1f}%")

        # Good VPIP-PFR gap
        gap = stats.vpip - stats.pfr
        if 3 <= gap <= 6:
            strengths.append("Healthy VPIP-PFR gap (not too passive)")

        # Good aggression
        af_range = self.benchmarks.af_range
        if af_range[0] <= stats.af <= af_range[1]:
            strengths.append(f"Aggression factor is balanced at {stats.af:.2f}")

        # Good 3-bet frequency
        three_bet_range = self.benchmarks.three_bet_range
        if three_bet_range[0] <= stats.three_bet <= three_bet_range[1]:
            strengths.append(f"3-bet frequency is optimal at {stats.three_bet:.1f}%")

        # Good fold to c-bet
        fold_cbet_range = self.benchmarks.fold_to_cbet_range
        if fold_cbet_range[0] <= stats.fold_to_cbet <= fold_cbet_range[1]:
            strengths.append("Defending well vs c-bets")

        return strengths

    def get_position_comparison(self) -> dict[str, dict]:
        """Get hero's stats by position compared to benchmarks."""
        hero_stats = self.analyze()
        comparison = {}

        for position, stats in hero_stats.by_position.items():
            bench = self.benchmarks.by_position.get(position)
            if not bench:
                continue

            comparison[position] = {
                "hands": stats.hands,
                "vpip": {
                    "hero": stats.vpip,
                    "target": bench.rfi if position != "BB" else 45.0,
                    "diff": stats.vpip
                    - (bench.rfi if position != "BB" else 45.0),
                },
                "pfr": {
                    "hero": stats.pfr,
                    "target": bench.rfi,
                    "diff": stats.pfr - bench.rfi,
                },
                "three_bet": {
                    "hero": stats.three_bet,
                    "target": bench.three_bet_vs_open,
                    "diff": stats.three_bet - bench.three_bet_vs_open,
                },
                "cbet": {
                    "hero": stats.cbet,
                    "target": bench.cbet_flop,
                    "diff": stats.cbet - bench.cbet_flop,
                },
            }

        return comparison
