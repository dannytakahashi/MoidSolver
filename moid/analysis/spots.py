"""Spot analysis for specific poker situations.

This module provides detailed analysis of hero's play in specific
spot types, comparing frequencies to optimal play.
"""

import sqlite3
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .benchmarks import GTOBenchmarks, GTO_BENCHMARKS, classify_board_texture


class SpotType(Enum):
    """Types of spots to analyze."""

    # Preflop spots
    RFI = auto()  # Raise first in opportunity
    FACING_RFI = auto()  # Facing an open raise
    FACING_3BET = auto()  # Our open faced a 3-bet
    VS_3BET_CALLER = auto()  # We 3-bet and got called
    SQUEEZE = auto()  # Facing open and call(s)
    BLIND_DEFENSE = auto()  # Defending BB vs raise

    # Postflop spots
    CBET_OPPORTUNITY = auto()  # We were PFR and can c-bet
    FACING_CBET = auto()  # We called PF and face c-bet
    CBET_IP = auto()  # C-betting in position
    CBET_OOP = auto()  # C-betting out of position
    PROBE_BET = auto()  # Betting when PFR checks
    DELAYED_CBET = auto()  # C-betting turn after checking flop

    # Specific situations
    DONK_BET = auto()  # Leading into aggressor
    CHECK_RAISE = auto()  # Check-raising opportunity
    FACING_CHECK_RAISE = auto()  # We bet and got raised
    BARREL_TURN = auto()  # C-bet turn after c-betting flop
    BARREL_RIVER = auto()  # C-bet river after c-betting turn


@dataclass
class SpotStats:
    """Statistics for a specific spot type."""

    spot_type: SpotType
    position: Optional[str] = None

    # Sample sizes
    opportunities: int = 0
    actions_taken: int = 0

    # Action frequencies
    fold_pct: float = 0.0
    check_pct: float = 0.0
    call_pct: float = 0.0
    bet_pct: float = 0.0  # Includes raises
    all_in_pct: float = 0.0

    # Optimal frequencies (from benchmarks)
    optimal_fold: Optional[float] = None
    optimal_call: Optional[float] = None
    optimal_bet: Optional[float] = None

    # EV metrics (if available)
    avg_result: float = 0.0  # Average result in BBs when in this spot

    def __repr__(self) -> str:
        pos = f"[{self.position}]" if self.position else ""
        return f"SpotStats({self.spot_type.name}{pos}, n={self.opportunities})"

    @property
    def is_exploitable(self) -> bool:
        """Check if hero's frequencies are significantly off optimal."""
        if self.optimal_fold is not None:
            if abs(self.fold_pct - self.optimal_fold) > 15:
                return True
        if self.optimal_bet is not None:
            if abs(self.bet_pct - self.optimal_bet) > 15:
                return True
        return False

    @property
    def deviation_summary(self) -> str:
        """Get summary of how hero deviates from optimal."""
        deviations = []
        if self.optimal_fold is not None:
            diff = self.fold_pct - self.optimal_fold
            if abs(diff) > 5:
                direction = "too much" if diff > 0 else "not enough"
                deviations.append(f"Folding {direction} ({diff:+.1f}%)")
        if self.optimal_bet is not None:
            diff = self.bet_pct - self.optimal_bet
            if abs(diff) > 5:
                direction = "too much" if diff > 0 else "not enough"
                deviations.append(f"Betting {direction} ({diff:+.1f}%)")
        return "; ".join(deviations) if deviations else "Near optimal"


@dataclass
class SpotBreakdown:
    """Detailed breakdown of a spot by various factors."""

    spot_type: SpotType
    overall: SpotStats
    by_position: dict[str, SpotStats] = field(default_factory=dict)
    by_board_texture: dict[str, SpotStats] = field(default_factory=dict)
    by_stack_depth: dict[str, SpotStats] = field(default_factory=dict)


class SpotAnalyzer:
    """
    Analyze hero's play in specific spots.

    Provides granular analysis of specific situations to
    identify exactly where leaks occur.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        benchmarks: Optional[GTOBenchmarks] = None,
    ):
        """
        Initialize spot analyzer.

        Args:
            conn: Database connection
            benchmarks: GTO benchmarks for comparison
        """
        self.conn = conn
        self.benchmarks = benchmarks or GTO_BENCHMARKS

    def analyze_spot(
        self,
        spot_type: SpotType,
        position: Optional[str] = None,
    ) -> SpotStats:
        """
        Analyze a specific spot type.

        Args:
            spot_type: Type of spot to analyze
            position: Optional position filter

        Returns:
            SpotStats with hero's frequencies and comparisons
        """
        if spot_type == SpotType.RFI:
            return self._analyze_rfi(position)
        elif spot_type == SpotType.FACING_RFI:
            return self._analyze_facing_rfi(position)
        elif spot_type == SpotType.FACING_3BET:
            return self._analyze_facing_3bet(position)
        elif spot_type == SpotType.BLIND_DEFENSE:
            return self._analyze_blind_defense()
        elif spot_type == SpotType.CBET_OPPORTUNITY:
            return self._analyze_cbet(position)
        elif spot_type == SpotType.FACING_CBET:
            return self._analyze_facing_cbet(position)
        elif spot_type == SpotType.BARREL_TURN:
            return self._analyze_barrel_turn(position)
        elif spot_type == SpotType.CHECK_RAISE:
            return self._analyze_check_raise(position)
        else:
            return SpotStats(spot_type=spot_type, position=position)

    def get_spot_breakdown(self, spot_type: SpotType) -> SpotBreakdown:
        """
        Get detailed breakdown of a spot.

        Args:
            spot_type: Type of spot to analyze

        Returns:
            SpotBreakdown with position and texture breakdowns
        """
        breakdown = SpotBreakdown(
            spot_type=spot_type,
            overall=self.analyze_spot(spot_type),
        )

        # Position breakdown
        positions = ["UTG", "UTG1", "CO", "BTN", "SB", "BB"]
        for pos in positions:
            stats = self.analyze_spot(spot_type, position=pos)
            if stats.opportunities > 0:
                breakdown.by_position[pos] = stats

        return breakdown

    def _analyze_rfi(self, position: Optional[str] = None) -> SpotStats:
        """Analyze raise first in opportunities."""
        stats = SpotStats(spot_type=SpotType.RFI, position=position)

        filters = ["p.is_hero = 1"]
        params = []

        if position:
            filters.append("p.position = ?")
            params.append(position)

        where_clause = " AND ".join(filters)

        # RFI opportunities: hands where hero acted first (no prior voluntary action)
        query = f"""
            WITH hero_hands AS (
                SELECT DISTINCT h.id as hand_id, p.position
                FROM hands h
                JOIN players p ON h.id = p.hand_id
                WHERE {where_clause}
            ),
            rfi_opps AS (
                -- Hands where hero had first action opportunity (no limps/raises before)
                SELECT hh.hand_id, hh.position,
                    (SELECT a.action_type FROM actions a
                     WHERE a.hand_id = hh.hand_id
                     AND a.position = hh.position
                     AND a.street = 'PREFLOP'
                     AND a.action_type NOT IN ('POST_SB', 'POST_BB', 'POST_ANTE')
                     ORDER BY a.action_order LIMIT 1) as hero_action
                FROM hero_hands hh
                WHERE NOT EXISTS (
                    SELECT 1 FROM actions a
                    WHERE a.hand_id = hh.hand_id
                    AND a.street = 'PREFLOP'
                    AND a.position != hh.position
                    AND a.action_type IN ('CALL', 'RAISE', 'BET', 'ALL_IN')
                    AND a.action_order < (
                        SELECT MIN(a2.action_order) FROM actions a2
                        WHERE a2.hand_id = hh.hand_id
                        AND a2.position = hh.position
                        AND a2.street = 'PREFLOP'
                        AND a2.action_type NOT IN ('POST_SB', 'POST_BB', 'POST_ANTE')
                    )
                )
            )
            SELECT
                COUNT(*) as opportunities,
                SUM(CASE WHEN hero_action = 'FOLD' THEN 1 ELSE 0 END) as folds,
                SUM(CASE WHEN hero_action IN ('RAISE', 'BET', 'ALL_IN') THEN 1 ELSE 0 END) as raises,
                SUM(CASE WHEN hero_action = 'CALL' THEN 1 ELSE 0 END) as limps
            FROM rfi_opps
        """
        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()

        stats.opportunities = row[0] or 0
        if stats.opportunities > 0:
            stats.fold_pct = (row[1] or 0) / stats.opportunities * 100
            stats.bet_pct = (row[2] or 0) / stats.opportunities * 100
            stats.call_pct = (row[3] or 0) / stats.opportunities * 100

        # Set optimal RFI based on position
        if position and position in self.benchmarks.by_position:
            bench = self.benchmarks.by_position[position]
            stats.optimal_bet = bench.rfi
            stats.optimal_fold = 100.0 - bench.rfi  # Simplified

        return stats

    def _analyze_facing_rfi(self, position: Optional[str] = None) -> SpotStats:
        """Analyze decisions when facing an open raise."""
        stats = SpotStats(spot_type=SpotType.FACING_RFI, position=position)

        filters = ["p.is_hero = 1"]
        params = []

        if position:
            filters.append("p.position = ?")
            params.append(position)

        where_clause = " AND ".join(filters)

        query = f"""
            WITH hero_hands AS (
                SELECT DISTINCT h.id as hand_id, p.position
                FROM hands h
                JOIN players p ON h.id = p.hand_id
                WHERE {where_clause}
            ),
            facing_open AS (
                SELECT hh.hand_id, hh.position,
                    (SELECT a2.action_type FROM actions a2
                     WHERE a2.hand_id = hh.hand_id
                     AND a2.position = hh.position
                     AND a2.street = 'PREFLOP'
                     AND a2.action_type NOT IN ('POST_SB', 'POST_BB')
                     ORDER BY a2.action_order LIMIT 1) as hero_action
                FROM hero_hands hh
                WHERE EXISTS (
                    SELECT 1 FROM actions a
                    WHERE a.hand_id = hh.hand_id
                    AND a.street = 'PREFLOP'
                    AND a.action_type IN ('RAISE', 'BET')
                    AND a.position != hh.position
                )
            )
            SELECT
                COUNT(*) as opportunities,
                SUM(CASE WHEN hero_action = 'FOLD' THEN 1 ELSE 0 END) as folds,
                SUM(CASE WHEN hero_action = 'CALL' THEN 1 ELSE 0 END) as calls,
                SUM(CASE WHEN hero_action IN ('RAISE', 'ALL_IN') THEN 1 ELSE 0 END) as raises
            FROM facing_open
        """
        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()

        stats.opportunities = row[0] or 0
        if stats.opportunities > 0:
            stats.fold_pct = (row[1] or 0) / stats.opportunities * 100
            stats.call_pct = (row[2] or 0) / stats.opportunities * 100
            stats.bet_pct = (row[3] or 0) / stats.opportunities * 100

        # Set benchmarks
        if position and position in self.benchmarks.by_position:
            bench = self.benchmarks.by_position[position]
            stats.optimal_fold = bench.fold_vs_open
            stats.optimal_call = bench.call_vs_open
            stats.optimal_bet = bench.three_bet_vs_open

        return stats

    def _analyze_facing_3bet(self, position: Optional[str] = None) -> SpotStats:
        """Analyze decisions when our open faces a 3-bet."""
        stats = SpotStats(spot_type=SpotType.FACING_3BET, position=position)

        filters = ["p.is_hero = 1"]
        params = []

        if position:
            filters.append("p.position = ?")
            params.append(position)

        where_clause = " AND ".join(filters)

        query = f"""
            WITH hero_opens AS (
                SELECT h.id as hand_id, p.position
                FROM hands h
                JOIN players p ON h.id = p.hand_id
                JOIN actions a ON h.id = a.hand_id
                    AND a.position = p.position
                    AND a.street = 'PREFLOP'
                    AND a.action_type IN ('RAISE', 'BET')
                WHERE {where_clause}
            ),
            faced_3bet AS (
                SELECT ho.hand_id, ho.position,
                    (SELECT a2.action_type FROM actions a2
                     WHERE a2.hand_id = ho.hand_id
                     AND a2.position = ho.position
                     AND a2.street = 'PREFLOP'
                     AND a2.action_order > (
                         SELECT MAX(a3.action_order) FROM actions a3
                         WHERE a3.hand_id = ho.hand_id
                         AND a3.street = 'PREFLOP'
                         AND a3.action_type = 'RAISE'
                         AND a3.position != ho.position
                     )
                     ORDER BY a2.action_order LIMIT 1) as hero_response
                FROM hero_opens ho
                WHERE EXISTS (
                    SELECT 1 FROM actions a
                    WHERE a.hand_id = ho.hand_id
                    AND a.street = 'PREFLOP'
                    AND a.action_type = 'RAISE'
                    AND a.position != ho.position
                )
            )
            SELECT
                COUNT(*) as opportunities,
                SUM(CASE WHEN hero_response = 'FOLD' THEN 1 ELSE 0 END) as folds,
                SUM(CASE WHEN hero_response = 'CALL' THEN 1 ELSE 0 END) as calls,
                SUM(CASE WHEN hero_response IN ('RAISE', 'ALL_IN') THEN 1 ELSE 0 END) as raises
            FROM faced_3bet
        """
        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()

        stats.opportunities = row[0] or 0
        if stats.opportunities > 0:
            stats.fold_pct = (row[1] or 0) / stats.opportunities * 100
            stats.call_pct = (row[2] or 0) / stats.opportunities * 100
            stats.bet_pct = (row[3] or 0) / stats.opportunities * 100

        # Set benchmarks
        if position and position in self.benchmarks.by_position:
            bench = self.benchmarks.by_position[position]
            stats.optimal_fold = bench.fold_vs_3bet
            stats.optimal_call = bench.call_vs_3bet
            stats.optimal_bet = bench.four_bet_vs_3bet

        return stats

    def _analyze_blind_defense(self) -> SpotStats:
        """Analyze BB defense vs raises."""
        stats = SpotStats(spot_type=SpotType.BLIND_DEFENSE, position="BB")

        query = """
            WITH bb_hands AS (
                SELECT h.id as hand_id
                FROM hands h
                JOIN players p ON h.id = p.hand_id
                WHERE p.is_hero = 1 AND p.position = 'BB'
            ),
            facing_raise AS (
                SELECT bh.hand_id,
                    (SELECT a2.action_type FROM actions a2
                     WHERE a2.hand_id = bh.hand_id
                     AND a2.position = 'BB'
                     AND a2.street = 'PREFLOP'
                     AND a2.action_type NOT IN ('POST_BB', 'POST_ANTE')
                     ORDER BY a2.action_order LIMIT 1) as hero_action
                FROM bb_hands bh
                WHERE EXISTS (
                    SELECT 1 FROM actions a
                    WHERE a.hand_id = bh.hand_id
                    AND a.street = 'PREFLOP'
                    AND a.action_type IN ('RAISE', 'BET')
                    AND a.position != 'BB'
                )
            )
            SELECT
                COUNT(*) as opportunities,
                SUM(CASE WHEN hero_action = 'FOLD' THEN 1 ELSE 0 END) as folds,
                SUM(CASE WHEN hero_action = 'CALL' THEN 1 ELSE 0 END) as calls,
                SUM(CASE WHEN hero_action IN ('RAISE', 'ALL_IN') THEN 1 ELSE 0 END) as raises
            FROM facing_raise
        """
        cursor = self.conn.execute(query)
        row = cursor.fetchone()

        stats.opportunities = row[0] or 0
        if stats.opportunities > 0:
            stats.fold_pct = (row[1] or 0) / stats.opportunities * 100
            stats.call_pct = (row[2] or 0) / stats.opportunities * 100
            stats.bet_pct = (row[3] or 0) / stats.opportunities * 100

        # BB should defend ~50% vs BTN opens in GTO
        bb_bench = self.benchmarks.by_position.get("BB")
        if bb_bench:
            stats.optimal_fold = bb_bench.fold_vs_open
            stats.optimal_call = bb_bench.call_vs_open
            stats.optimal_bet = bb_bench.three_bet_vs_open

        return stats

    def _analyze_cbet(self, position: Optional[str] = None) -> SpotStats:
        """Analyze c-bet opportunities."""
        stats = SpotStats(spot_type=SpotType.CBET_OPPORTUNITY, position=position)

        filters = ["p.is_hero = 1"]
        params = []

        if position:
            filters.append("p.position = ?")
            params.append(position)

        where_clause = " AND ".join(filters)

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
                AND h.board IS NOT NULL AND h.board != ''
            ),
            cbet_decisions AS (
                SELECT hp.hand_id, hp.position,
                    (SELECT a.action_type FROM actions a
                     WHERE a.hand_id = hp.hand_id
                     AND a.position = hp.position
                     AND a.street = 'FLOP'
                     ORDER BY a.action_order LIMIT 1) as flop_action
                FROM hero_pfr hp
            )
            SELECT
                COUNT(*) as opportunities,
                SUM(CASE WHEN flop_action IN ('CHECK') THEN 1 ELSE 0 END) as checks,
                SUM(CASE WHEN flop_action IN ('BET', 'RAISE', 'ALL_IN') THEN 1 ELSE 0 END) as bets
            FROM cbet_decisions
        """
        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()

        stats.opportunities = row[0] or 0
        if stats.opportunities > 0:
            stats.check_pct = (row[1] or 0) / stats.opportunities * 100
            stats.bet_pct = (row[2] or 0) / stats.opportunities * 100

        # Set benchmarks
        if position and position in self.benchmarks.by_position:
            bench = self.benchmarks.by_position[position]
            stats.optimal_bet = bench.cbet_flop

        return stats

    def _analyze_facing_cbet(self, position: Optional[str] = None) -> SpotStats:
        """Analyze decisions when facing a c-bet."""
        stats = SpotStats(spot_type=SpotType.FACING_CBET, position=position)

        filters = ["p.is_hero = 1"]
        params = []

        if position:
            filters.append("p.position = ?")
            params.append(position)

        where_clause = " AND ".join(filters)

        query = f"""
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
            facing_cbet AS (
                SELECT hcp.hand_id, hcp.position,
                    (SELECT a2.action_type FROM actions a2
                     WHERE a2.hand_id = hcp.hand_id
                     AND a2.position = hcp.position
                     AND a2.street = 'FLOP'
                     ORDER BY a2.action_order LIMIT 1) as hero_response
                FROM hero_called_pf hcp
                WHERE EXISTS (
                    SELECT 1 FROM actions a
                    WHERE a.hand_id = hcp.hand_id
                    AND a.street = 'FLOP'
                    AND a.action_type IN ('BET', 'RAISE')
                    AND a.position != hcp.position
                )
            )
            SELECT
                COUNT(*) as opportunities,
                SUM(CASE WHEN hero_response = 'FOLD' THEN 1 ELSE 0 END) as folds,
                SUM(CASE WHEN hero_response = 'CALL' THEN 1 ELSE 0 END) as calls,
                SUM(CASE WHEN hero_response IN ('RAISE', 'ALL_IN') THEN 1 ELSE 0 END) as raises
            FROM facing_cbet
        """
        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()

        stats.opportunities = row[0] or 0
        if stats.opportunities > 0:
            stats.fold_pct = (row[1] or 0) / stats.opportunities * 100
            stats.call_pct = (row[2] or 0) / stats.opportunities * 100
            stats.bet_pct = (row[3] or 0) / stats.opportunities * 100

        # Set benchmarks
        if position and position in self.benchmarks.by_position:
            bench = self.benchmarks.by_position[position]
            stats.optimal_fold = bench.fold_vs_cbet
            stats.optimal_call = bench.call_vs_cbet
            stats.optimal_bet = bench.raise_vs_cbet

        return stats

    def _analyze_barrel_turn(self, position: Optional[str] = None) -> SpotStats:
        """Analyze turn c-bet after flop c-bet."""
        stats = SpotStats(spot_type=SpotType.BARREL_TURN, position=position)

        filters = ["p.is_hero = 1"]
        params = []

        if position:
            filters.append("p.position = ?")
            params.append(position)

        where_clause = " AND ".join(filters)

        query = f"""
            WITH hero_cbet_flop AS (
                SELECT h.id as hand_id, p.position
                FROM hands h
                JOIN players p ON h.id = p.hand_id
                JOIN actions a ON h.id = a.hand_id
                    AND a.position = p.position
                    AND a.street = 'FLOP'
                    AND a.action_type IN ('BET', 'RAISE')
                WHERE {where_clause}
                AND LENGTH(h.board) >= 8  -- At least turn card present
            ),
            turn_decisions AS (
                SELECT hcf.hand_id, hcf.position,
                    (SELECT a.action_type FROM actions a
                     WHERE a.hand_id = hcf.hand_id
                     AND a.position = hcf.position
                     AND a.street = 'TURN'
                     ORDER BY a.action_order LIMIT 1) as turn_action
                FROM hero_cbet_flop hcf
            )
            SELECT
                COUNT(*) as opportunities,
                SUM(CASE WHEN turn_action = 'CHECK' THEN 1 ELSE 0 END) as checks,
                SUM(CASE WHEN turn_action IN ('BET', 'RAISE', 'ALL_IN') THEN 1 ELSE 0 END) as bets
            FROM turn_decisions
        """
        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()

        stats.opportunities = row[0] or 0
        if stats.opportunities > 0:
            stats.check_pct = (row[1] or 0) / stats.opportunities * 100
            stats.bet_pct = (row[2] or 0) / stats.opportunities * 100

        # Set benchmarks
        if position and position in self.benchmarks.by_position:
            bench = self.benchmarks.by_position[position]
            stats.optimal_bet = bench.cbet_turn

        return stats

    def _analyze_check_raise(self, position: Optional[str] = None) -> SpotStats:
        """Analyze check-raise frequency."""
        stats = SpotStats(spot_type=SpotType.CHECK_RAISE, position=position)

        filters = ["p.is_hero = 1"]
        params = []

        if position:
            filters.append("p.position = ?")
            params.append(position)

        where_clause = " AND ".join(filters)

        # Check-raise opportunities: hero checked, then villain bet
        query = f"""
            WITH hero_checks AS (
                SELECT h.id as hand_id, p.position, a.street
                FROM hands h
                JOIN players p ON h.id = p.hand_id
                JOIN actions a ON h.id = a.hand_id
                    AND a.position = p.position
                    AND a.action_type = 'CHECK'
                WHERE {where_clause}
            ),
            facing_bet AS (
                SELECT hc.hand_id, hc.position, hc.street,
                    (SELECT a2.action_type FROM actions a2
                     WHERE a2.hand_id = hc.hand_id
                     AND a2.position = hc.position
                     AND a2.street = hc.street
                     AND a2.action_order > (
                         SELECT MAX(a3.action_order) FROM actions a3
                         WHERE a3.hand_id = hc.hand_id
                         AND a3.street = hc.street
                         AND a3.action_type IN ('BET', 'RAISE')
                         AND a3.position != hc.position
                     )
                     ORDER BY a2.action_order LIMIT 1) as hero_response
                FROM hero_checks hc
                WHERE EXISTS (
                    SELECT 1 FROM actions a
                    WHERE a.hand_id = hc.hand_id
                    AND a.street = hc.street
                    AND a.action_type IN ('BET', 'RAISE')
                    AND a.position != hc.position
                )
            )
            SELECT
                COUNT(*) as opportunities,
                SUM(CASE WHEN hero_response = 'FOLD' THEN 1 ELSE 0 END) as folds,
                SUM(CASE WHEN hero_response = 'CALL' THEN 1 ELSE 0 END) as calls,
                SUM(CASE WHEN hero_response IN ('RAISE', 'ALL_IN') THEN 1 ELSE 0 END) as raises
            FROM facing_bet
        """
        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()

        stats.opportunities = row[0] or 0
        if stats.opportunities > 0:
            stats.fold_pct = (row[1] or 0) / stats.opportunities * 100
            stats.call_pct = (row[2] or 0) / stats.opportunities * 100
            stats.bet_pct = (row[3] or 0) / stats.opportunities * 100

        # Check-raise should be around 8-12% in most spots
        stats.optimal_bet = 10.0

        return stats

    def get_spot_summary(self) -> dict[str, SpotStats]:
        """Get summary of key spots for quick review."""
        spots = [
            SpotType.RFI,
            SpotType.FACING_3BET,
            SpotType.BLIND_DEFENSE,
            SpotType.CBET_OPPORTUNITY,
            SpotType.FACING_CBET,
            SpotType.BARREL_TURN,
        ]

        return {spot.name: self.analyze_spot(spot) for spot in spots}
