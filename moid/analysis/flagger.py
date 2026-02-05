"""Hand flagger for identifying hands worth reviewing.

This module identifies hands that warrant manual review:
- Large pots where hero made unusual plays
- Hands where hero's action deviated significantly from solver
- Potential mistakes or interesting spots
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Iterator


class FlagReason(Enum):
    """Reasons a hand might be flagged for review."""

    LARGE_POT_LOSS = auto()  # Lost a big pot
    LARGE_POT_WIN = auto()  # Won a big pot (good to review wins too)
    UNUSUAL_LINE = auto()  # Took an uncommon line
    BIG_FOLD = auto()  # Made a large fold
    BIG_CALL = auto()  # Made a large call
    SHOWDOWN_LOSS = auto()  # Lost at showdown
    OVERBET = auto()  # Made or faced an overbet
    ALL_IN_PREFLOP = auto()  # All-in preflop situation
    CHECK_RAISE_FACED = auto()  # Faced a check-raise
    THREE_BET_POT = auto()  # Played a 3-bet pot
    MULTIWAY_POT = auto()  # Multiway pot that went to showdown


@dataclass
class FlaggedHand:
    """A hand flagged for review."""

    hand_id: str
    db_id: int
    timestamp: datetime
    stakes: tuple[float, float]
    board: list[str]
    hero_position: str
    hero_cards: Optional[tuple[str, str]]
    pot_size: float  # In BBs
    hero_result: float  # In BBs
    flags: list[FlagReason]
    priority: int  # 1 = highest, 3 = lowest
    summary: str

    def __repr__(self) -> str:
        flags_str = ", ".join(f.name for f in self.flags)
        return f"FlaggedHand({self.hand_id}, pot={self.pot_size:.1f}bb, {flags_str})"


@dataclass
class FlagCriteria:
    """Criteria for flagging hands."""

    # Pot size thresholds (in BBs)
    large_pot_threshold: float = 30.0  # Flag pots over this size
    very_large_pot_threshold: float = 60.0  # High priority flag

    # Result thresholds (in BBs)
    big_loss_threshold: float = -20.0  # Flag losses bigger than this
    big_win_threshold: float = 25.0  # Flag wins bigger than this

    # Action thresholds
    big_fold_threshold: float = 15.0  # Folding after putting in this much
    big_call_threshold: float = 20.0  # Calling this much on river

    # Overbet threshold
    overbet_threshold: float = 1.2  # Bet > 120% pot


class HandFlagger:
    """
    Identify hands worth reviewing.

    Scans the database for hands with characteristics that
    suggest they're worth manual review.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        criteria: Optional[FlagCriteria] = None,
    ):
        """
        Initialize hand flagger.

        Args:
            conn: Database connection
            criteria: Flagging criteria (uses defaults if None)
        """
        self.conn = conn
        self.criteria = criteria or FlagCriteria()

    def flag_hands(
        self,
        limit: int = 50,
        min_pot: Optional[float] = None,
        only_losses: bool = False,
        since: Optional[datetime] = None,
    ) -> list[FlaggedHand]:
        """
        Get flagged hands for review.

        Args:
            limit: Maximum number of hands to return
            min_pot: Minimum pot size to consider
            only_losses: Only flag hands where hero lost
            since: Only flag hands since this date

        Returns:
            List of FlaggedHand objects sorted by priority
        """
        flagged = []

        # Get large pot hands first
        large_pot_hands = self._get_large_pot_hands(min_pot, since)
        for hand_info in large_pot_hands:
            flags = self._compute_flags(hand_info)
            if flags:
                if only_losses and hand_info["result"] >= 0:
                    continue

                priority = self._compute_priority(hand_info, flags)
                summary = self._generate_summary(hand_info, flags)

                flagged.append(
                    FlaggedHand(
                        hand_id=hand_info["hand_id"],
                        db_id=hand_info["db_id"],
                        timestamp=hand_info["timestamp"],
                        stakes=hand_info["stakes"],
                        board=hand_info["board"],
                        hero_position=hand_info["position"],
                        hero_cards=hand_info["hole_cards"],
                        pot_size=hand_info["pot"],
                        hero_result=hand_info["result"],
                        flags=flags,
                        priority=priority,
                        summary=summary,
                    )
                )

        # Sort by priority then by pot size
        flagged.sort(key=lambda h: (h.priority, -h.pot_size))

        return flagged[:limit]

    def _get_large_pot_hands(
        self,
        min_pot: Optional[float] = None,
        since: Optional[datetime] = None,
    ) -> Iterator[dict]:
        """Get hands with large pots where hero was involved."""
        min_pot = min_pot or self.criteria.large_pot_threshold

        # Only include hands where hero actually participated
        # (didn't just fold preflop without putting money in)
        query = """
            SELECT
                h.id as db_id,
                h.hand_id,
                h.timestamp,
                h.sb,
                h.bb,
                h.board,
                h.total_pot,
                p.position,
                p.hole_cards,
                p.result,
                p.showed_cards,
                p.is_winner,
                (SELECT COUNT(*) FROM actions a
                 WHERE a.hand_id = h.id
                 AND a.position = p.position
                 AND a.action_type IN ('CALL', 'RAISE', 'BET', 'ALL_IN')
                ) as hero_actions
            FROM hands h
            JOIN players p ON h.id = p.hand_id
            WHERE p.is_hero = 1
            AND h.total_pot >= ?
            AND (
                -- Hero put money in voluntarily (not just blinds)
                EXISTS (
                    SELECT 1 FROM actions a
                    WHERE a.hand_id = h.id
                    AND a.position = p.position
                    AND a.action_type IN ('CALL', 'RAISE', 'BET', 'ALL_IN')
                )
                -- Or hero had a significant result (win/loss)
                OR ABS(p.result) >= 5.0
            )
        """
        params = [min_pot]

        if since:
            query += " AND h.timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY ABS(p.result) DESC, h.total_pot DESC"

        cursor = self.conn.execute(query, params)

        for row in cursor:
            board = row["board"].split() if row["board"] else []
            hole_cards = None
            if row["hole_cards"]:
                cards = row["hole_cards"].split()
                if len(cards) == 2:
                    hole_cards = (cards[0], cards[1])

            yield {
                "db_id": row["db_id"],
                "hand_id": row["hand_id"],
                "timestamp": datetime.fromisoformat(row["timestamp"]),
                "stakes": (row["sb"], row["bb"]),
                "board": board,
                "pot": row["total_pot"],
                "position": row["position"],
                "hole_cards": hole_cards,
                "result": row["result"] or 0.0,
                "showed_cards": bool(row["showed_cards"]),
                "is_winner": bool(row["is_winner"]),
            }

    def _compute_flags(self, hand_info: dict) -> list[FlagReason]:
        """Compute which flags apply to a hand."""
        flags = []
        result = hand_info["result"]
        pot = hand_info["pot"]

        # Big result flags (most important - based on actual hero result)
        if result <= self.criteria.big_loss_threshold:
            flags.append(FlagReason.LARGE_POT_LOSS)
        elif result >= self.criteria.big_win_threshold:
            flags.append(FlagReason.LARGE_POT_WIN)
        # Large pot but modest result - only flag if truly significant pot
        elif pot >= self.criteria.very_large_pot_threshold:
            if result < -5:  # Lost something meaningful
                flags.append(FlagReason.LARGE_POT_LOSS)
            elif result > 5:  # Won something meaningful
                flags.append(FlagReason.LARGE_POT_WIN)

        # Showdown loss
        if hand_info["showed_cards"] and not hand_info["is_winner"]:
            flags.append(FlagReason.SHOWDOWN_LOSS)

        # Get action-based flags
        action_flags = self._get_action_flags(hand_info["db_id"], hand_info["position"])
        flags.extend(action_flags)

        return flags

    def _get_action_flags(self, db_id: int, hero_position: str) -> list[FlagReason]:
        """Get flags based on actions taken in the hand."""
        flags = []

        # Check for 3-bet pot
        query = """
            SELECT COUNT(*) as raise_count
            FROM actions
            WHERE hand_id = ? AND street = 'PREFLOP' AND action_type = 'RAISE'
        """
        cursor = self.conn.execute(query, (db_id,))
        if cursor.fetchone()[0] >= 2:
            flags.append(FlagReason.THREE_BET_POT)

        # Check for all-in preflop
        query = """
            SELECT COUNT(*) FROM actions
            WHERE hand_id = ? AND street = 'PREFLOP' AND is_all_in = 1
        """
        cursor = self.conn.execute(query, (db_id,))
        if cursor.fetchone()[0] > 0:
            flags.append(FlagReason.ALL_IN_PREFLOP)

        # Check for check-raise faced by hero
        query = """
            SELECT COUNT(*) FROM actions a1
            WHERE a1.hand_id = ?
            AND a1.position = ?
            AND a1.action_type IN ('BET', 'RAISE')
            AND EXISTS (
                SELECT 1 FROM actions a2
                WHERE a2.hand_id = a1.hand_id
                AND a2.street = a1.street
                AND a2.action_type = 'RAISE'
                AND a2.position != ?
                AND a2.action_order > a1.action_order
            )
        """
        cursor = self.conn.execute(query, (db_id, hero_position, hero_position))
        if cursor.fetchone()[0] > 0:
            flags.append(FlagReason.CHECK_RAISE_FACED)

        # Check for big folds (folded after investing significant amount)
        # This is approximated by looking at pots where hero folded but had invested
        query = """
            SELECT SUM(amount) as invested
            FROM actions
            WHERE hand_id = ? AND position = ?
            AND action_type IN ('CALL', 'RAISE', 'BET', 'ALL_IN')
        """
        cursor = self.conn.execute(query, (db_id, hero_position))
        row = cursor.fetchone()
        invested = row[0] or 0

        # Check if hero folded
        query = """
            SELECT COUNT(*) FROM actions
            WHERE hand_id = ? AND position = ? AND action_type = 'FOLD'
        """
        cursor = self.conn.execute(query, (db_id, hero_position))
        if cursor.fetchone()[0] > 0 and invested >= self.criteria.big_fold_threshold:
            flags.append(FlagReason.BIG_FOLD)

        return flags

    def _compute_priority(
        self, hand_info: dict, flags: list[FlagReason]
    ) -> int:
        """Compute priority (1=highest, 3=lowest)."""
        pot = hand_info["pot"]
        result = hand_info["result"]

        # Very large pots are always high priority
        if pot >= self.criteria.very_large_pot_threshold:
            return 1

        # Big losses are high priority
        if result <= self.criteria.big_loss_threshold * 1.5:
            return 1

        # Interesting situations
        if FlagReason.THREE_BET_POT in flags or FlagReason.ALL_IN_PREFLOP in flags:
            return 2

        # Showdown losses are medium priority
        if FlagReason.SHOWDOWN_LOSS in flags:
            return 2

        return 3

    def _generate_summary(
        self, hand_info: dict, flags: list[FlagReason]
    ) -> str:
        """Generate a human-readable summary of the hand."""
        parts = []

        # Position and cards
        pos = hand_info["position"]
        cards = ""
        if hand_info["hole_cards"]:
            cards = f" ({hand_info['hole_cards'][0]}{hand_info['hole_cards'][1]})"
        parts.append(f"{pos}{cards}")

        # Pot and result
        pot = hand_info["pot"]
        result = hand_info["result"]
        result_str = f"+{result:.1f}" if result >= 0 else f"{result:.1f}"
        parts.append(f"{pot:.1f}bb pot, {result_str}bb")

        # Key flags
        flag_descriptions = {
            FlagReason.LARGE_POT_LOSS: "big loss",
            FlagReason.LARGE_POT_WIN: "big win",
            FlagReason.THREE_BET_POT: "3-bet pot",
            FlagReason.ALL_IN_PREFLOP: "all-in preflop",
            FlagReason.SHOWDOWN_LOSS: "lost at showdown",
            FlagReason.BIG_FOLD: "big fold",
            FlagReason.CHECK_RAISE_FACED: "faced check-raise",
        }
        flag_strs = [
            flag_descriptions[f] for f in flags if f in flag_descriptions
        ]
        if flag_strs:
            parts.append(", ".join(flag_strs))

        return " | ".join(parts)

    def get_stats(self) -> dict:
        """Get overall flagging statistics."""
        query = """
            SELECT
                COUNT(*) as total_hands,
                SUM(CASE WHEN total_pot >= ? THEN 1 ELSE 0 END) as large_pots,
                SUM(CASE WHEN p.result < ? THEN 1 ELSE 0 END) as big_losses
            FROM hands h
            JOIN players p ON h.id = p.hand_id
            WHERE p.is_hero = 1
        """
        cursor = self.conn.execute(
            query,
            (
                self.criteria.large_pot_threshold,
                self.criteria.big_loss_threshold,
            ),
        )
        row = cursor.fetchone()

        return {
            "total_hero_hands": row[0] or 0,
            "large_pot_hands": row[1] or 0,
            "big_loss_hands": row[2] or 0,
        }

    def flag_by_criteria(
        self,
        criteria_type: str,
        limit: int = 20,
    ) -> list[FlaggedHand]:
        """
        Get hands flagged by specific criteria.

        Args:
            criteria_type: One of "losses", "3bet_pots", "showdowns", "big_pots"
            limit: Maximum hands to return

        Returns:
            List of flagged hands
        """
        if criteria_type == "losses":
            return self.flag_hands(limit=limit, only_losses=True)
        elif criteria_type == "3bet_pots":
            return self._flag_by_action("THREE_BET_POT", limit)
        elif criteria_type == "showdowns":
            return self._flag_showdown_losses(limit)
        elif criteria_type == "big_pots":
            return self.flag_hands(
                limit=limit,
                min_pot=self.criteria.very_large_pot_threshold,
            )
        else:
            return self.flag_hands(limit=limit)

    def _flag_showdown_losses(self, limit: int) -> list[FlaggedHand]:
        """Get hands where hero lost at showdown."""
        query = """
            SELECT
                h.id as db_id,
                h.hand_id,
                h.timestamp,
                h.sb,
                h.bb,
                h.board,
                h.total_pot,
                p.position,
                p.hole_cards,
                p.result
            FROM hands h
            JOIN players p ON h.id = p.hand_id
            WHERE p.is_hero = 1
            AND p.showed_cards = 1
            AND p.is_winner = 0
            ORDER BY h.total_pot DESC
            LIMIT ?
        """
        cursor = self.conn.execute(query, (limit,))
        flagged = []

        for row in cursor:
            board = row["board"].split() if row["board"] else []
            hole_cards = None
            if row["hole_cards"]:
                cards = row["hole_cards"].split()
                if len(cards) == 2:
                    hole_cards = (cards[0], cards[1])

            hand_info = {
                "db_id": row["db_id"],
                "hand_id": row["hand_id"],
                "timestamp": datetime.fromisoformat(row["timestamp"]),
                "stakes": (row["sb"], row["bb"]),
                "board": board,
                "pot": row["total_pot"],
                "position": row["position"],
                "hole_cards": hole_cards,
                "result": row["result"] or 0.0,
                "showed_cards": True,
                "is_winner": False,
            }

            flags = [FlagReason.SHOWDOWN_LOSS]
            if row["total_pot"] >= self.criteria.large_pot_threshold:
                flags.append(FlagReason.LARGE_POT_LOSS)

            flagged.append(
                FlaggedHand(
                    hand_id=row["hand_id"],
                    db_id=row["db_id"],
                    timestamp=hand_info["timestamp"],
                    stakes=hand_info["stakes"],
                    board=board,
                    hero_position=row["position"],
                    hero_cards=hole_cards,
                    pot_size=row["total_pot"],
                    hero_result=row["result"] or 0.0,
                    flags=flags,
                    priority=2,
                    summary=self._generate_summary(hand_info, flags),
                )
            )

        return flagged

    def _flag_by_action(self, flag_name: str, limit: int) -> list[FlaggedHand]:
        """Get hands with a specific flag type."""
        all_flagged = self.flag_hands(limit=limit * 3)
        target_flag = FlagReason[flag_name]
        filtered = [h for h in all_flagged if target_flag in h.flags]
        return filtered[:limit]
