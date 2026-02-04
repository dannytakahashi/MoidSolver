"""Data access layer for hand history database."""

import sqlite3
from pathlib import Path
from typing import Iterator, Optional

from moid.parser.models import Action, ActionType, Hand, Player, Position, Street


class HandRepository:
    """Repository for storing and querying poker hands."""

    def __init__(self, conn: sqlite3.Connection):
        """
        Initialize repository with database connection.

        Args:
            conn: SQLite database connection
        """
        self.conn = conn

    def insert_hand(self, hand: Hand) -> int:
        """
        Insert a hand into the database.

        Args:
            hand: Hand object to insert

        Returns:
            Database ID of inserted hand
        """
        cursor = self.conn.execute(
            """
            INSERT INTO hands (
                hand_id, timestamp, sb, bb, table_name, board,
                total_pot, rake, num_players, went_to_showdown, is_heads_up_postflop
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hand.hand_id,
                hand.timestamp.isoformat(),
                hand.stakes[0],
                hand.stakes[1],
                hand.table_name,
                " ".join(hand.board) if hand.board else None,
                hand.total_pot,
                hand.rake,
                hand.num_players,
                hand.went_to_showdown(),
                hand.is_heads_up_postflop(),
            ),
        )

        db_hand_id = cursor.lastrowid

        # Insert players
        for player in hand.players:
            self._insert_player(db_hand_id, player, player.position in hand.winners)

        # Insert actions
        for i, action in enumerate(hand.actions):
            self._insert_action(db_hand_id, action, i)

        return db_hand_id

    def _insert_player(self, hand_id: int, player: Player, is_winner: bool) -> None:
        """Insert a player record."""
        hole_cards = None
        if player.hole_cards:
            hole_cards = f"{player.hole_cards[0]} {player.hole_cards[1]}"

        self.conn.execute(
            """
            INSERT INTO players (
                hand_id, position, stack, hole_cards, result,
                is_hero, showed_cards, is_winner
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hand_id,
                player.position.name,
                player.stack,
                hole_cards,
                player.result,
                player.is_hero,
                player.showed_cards,
                is_winner,
            ),
        )

    def _insert_action(self, hand_id: int, action: Action, order: int) -> None:
        """Insert an action record."""
        self.conn.execute(
            """
            INSERT INTO actions (
                hand_id, position, street, action_type, amount, is_all_in, action_order
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hand_id,
                action.position.name,
                action.street.name,
                action.action_type.name,
                action.amount,
                action.is_all_in,
                order,
            ),
        )

    def insert_hands(self, hands: Iterator[Hand], batch_size: int = 1000) -> int:
        """
        Insert multiple hands efficiently using transactions.

        Args:
            hands: Iterator of Hand objects
            batch_size: Number of hands per transaction

        Returns:
            Total number of hands actually inserted (excludes duplicates)
        """
        count = 0
        batch = []

        for hand in hands:
            batch.append(hand)
            if len(batch) >= batch_size:
                count += self._insert_batch(batch)
                batch = []

        if batch:
            count += self._insert_batch(batch)

        return count

    def _insert_batch(self, hands: list[Hand]) -> int:
        """
        Insert a batch of hands in a single transaction.

        Returns:
            Number of hands actually inserted
        """
        inserted = 0
        try:
            for hand in hands:
                try:
                    self.insert_hand(hand)
                    inserted += 1
                except sqlite3.IntegrityError:
                    # Hand already exists, skip
                    continue
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return inserted

    def get_hand(self, hand_id: str) -> Optional[Hand]:
        """
        Retrieve a hand by its original hand ID.

        Args:
            hand_id: Original hand ID from poker site

        Returns:
            Hand object or None if not found
        """
        cursor = self.conn.execute(
            "SELECT * FROM hands WHERE hand_id = ?", (hand_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        return self._row_to_hand(row)

    def _row_to_hand(self, row: sqlite3.Row) -> Hand:
        """Convert a database row to Hand object."""
        from datetime import datetime

        db_id = row["id"]

        # Parse board
        board = []
        if row["board"]:
            board = row["board"].split()

        hand = Hand(
            hand_id=row["hand_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            stakes=(row["sb"], row["bb"]),
            table_name=row["table_name"] or "",
            board=board,
            total_pot=row["total_pot"] or 0.0,
            rake=row["rake"] or 0.0,
        )

        # Load players
        players_cursor = self.conn.execute(
            "SELECT * FROM players WHERE hand_id = ?", (db_id,)
        )
        for p_row in players_cursor:
            hole_cards = None
            if p_row["hole_cards"]:
                cards = p_row["hole_cards"].split()
                if len(cards) == 2:
                    hole_cards = (cards[0], cards[1])

            player = Player(
                position=Position[p_row["position"]],
                stack=p_row["stack"],
                hole_cards=hole_cards,
                result=p_row["result"],
                is_hero=bool(p_row["is_hero"]),
                showed_cards=bool(p_row["showed_cards"]),
            )
            hand.players.append(player)
            if p_row["is_winner"]:
                hand.winners.append(player.position)

        # Load actions
        actions_cursor = self.conn.execute(
            "SELECT * FROM actions WHERE hand_id = ? ORDER BY action_order",
            (db_id,),
        )
        for a_row in actions_cursor:
            action = Action(
                position=Position[a_row["position"]],
                street=Street[a_row["street"]],
                action_type=ActionType[a_row["action_type"]],
                amount=a_row["amount"],
                is_all_in=bool(a_row["is_all_in"]),
            )
            hand.actions.append(action)

        return hand

    def count_hands(self) -> int:
        """Get total number of hands in database."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM hands")
        return cursor.fetchone()[0]

    def count_actions(
        self,
        position: Optional[str] = None,
        street: Optional[str] = None,
        action_type: Optional[str] = None,
    ) -> int:
        """
        Count actions with optional filters.

        Args:
            position: Filter by position (e.g., "BTN")
            street: Filter by street (e.g., "PREFLOP")
            action_type: Filter by action type (e.g., "RAISE")

        Returns:
            Count of matching actions
        """
        query = "SELECT COUNT(*) FROM actions WHERE 1=1"
        params = []

        if position:
            query += " AND position = ?"
            params.append(position)
        if street:
            query += " AND street = ?"
            params.append(street)
        if action_type:
            query += " AND action_type = ?"
            params.append(action_type)

        cursor = self.conn.execute(query, params)
        return cursor.fetchone()[0]

    def get_position_action_counts(
        self, position: str, street: str = "PREFLOP"
    ) -> dict[str, int]:
        """
        Get action type counts for a position on a street.

        Args:
            position: Position (e.g., "BTN")
            street: Street name (default: "PREFLOP")

        Returns:
            Dictionary mapping action types to counts
        """
        cursor = self.conn.execute(
            """
            SELECT action_type, COUNT(*) as count
            FROM actions
            WHERE position = ? AND street = ?
            GROUP BY action_type
            """,
            (position, street),
        )
        return {row["action_type"]: row["count"] for row in cursor}

    def get_hands_by_spot(
        self,
        hero_position: str,
        villain_position: str,
        street: str = "FLOP",
        heads_up: bool = True,
    ) -> Iterator[Hand]:
        """
        Get hands matching a specific spot.

        Args:
            hero_position: Hero's position
            villain_position: Villain's position
            street: Street to filter (default: FLOP)
            heads_up: Only heads-up pots (default: True)

        Yields:
            Matching Hand objects
        """
        query = """
            SELECT DISTINCT h.*
            FROM hands h
            JOIN players p1 ON h.id = p1.hand_id AND p1.position = ?
            JOIN players p2 ON h.id = p2.hand_id AND p2.position = ?
            JOIN actions a ON h.id = a.hand_id AND a.street = ?
        """
        params = [hero_position, villain_position, street]

        if heads_up:
            query += " WHERE h.is_heads_up_postflop = 1"

        cursor = self.conn.execute(query, params)
        for row in cursor:
            yield self._row_to_hand(row)

    def get_stack_distribution(self, position: Optional[str] = None) -> dict[str, int]:
        """
        Get distribution of stack sizes.

        Args:
            position: Optional position filter

        Returns:
            Dictionary with stack categories and counts
        """
        query = """
            SELECT
                CASE
                    WHEN stack < 50 THEN 'short'
                    WHEN stack <= 100 THEN 'medium'
                    ELSE 'deep'
                END as stack_category,
                COUNT(*) as count
            FROM players
        """
        params = []

        if position:
            query += " WHERE position = ?"
            params.append(position)

        query += " GROUP BY stack_category"

        cursor = self.conn.execute(query, params)
        return {row["stack_category"]: row["count"] for row in cursor}

    def vacuum(self) -> None:
        """Optimize database by running VACUUM."""
        self.conn.execute("VACUUM")

    def analyze(self) -> None:
        """Update statistics for query optimizer."""
        self.conn.execute("ANALYZE")
