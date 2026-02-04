"""Tests for database layer."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from moid.db.schema import create_database, get_connection, get_schema_version
from moid.db.repository import HandRepository
from moid.parser.models import Action, ActionType, Hand, Player, Position, Street


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = create_database(db_path)
    yield conn, db_path

    conn.close()
    db_path.unlink()


@pytest.fixture
def sample_hand():
    """Create a sample hand for testing."""
    return Hand(
        hand_id="TEST123",
        timestamp=datetime(2024, 1, 15, 14, 30, 0),
        stakes=(0.02, 0.05),
        table_name="12345",
        board=["As", "Kh", "7d", "2c", "9s"],
        total_pot=10.0,
        rake=0.50,
        players=[
            Player(Position.BTN, stack=100.0, hole_cards=("Ac", "Kd"), result=5.0),
            Player(Position.BB, stack=95.0, is_hero=True),
        ],
        actions=[
            Action(Position.BTN, Street.PREFLOP, ActionType.RAISE, 3.0),
            Action(Position.BB, Street.PREFLOP, ActionType.CALL, 2.0),
            Action(Position.BB, Street.FLOP, ActionType.CHECK),
            Action(Position.BTN, Street.FLOP, ActionType.BET, 4.0),
            Action(Position.BB, Street.FLOP, ActionType.FOLD),
        ],
        winners=[Position.BTN],
    )


class TestSchema:
    def test_create_database(self, temp_db):
        conn, db_path = temp_db

        # Check tables exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor}

        assert "hands" in tables
        assert "players" in tables
        assert "actions" in tables
        assert "schema_version" in tables

    def test_schema_version(self, temp_db):
        conn, _ = temp_db
        version = get_schema_version(conn)
        assert version == 1

    def test_create_database_force(self, temp_db):
        _, db_path = temp_db

        # Create again with force
        conn2 = create_database(db_path, force=True)
        version = get_schema_version(conn2)
        assert version == 1
        conn2.close()


class TestHandRepository:
    def test_insert_hand(self, temp_db, sample_hand):
        conn, _ = temp_db
        repo = HandRepository(conn)

        db_id = repo.insert_hand(sample_hand)
        conn.commit()

        assert db_id > 0
        assert repo.count_hands() == 1

    def test_insert_duplicate_hand(self, temp_db, sample_hand):
        conn, _ = temp_db
        repo = HandRepository(conn)

        repo.insert_hand(sample_hand)
        conn.commit()

        # Inserting same hand_id should raise IntegrityError
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            repo.insert_hand(sample_hand)

    def test_get_hand(self, temp_db, sample_hand):
        conn, _ = temp_db
        repo = HandRepository(conn)

        repo.insert_hand(sample_hand)
        conn.commit()

        retrieved = repo.get_hand("TEST123")

        assert retrieved is not None
        assert retrieved.hand_id == sample_hand.hand_id
        assert retrieved.stakes == sample_hand.stakes
        assert len(retrieved.players) == len(sample_hand.players)
        assert len(retrieved.actions) == len(sample_hand.actions)
        assert retrieved.board == sample_hand.board

    def test_get_hand_not_found(self, temp_db):
        conn, _ = temp_db
        repo = HandRepository(conn)

        result = repo.get_hand("NONEXISTENT")
        assert result is None

    def test_insert_hands_batch(self, temp_db):
        conn, _ = temp_db
        repo = HandRepository(conn)

        hands = [
            Hand(
                hand_id=f"BATCH{i}",
                timestamp=datetime.now(),
                stakes=(0.02, 0.05),
            )
            for i in range(100)
        ]

        count = repo.insert_hands(iter(hands), batch_size=25)

        assert count == 100
        assert repo.count_hands() == 100

    def test_count_actions(self, temp_db, sample_hand):
        conn, _ = temp_db
        repo = HandRepository(conn)

        repo.insert_hand(sample_hand)
        conn.commit()

        total = repo.count_actions()
        assert total == 5

        preflop = repo.count_actions(street="PREFLOP")
        assert preflop == 2

        btn_actions = repo.count_actions(position="BTN")
        assert btn_actions == 2

    def test_get_position_action_counts(self, temp_db, sample_hand):
        conn, _ = temp_db
        repo = HandRepository(conn)

        repo.insert_hand(sample_hand)
        conn.commit()

        counts = repo.get_position_action_counts("BTN", "PREFLOP")

        assert "RAISE" in counts
        assert counts["RAISE"] == 1

    def test_get_stack_distribution(self, temp_db, sample_hand):
        conn, _ = temp_db
        repo = HandRepository(conn)

        repo.insert_hand(sample_hand)
        conn.commit()

        dist = repo.get_stack_distribution()

        # Both players have 95-100bb stacks (medium)
        assert "medium" in dist
        assert dist["medium"] == 2

    def test_player_hole_cards_preserved(self, temp_db, sample_hand):
        conn, _ = temp_db
        repo = HandRepository(conn)

        repo.insert_hand(sample_hand)
        conn.commit()

        retrieved = repo.get_hand("TEST123")
        btn_player = retrieved.get_player(Position.BTN)

        assert btn_player.hole_cards == ("Ac", "Kd")

    def test_winners_preserved(self, temp_db, sample_hand):
        conn, _ = temp_db
        repo = HandRepository(conn)

        repo.insert_hand(sample_hand)
        conn.commit()

        retrieved = repo.get_hand("TEST123")
        assert Position.BTN in retrieved.winners
