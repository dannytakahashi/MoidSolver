"""Tests for analysis module."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from moid.db.schema import create_database
from moid.db.repository import HandRepository
from moid.parser.models import Action, ActionType, Hand, Player, Position, Street
from moid.analysis.stats import PlayerStats, compute_stats
from moid.analysis.population import PopulationAnalyzer, PopulationStats
from moid.classifier.archetypes import (
    PlayerArchetype, ArchetypeClassifier, BayesianClassifier
)


@pytest.fixture
def populated_db():
    """Create a database with sample hands."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = create_database(db_path)
    repo = HandRepository(conn)

    # Insert sample hands with various actions
    hands = []

    for i in range(100):
        # Vary VPIP by position
        btn_vpip = i % 3 != 0  # 66% VPIP
        bb_vpip = True  # Always in

        actions = []

        # BTN action
        if btn_vpip:
            if i % 5 == 0:  # 20% raise when in pot
                actions.append(Action(Position.BTN, Street.PREFLOP, ActionType.RAISE, 3.0))
            else:
                actions.append(Action(Position.BTN, Street.PREFLOP, ActionType.CALL, 1.0))
        else:
            actions.append(Action(Position.BTN, Street.PREFLOP, ActionType.FOLD))

        # BB action (already in)
        if btn_vpip and i % 5 == 0:  # Facing raise
            if i % 10 == 0:  # 10% 3-bet
                actions.append(Action(Position.BB, Street.PREFLOP, ActionType.RAISE, 9.0))
            else:
                actions.append(Action(Position.BB, Street.PREFLOP, ActionType.CALL, 2.0))

        # Flop actions (if not folded)
        if btn_vpip:
            board = ["As", "Kh", "7d"]

            # BB check
            actions.append(Action(Position.BB, Street.FLOP, ActionType.CHECK))

            # BTN cbet sometimes
            if i % 4 == 0:  # 25% cbet
                actions.append(Action(Position.BTN, Street.FLOP, ActionType.BET, 4.0))
                # BB folds to cbet sometimes
                if i % 8 == 0:  # 50% fold to cbet
                    actions.append(Action(Position.BB, Street.FLOP, ActionType.FOLD))
                else:
                    actions.append(Action(Position.BB, Street.FLOP, ActionType.CALL, 4.0))
            else:
                actions.append(Action(Position.BTN, Street.FLOP, ActionType.CHECK))
        else:
            board = []

        hand = Hand(
            hand_id=f"TEST{i:04d}",
            timestamp=datetime.now(),
            stakes=(0.02, 0.05),
            board=board,
            players=[
                Player(Position.BTN, stack=100.0),
                Player(Position.BB, stack=100.0, is_hero=True),
            ],
            actions=actions,
        )
        hands.append(hand)

    repo.insert_hands(iter(hands))
    conn.commit()

    yield conn, db_path

    conn.close()
    db_path.unlink()


class TestPlayerStats:
    def test_default_values(self):
        stats = PlayerStats()
        assert stats.hands == 0
        assert stats.vpip == 0.0
        assert stats.pfr == 0.0

    def test_repr(self):
        stats = PlayerStats(
            hands=100,
            vpip=35.0,
            pfr=12.0,
            three_bet=5.0,
            af=1.5,
        )
        repr_str = repr(stats)

        assert "100" in repr_str
        assert "35.0" in repr_str
        assert "12.0" in repr_str


class TestComputeStats:
    def test_basic_stats(self, populated_db):
        conn, _ = populated_db
        stats = compute_stats(conn)

        assert stats.hands > 0
        assert 0 <= stats.vpip <= 100
        assert 0 <= stats.pfr <= 100

    def test_position_filter(self, populated_db):
        conn, _ = populated_db

        btn_stats = compute_stats(conn, position="BTN")
        bb_stats = compute_stats(conn, position="BB")

        # Both should have hands
        assert btn_stats.hands > 0
        assert bb_stats.hands > 0

    def test_stack_filter(self, populated_db):
        conn, _ = populated_db

        # All our sample hands have 100bb stacks
        deep_stats = compute_stats(conn, min_stack=50)
        short_stats = compute_stats(conn, max_stack=50)

        assert deep_stats.hands > 0
        assert short_stats.hands == 0


class TestPopulationAnalyzer:
    def test_analyze(self, populated_db):
        conn, _ = populated_db
        analyzer = PopulationAnalyzer(conn)

        pop_stats = analyzer.analyze()

        assert isinstance(pop_stats, PopulationStats)
        assert pop_stats.overall.hands > 0

    def test_by_position(self, populated_db):
        conn, _ = populated_db
        analyzer = PopulationAnalyzer(conn)

        pop_stats = analyzer.analyze()

        assert "BTN" in pop_stats.by_position
        assert "BB" in pop_stats.by_position

    def test_get_exploits(self, populated_db):
        conn, _ = populated_db
        analyzer = PopulationAnalyzer(conn)

        exploits = analyzer.get_exploits()

        # Should return list of strings
        assert isinstance(exploits, list)


class TestPopulationStats:
    def test_get_tendency(self):
        stats = PopulationStats()
        stats.overall = PlayerStats(vpip=45.0)

        tendency = stats.get_tendency("vpip")
        assert "high" in tendency.lower()

    def test_get_tendency_low(self):
        stats = PopulationStats()
        stats.overall = PlayerStats(vpip=15.0)

        tendency = stats.get_tendency("vpip")
        assert "low" in tendency.lower()


class TestArchetypeClassifier:
    def test_classify_fish(self):
        classifier = ArchetypeClassifier(min_hands=0)

        stats = PlayerStats(
            hands=100,
            vpip=45.0,
            pfr=8.0,
            af=0.8,
        )

        archetype = classifier.classify(stats)
        assert archetype == PlayerArchetype.FISH

    def test_classify_tag(self):
        classifier = ArchetypeClassifier(min_hands=0)

        stats = PlayerStats(
            hands=100,
            vpip=22.0,
            pfr=18.0,
            af=2.5,
        )

        archetype = classifier.classify(stats)
        assert archetype == PlayerArchetype.TAG

    def test_classify_nit(self):
        classifier = ArchetypeClassifier(min_hands=0)

        stats = PlayerStats(
            hands=100,
            vpip=12.0,
            pfr=10.0,
            af=1.5,
        )

        archetype = classifier.classify(stats)
        assert archetype == PlayerArchetype.NIT

    def test_insufficient_hands(self):
        classifier = ArchetypeClassifier(min_hands=50)

        stats = PlayerStats(
            hands=30,  # Below threshold
            vpip=45.0,
            pfr=8.0,
            af=0.8,
        )

        archetype = classifier.classify(stats)
        assert archetype == PlayerArchetype.UNKNOWN

    def test_get_exploits(self):
        classifier = ArchetypeClassifier()

        exploits = classifier.get_exploits(PlayerArchetype.FISH)

        assert isinstance(exploits, list)
        assert len(exploits) > 0


class TestBayesianClassifier:
    def test_initial_priors(self):
        classifier = BayesianClassifier()

        # Fish should be most common prior
        assert classifier.priors[PlayerArchetype.FISH] > classifier.priors[PlayerArchetype.MANIAC]

    def test_observe_preflop(self):
        classifier = BayesianClassifier()

        # Observe loose passive play
        for _ in range(20):
            classifier.observe_preflop(vpip=True, pfr=False)

        classification = classifier.get_classification()

        # Should lean towards fish or calling station
        assert classification in (
            PlayerArchetype.FISH,
            PlayerArchetype.CALLING_STATION,
            PlayerArchetype.UNKNOWN,
        )

    def test_observe_updates_posteriors(self):
        classifier = BayesianClassifier()

        initial = classifier.posteriors.copy()

        classifier.observe_preflop(vpip=True, pfr=True)

        # Posteriors should change
        changed = any(
            classifier.posteriors[a] != initial[a]
            for a in classifier.posteriors
        )
        assert changed

    def test_reset(self):
        classifier = BayesianClassifier()

        classifier.observe_preflop(vpip=True, pfr=True)
        classifier.reset()

        assert classifier.hands_count == 0
        assert classifier.vpip_count == 0

    def test_probabilities_sum_to_one(self):
        classifier = BayesianClassifier()

        for _ in range(10):
            classifier.observe_preflop(vpip=True, pfr=False)

        probs = classifier.get_probabilities()
        total = sum(probs.values())

        assert total == pytest.approx(1.0, rel=0.01)
