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


# Tests for new analysis modules


class TestGTOBenchmarks:
    def test_default_benchmarks(self):
        from moid.analysis.benchmarks import GTOBenchmarks, GTO_BENCHMARKS

        # Should have all positions
        assert "UTG" in GTO_BENCHMARKS.by_position
        assert "BTN" in GTO_BENCHMARKS.by_position
        assert "BB" in GTO_BENCHMARKS.by_position

    def test_position_benchmarks(self):
        from moid.analysis.benchmarks import GTO_BENCHMARKS

        btn_bench = GTO_BENCHMARKS.by_position["BTN"]
        utg_bench = GTO_BENCHMARKS.by_position["UTG"]

        # BTN should have higher RFI than UTG
        assert btn_bench.rfi > utg_bench.rfi

    def test_get_benchmark(self):
        from moid.analysis.benchmarks import get_benchmark

        rfi = get_benchmark("BTN", "rfi")
        assert rfi is not None
        assert 40 <= rfi <= 50

    def test_get_overall_range(self):
        from moid.analysis.benchmarks import GTO_BENCHMARKS

        vpip_range = GTO_BENCHMARKS.get_overall_range("vpip")
        assert vpip_range is not None
        assert len(vpip_range) == 2
        assert vpip_range[0] < vpip_range[1]


class TestMicrostakesAdjustments:
    def test_adjusted_rfi(self):
        from moid.analysis.benchmarks import MICROSTAKES_ADJUSTMENTS

        # BTN should have positive adjustment (open wider)
        btn_adj = MICROSTAKES_ADJUSTMENTS.get_adjusted_rfi("BTN", 45.0)
        assert btn_adj >= 45.0

        # EP should have negative adjustment (open tighter)
        utg_adj = MICROSTAKES_ADJUSTMENTS.get_adjusted_rfi("UTG", 15.0)
        assert utg_adj <= 15.0

    def test_exploits_list(self):
        from moid.analysis.benchmarks import MICROSTAKES_ADJUSTMENTS

        assert len(MICROSTAKES_ADJUSTMENTS.exploits) > 0


class TestClassifyBoardTexture:
    def test_dry_board(self):
        from moid.analysis.benchmarks import classify_board_texture

        texture = classify_board_texture(["Ks", "7h", "2d"])
        assert texture == "dry"

    def test_paired_board(self):
        from moid.analysis.benchmarks import classify_board_texture

        texture = classify_board_texture(["8s", "8h", "3d"])
        assert texture == "paired"

    def test_broadway_board(self):
        from moid.analysis.benchmarks import classify_board_texture

        texture = classify_board_texture(["As", "Kh", "Qd"])
        assert texture == "broadway"

    def test_wet_board(self):
        from moid.analysis.benchmarks import classify_board_texture

        texture = classify_board_texture(["Jh", "Th", "9c"])
        assert texture in ("wet", "semi_wet")


class TestHeroAnalyzer:
    def test_analyze(self, populated_db):
        from moid.analysis.hero import HeroAnalyzer

        conn, _ = populated_db
        analyzer = HeroAnalyzer(conn)

        hero_stats = analyzer.analyze()

        assert hero_stats.overall.hands > 0

    def test_leaks_identified(self, populated_db):
        from moid.analysis.hero import HeroAnalyzer

        conn, _ = populated_db
        analyzer = HeroAnalyzer(conn)

        hero_stats = analyzer.analyze()

        # Leaks should be a list
        assert isinstance(hero_stats.leaks, list)

    def test_position_stats(self, populated_db):
        from moid.analysis.hero import HeroAnalyzer

        conn, _ = populated_db
        analyzer = HeroAnalyzer(conn)

        hero_stats = analyzer.analyze()

        # Should have at least BB (hero is BB in our fixture)
        assert "BB" in hero_stats.by_position


class TestLeak:
    def test_leak_creation(self):
        from moid.analysis.hero import Leak

        leak = Leak(
            category="preflop",
            position="BTN",
            stat="vpip",
            hero_value=45.0,
            optimal_value=28.0,
            deviation=17.0,
            severity="major",
            description="VPIP too high",
            suggestion="Tighten up",
        )

        assert leak.severity == "major"
        assert leak.deviation == 17.0


class TestSpotAnalyzer:
    def test_analyze_spot(self, populated_db):
        from moid.analysis.spots import SpotAnalyzer, SpotType

        conn, _ = populated_db
        analyzer = SpotAnalyzer(conn)

        stats = analyzer.analyze_spot(SpotType.RFI)

        assert stats.spot_type == SpotType.RFI

    def test_get_spot_summary(self, populated_db):
        from moid.analysis.spots import SpotAnalyzer

        conn, _ = populated_db
        analyzer = SpotAnalyzer(conn)

        summary = analyzer.get_spot_summary()

        assert isinstance(summary, dict)
        assert len(summary) > 0


class TestSpotStats:
    def test_deviation_summary(self):
        from moid.analysis.spots import SpotStats, SpotType

        stats = SpotStats(
            spot_type=SpotType.RFI,
            opportunities=100,
            fold_pct=70.0,
            bet_pct=20.0,
            optimal_fold=50.0,
            optimal_bet=45.0,
        )

        summary = stats.deviation_summary
        assert "too much" in summary.lower() or "not enough" in summary.lower()

    def test_is_exploitable(self):
        from moid.analysis.spots import SpotStats, SpotType

        # Way off from optimal
        stats = SpotStats(
            spot_type=SpotType.FACING_CBET,
            opportunities=100,
            fold_pct=80.0,
            optimal_fold=45.0,
        )

        assert stats.is_exploitable


class TestHandFlagger:
    def test_flag_hands(self, populated_db):
        from moid.analysis.flagger import HandFlagger

        conn, _ = populated_db
        flagger = HandFlagger(conn)

        flagged = flagger.flag_hands(limit=10)

        assert isinstance(flagged, list)

    def test_get_stats(self, populated_db):
        from moid.analysis.flagger import HandFlagger

        conn, _ = populated_db
        flagger = HandFlagger(conn)

        stats = flagger.get_stats()

        assert "total_hero_hands" in stats
        assert stats["total_hero_hands"] >= 0


class TestFlaggedHand:
    def test_flagged_hand_creation(self):
        from moid.analysis.flagger import FlaggedHand, FlagReason
        from datetime import datetime

        hand = FlaggedHand(
            hand_id="TEST001",
            db_id=1,
            timestamp=datetime.now(),
            stakes=(0.02, 0.05),
            board=["As", "Kh", "7d"],
            hero_position="BTN",
            hero_cards=("Ac", "Kc"),
            pot_size=45.0,
            hero_result=-25.0,
            flags=[FlagReason.LARGE_POT_LOSS],
            priority=1,
            summary="Large pot loss with top two pair",
        )

        assert hand.priority == 1
        assert FlagReason.LARGE_POT_LOSS in hand.flags


class TestFlagCriteria:
    def test_default_criteria(self):
        from moid.analysis.flagger import FlagCriteria

        criteria = FlagCriteria()

        assert criteria.large_pot_threshold > 0
        assert criteria.big_loss_threshold < 0
