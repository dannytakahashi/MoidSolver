"""Tests for equity calculations."""

import pytest

from moid.game.cards import Card, Hand
from moid.game.equity import (
    EquityCalculator, calculate_equity,
    calculate_hand_strength, get_hand_class
)


@pytest.fixture
def calculator():
    return EquityCalculator()


@pytest.fixture
def board_flop():
    return [
        Card.from_string("Ks"),
        Card.from_string("7d"),
        Card.from_string("2c"),
    ]


@pytest.fixture
def board_river():
    return [
        Card.from_string("Ks"),
        Card.from_string("7d"),
        Card.from_string("2c"),
        Card.from_string("9h"),
        Card.from_string("3s"),
    ]


class TestEquityCalculator:
    def test_hand_vs_hand_river(self, calculator, board_river):
        # AA vs KK on K-high board - KK wins
        aa = Hand.from_string("AsAh")
        kk = Hand.from_string("KhKc")

        eq_aa, eq_kk, _ = calculator.hand_vs_hand(aa, kk, board_river)

        # KK has trips, AA has one pair
        assert eq_kk > eq_aa
        assert eq_kk == 1.0
        assert eq_aa == 0.0

    def test_hand_vs_hand_flop(self, calculator, board_flop):
        # Overpair vs underpair on flop
        aa = Hand.from_string("AsAh")
        jj = Hand.from_string("JsJh")

        eq_aa, eq_jj, _ = calculator.hand_vs_hand(
            aa, jj, board_flop, num_simulations=1000
        )

        # AA should be heavily favored
        assert eq_aa > 0.8
        assert eq_jj < 0.2

    def test_hand_vs_hand_tie(self, calculator, board_river):
        # Same hand should tie
        ak1 = Hand.from_string("AcKh")
        ak2 = Hand.from_string("AdKc")

        eq1, eq2, ties = calculator.hand_vs_hand(ak1, ak2, board_river)

        # Both should have equal equity (split pot)
        assert eq1 == eq2 == 0.5

    def test_duplicate_cards_raises(self, calculator, board_river):
        # Hand overlaps with board
        ks_hand = Hand.from_string("KsQh")

        with pytest.raises(ValueError, match="Duplicate"):
            calculator.hand_vs_hand(
                ks_hand,
                Hand.from_string("JsTh"),
                board_river
            )


class TestCalculateEquity:
    def test_equity_vs_random(self, board_flop):
        # Top pair should have good equity vs random
        ak = Hand.from_string("AsKh")  # Top pair
        equity = calculate_equity(ak, board_flop, num_simulations=500)

        # Should be favored against random hand
        assert equity > 0.5

    def test_equity_multiway(self, board_flop):
        # Equity decreases with more opponents
        ak = Hand.from_string("AsKh")

        eq_heads_up = calculate_equity(
            ak, board_flop, num_opponents=1, num_simulations=500
        )
        eq_three_way = calculate_equity(
            ak, board_flop, num_opponents=2, num_simulations=500
        )

        assert eq_heads_up > eq_three_way


class TestHandStrength:
    def test_calculate_hand_strength(self, board_river):
        # Trips should rank better than one pair
        kk = Hand.from_string("KhKc")  # Trips
        aa = Hand.from_string("AsAh")  # One pair

        kk_rank = calculate_hand_strength(kk, board_river)
        aa_rank = calculate_hand_strength(aa, board_river)

        # Lower rank is better in treys
        assert kk_rank < aa_rank

    def test_get_hand_class(self, board_river):
        kk = Hand.from_string("KhKc")
        hand_class = get_hand_class(kk, board_river)

        assert "Three of a Kind" in hand_class

    def test_hand_strength_requires_5_cards(self):
        board = [Card.from_string("As"), Card.from_string("Kh")]

        with pytest.raises(ValueError, match="5 cards"):
            calculate_hand_strength(Hand.from_string("AhAd"), board)


class TestEquityEdgeCases:
    def test_flush_draw(self, calculator):
        # Flush draw on flop
        flush_draw = Hand.from_string("AsKs")
        board = [
            Card.from_string("Qs"),
            Card.from_string("7s"),
            Card.from_string("2h"),
        ]
        made_hand = Hand.from_string("QhQd")

        eq_draw, eq_made, _ = calculator.hand_vs_hand(
            flush_draw, made_hand, board, num_simulations=1000
        )

        # Flush draw should have reasonable equity (around 35%)
        assert 0.25 < eq_draw < 0.50

    def test_straight_draw(self, calculator):
        # Open-ended straight draw
        oesd = Hand.from_string("JhTh")
        board = [
            Card.from_string("9s"),
            Card.from_string("8d"),
            Card.from_string("2c"),
        ]
        overpair = Hand.from_string("KsKd")

        eq_oesd, eq_kk, _ = calculator.hand_vs_hand(
            oesd, overpair, board, num_simulations=1000
        )

        # OESD should have about 30% equity
        assert 0.20 < eq_oesd < 0.45
