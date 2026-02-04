"""Tests for hand history parser."""

import pytest
from datetime import datetime
from textwrap import dedent

from moid.parser.models import (
    Position, Street, ActionType, Action, Player, Hand
)
from moid.parser.ignition import IgnitionParser


class TestPosition:
    def test_from_string_basic(self):
        assert Position.from_string("UTG") == Position.UTG
        assert Position.from_string("BTN") == Position.BTN
        assert Position.from_string("BB") == Position.BB

    def test_from_string_with_me_suffix(self):
        assert Position.from_string("BTN [ME]") == Position.BTN
        assert Position.from_string("BB[ME]") == Position.BB

    def test_from_string_aliases(self):
        assert Position.from_string("Dealer") == Position.BTN
        assert Position.from_string("Small Blind") == Position.SB
        assert Position.from_string("Big Blind") == Position.BB
        assert Position.from_string("Cutoff") == Position.CO

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            Position.from_string("Invalid")


class TestStreet:
    def test_from_string(self):
        assert Street.from_string("PREFLOP") == Street.PREFLOP
        assert Street.from_string("flop") == Street.FLOP
        assert Street.from_string("Turn") == Street.TURN
        assert Street.from_string("RIVER") == Street.RIVER

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            Street.from_string("invalid")


class TestActionType:
    def test_from_string_basic(self):
        assert ActionType.from_string("Folds") == ActionType.FOLD
        assert ActionType.from_string("Checks") == ActionType.CHECK
        assert ActionType.from_string("Calls") == ActionType.CALL
        assert ActionType.from_string("Bets") == ActionType.BET
        assert ActionType.from_string("Raises") == ActionType.RAISE

    def test_from_string_all_in(self):
        assert ActionType.from_string("All-in") == ActionType.ALL_IN
        assert ActionType.from_string("all in") == ActionType.ALL_IN

    def test_from_string_blinds(self):
        assert ActionType.from_string("posts small blind") == ActionType.POST_SB
        assert ActionType.from_string("posts big blind") == ActionType.POST_BB


class TestAction:
    def test_repr_without_amount(self):
        action = Action(Position.BTN, Street.PREFLOP, ActionType.FOLD)
        assert "BTN" in repr(action)
        assert "FOLD" in repr(action)

    def test_repr_with_amount(self):
        action = Action(Position.BTN, Street.PREFLOP, ActionType.RAISE, amount=3.0)
        assert "3.00" in repr(action)


class TestPlayer:
    def test_repr_without_cards(self):
        player = Player(Position.BTN, stack=100.0)
        assert "BTN" in repr(player)
        assert "100.0bb" in repr(player)

    def test_repr_with_cards(self):
        player = Player(Position.BTN, stack=100.0, hole_cards=("As", "Kh"))
        assert "[AsKh]" in repr(player)


class TestHand:
    def test_board_properties(self):
        hand = Hand(
            hand_id="123",
            timestamp=datetime.now(),
            stakes=(0.05, 0.10),
            board=["As", "Kh", "Td", "2c", "9s"],
        )

        assert hand.flop == ["As", "Kh", "Td"]
        assert hand.turn == "2c"
        assert hand.river == "9s"

    def test_board_properties_incomplete(self):
        hand = Hand(
            hand_id="123",
            timestamp=datetime.now(),
            stakes=(0.05, 0.10),
            board=["As", "Kh", "Td"],
        )

        assert hand.flop == ["As", "Kh", "Td"]
        assert hand.turn is None
        assert hand.river is None

    def test_stakes_properties(self):
        hand = Hand(
            hand_id="123",
            timestamp=datetime.now(),
            stakes=(0.05, 0.10),
        )

        assert hand.sb == 0.05
        assert hand.bb == 0.10

    def test_get_player(self):
        hand = Hand(
            hand_id="123",
            timestamp=datetime.now(),
            stakes=(0.05, 0.10),
            players=[
                Player(Position.BTN, stack=100),
                Player(Position.BB, stack=95),
            ],
        )

        assert hand.get_player(Position.BTN).stack == 100
        assert hand.get_player(Position.BB).stack == 95
        assert hand.get_player(Position.SB) is None

    def test_get_actions_filtered(self):
        hand = Hand(
            hand_id="123",
            timestamp=datetime.now(),
            stakes=(0.05, 0.10),
            actions=[
                Action(Position.BTN, Street.PREFLOP, ActionType.RAISE, 3.0),
                Action(Position.BB, Street.PREFLOP, ActionType.CALL, 2.0),
                Action(Position.BB, Street.FLOP, ActionType.CHECK),
                Action(Position.BTN, Street.FLOP, ActionType.BET, 4.0),
            ],
        )

        preflop = hand.get_actions(street=Street.PREFLOP)
        assert len(preflop) == 2

        btn_actions = hand.get_actions(position=Position.BTN)
        assert len(btn_actions) == 2

        btn_flop = hand.get_actions(street=Street.FLOP, position=Position.BTN)
        assert len(btn_flop) == 1


class TestIgnitionParser:
    def test_parse_header(self):
        parser = IgnitionParser()
        line = "Ignition Hand #4561234567: HOLDEM No Limit - 0.02/0.05 - 2024-01-15 14:30:00"
        hand = parser._parse_header(line)

        assert hand is not None
        assert hand.hand_id == "4561234567"
        assert hand.stakes == (0.02, 0.05)

    def test_parse_header_with_zone(self):
        parser = IgnitionParser()
        line = "Ignition Hand #1234: Zone Poker HOLDEM No Limit - $0.05/$0.10 - 2024-01-15 10:00:00"
        hand = parser._parse_header(line)

        assert hand is not None
        assert hand.stakes == (0.05, 0.10)

    def test_normalize_card(self):
        parser = IgnitionParser()
        assert parser._normalize_card("As") == "As"
        assert parser._normalize_card("th") == "Th"
        assert parser._normalize_card("KH") == "Kh"

    def test_parse_cards(self):
        parser = IgnitionParser()
        cards = parser._parse_cards("[As Kh 7d]")
        assert cards == ["As", "Kh", "7d"]

    def test_parse_amount(self):
        parser = IgnitionParser()
        assert parser._parse_amount("1.50") == 1.50
        assert parser._parse_amount("$2.00") == 2.00
        assert parser._parse_amount("1,000.50") == 1000.50
        assert parser._parse_amount("") == 0.0


class TestIgnitionParserIntegration:
    """Integration tests with sample hand text."""

    SAMPLE_HAND = dedent("""
        Ignition Hand #4561234567: HOLDEM No Limit - 0.02/0.05 - 2024-01-15 14:30:00
        Table #12345

        Seat 1: UTG ($5.00 in chips)
        Seat 2: UTG+1 ($4.50 in chips)
        Seat 3: Dealer ($5.25 in chips)
        Seat 4: Small Blind ($5.00 in chips)
        Seat 5: Big Blind [ME] ($5.00 in chips)

        Small Blind : Posts small blind $0.02
        Big Blind [ME] : Posts big blind $0.05

        *** HOLE CARDS ***
        Big Blind [ME] : Card dealt to a spot [As Kh]
        UTG : Folds
        UTG+1 : Folds
        Dealer : Raises $0.15 to $0.15
        Small Blind : Folds
        Big Blind [ME] : Calls $0.10

        *** FLOP *** [Ks 7d 2c]
        Big Blind [ME] : Checks
        Dealer : Bets $0.20
        Big Blind [ME] : Calls $0.20

        *** TURN *** [Ks 7d 2c] [9h]
        Big Blind [ME] : Checks
        Dealer : Checks

        *** RIVER *** [Ks 7d 2c 9h] [3s]
        Big Blind [ME] : Bets $0.30
        Dealer : Folds

        Big Blind [ME] : Hand result $0.70
    """).strip()

    def test_parse_sample_hand(self):
        parser = IgnitionParser()
        hand = parser.parse_hand(self.SAMPLE_HAND)

        assert hand is not None
        assert hand.hand_id == "4561234567"
        assert hand.stakes == (0.02, 0.05)
        assert len(hand.players) >= 2
        assert len(hand.board) == 5
        assert hand.board == ["Ks", "7d", "2c", "9h", "3s"]

    def test_parse_actions(self):
        parser = IgnitionParser()
        hand = parser.parse_hand(self.SAMPLE_HAND)

        # Check preflop actions
        preflop = hand.get_actions(street=Street.PREFLOP)
        action_types = [a.action_type for a in preflop]

        assert ActionType.FOLD in action_types
        assert ActionType.RAISE in action_types
        assert ActionType.CALL in action_types

    def test_parse_hole_cards(self):
        parser = IgnitionParser()
        hand = parser.parse_hand(self.SAMPLE_HAND)

        bb_player = hand.get_player(Position.BB)
        assert bb_player is not None
        assert bb_player.hole_cards == ("As", "Kh")
        assert bb_player.is_hero
