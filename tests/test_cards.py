"""Tests for card and hand representation."""

import pytest

from moid.game.cards import (
    Card, Hand, Deck, Rank, Suit,
    RANK_STR, STR_RANK, get_all_hands, parse_range
)


class TestCard:
    def test_from_string(self):
        card = Card.from_string("As")
        assert card.rank == Rank.ACE
        assert card.suit == Suit.SPADES

    def test_from_string_ten(self):
        card = Card.from_string("Th")
        assert card.rank == Rank.TEN
        assert card.suit == Suit.HEARTS

    def test_from_string_lowercase(self):
        card = Card.from_string("kd")
        assert card.rank == Rank.KING
        assert card.suit == Suit.DIAMONDS

    def test_str(self):
        card = Card(Rank.ACE, Suit.SPADES)
        assert str(card) == "As"

    def test_from_string_invalid_rank(self):
        with pytest.raises(ValueError):
            Card.from_string("Xs")

    def test_from_string_invalid_suit(self):
        with pytest.raises(ValueError):
            Card.from_string("Ax")

    def test_equality(self):
        card1 = Card.from_string("As")
        card2 = Card.from_string("As")
        assert card1 == card2

    def test_to_treys(self):
        card = Card.from_string("As")
        treys_card = card.to_treys()
        assert isinstance(treys_card, int)


class TestHand:
    def test_from_string_specific(self):
        hand = Hand.from_string("AsKh")
        assert hand.card1.rank == Rank.ACE
        assert hand.card2.rank == Rank.KING

    def test_from_string_pair(self):
        hand = Hand.from_string("AA")
        assert hand.card1.rank == Rank.ACE
        assert hand.card2.rank == Rank.ACE
        assert hand.is_pair

    def test_from_string_suited(self):
        hand = Hand.from_string("AKs")
        assert hand.is_suited
        assert not hand.is_pair

    def test_from_string_offsuit(self):
        hand = Hand.from_string("AKo")
        assert not hand.is_suited
        assert not hand.is_pair

    def test_canonical_pair(self):
        hand = Hand.from_string("AsAh")
        assert hand.canonical == "AA"

    def test_canonical_suited(self):
        hand = Hand.from_string("AsKs")
        assert hand.canonical == "AKs"

    def test_canonical_offsuit(self):
        hand = Hand.from_string("AsKh")
        assert hand.canonical == "AKo"

    def test_card_ordering(self):
        # Lower card first in string should still have higher rank first
        hand = Hand.from_string("KsAs")
        assert hand.card1.rank == Rank.ACE
        assert hand.card2.rank == Rank.KING

    def test_str(self):
        hand = Hand.from_string("AsKh")
        assert str(hand) == "AsKh"


class TestDeck:
    def test_full_deck(self):
        deck = Deck()
        assert len(deck) == 52

    def test_deal(self):
        deck = Deck()
        cards = deck.deal(5)
        assert len(cards) == 5
        assert len(deck) == 47

    def test_deal_too_many(self):
        deck = Deck()
        with pytest.raises(ValueError):
            deck.deal(53)

    def test_remove(self):
        deck = Deck()
        card = Card.from_string("As")
        deck.remove([card])
        assert len(deck) == 51
        assert card not in deck.cards

    def test_shuffle(self):
        deck1 = Deck()
        deck2 = Deck()
        deck2.shuffle()

        # Cards should be in different order after shuffle (very likely)
        # This could theoretically fail but probability is astronomically low
        same_order = all(
            c1 == c2 for c1, c2 in zip(deck1.cards[:10], deck2.cards[:10])
        )
        assert not same_order

    def test_reset(self):
        deck = Deck()
        deck.deal(20)
        assert len(deck) == 32

        deck.reset()
        assert len(deck) == 52


class TestHandHelpers:
    def test_get_all_hands(self):
        hands = get_all_hands()
        # 13 pairs + 78 suited + 78 offsuit = 169
        assert len(hands) == 169

        # Check some specific hands exist
        assert "AA" in hands
        assert "AKs" in hands
        assert "AKo" in hands
        assert "72o" in hands

    def test_parse_range_single(self):
        hands = parse_range("AA")
        assert hands == ["AA"]

    def test_parse_range_pair_plus(self):
        hands = parse_range("TT+")
        assert "TT" in hands
        assert "JJ" in hands
        assert "QQ" in hands
        assert "KK" in hands
        assert "AA" in hands
        assert len(hands) == 5

    def test_parse_range_pair_range(self):
        hands = parse_range("22-55")
        assert hands == ["22", "33", "44", "55"]

    def test_parse_range_suited_plus(self):
        hands = parse_range("ATs+")
        assert "ATs" in hands
        assert "AJs" in hands
        assert "AQs" in hands
        assert "AKs" in hands
        # AAs is not a thing, so should not include ace
        assert "AAs" not in hands
