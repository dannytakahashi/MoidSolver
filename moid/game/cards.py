"""Card and hand representation utilities."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import random

from treys import Card as TreysCard, Evaluator


class Rank(IntEnum):
    """Card ranks (2-14 where 14 is Ace)."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class Suit(IntEnum):
    """Card suits."""
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


# Mapping for string conversion
RANK_STR = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"
}
STR_RANK = {v: k for k, v in RANK_STR.items()}

SUIT_STR = {0: "c", 1: "d", 2: "h", 3: "s"}
STR_SUIT = {v: k for k, v in SUIT_STR.items()}


@dataclass(frozen=True)
class Card:
    """A playing card."""
    rank: int  # 2-14
    suit: int  # 0-3

    def __str__(self) -> str:
        return f"{RANK_STR[self.rank]}{SUIT_STR[self.suit]}"

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def from_string(cls, s: str) -> "Card":
        """Parse card from string like 'As', 'Th', '2c'."""
        if len(s) != 2:
            raise ValueError(f"Invalid card string: {s}")
        rank_char = s[0].upper()
        suit_char = s[1].lower()

        if rank_char not in STR_RANK:
            raise ValueError(f"Invalid rank: {rank_char}")
        if suit_char not in STR_SUIT:
            raise ValueError(f"Invalid suit: {suit_char}")

        return cls(rank=STR_RANK[rank_char], suit=STR_SUIT[suit_char])

    def to_treys(self) -> int:
        """Convert to treys library card format."""
        return TreysCard.new(str(self))


@dataclass
class Hand:
    """A two-card starting hand."""
    card1: Card
    card2: Card

    def __post_init__(self):
        # Ensure card1 has higher or equal rank
        if self.card1.rank < self.card2.rank:
            self.card1, self.card2 = self.card2, self.card1

    @property
    def is_pair(self) -> bool:
        """Check if hand is a pocket pair."""
        return self.card1.rank == self.card2.rank

    @property
    def is_suited(self) -> bool:
        """Check if hand is suited."""
        return self.card1.suit == self.card2.suit

    @property
    def canonical(self) -> str:
        """
        Get canonical hand notation (e.g., 'AKs', 'QQ', '72o').

        This groups equivalent hands regardless of specific suits.
        """
        r1 = RANK_STR[self.card1.rank]
        r2 = RANK_STR[self.card2.rank]

        if self.is_pair:
            return f"{r1}{r2}"
        elif self.is_suited:
            return f"{r1}{r2}s"
        else:
            return f"{r1}{r2}o"

    def __str__(self) -> str:
        return f"{self.card1}{self.card2}"

    def __repr__(self) -> str:
        return f"Hand({self.card1}, {self.card2})"

    @classmethod
    def from_string(cls, s: str) -> "Hand":
        """Parse hand from string like 'AsKh' or 'AKs'."""
        if len(s) == 4:
            # Specific cards: 'AsKh'
            card1 = Card.from_string(s[:2])
            card2 = Card.from_string(s[2:])
            return cls(card1, card2)
        elif len(s) == 2:
            # Pair: 'AA'
            rank = STR_RANK[s[0].upper()]
            return cls(
                Card(rank, Suit.SPADES),
                Card(rank, Suit.HEARTS)
            )
        elif len(s) == 3:
            # Suited or offsuit: 'AKs' or 'AKo'
            r1 = STR_RANK[s[0].upper()]
            r2 = STR_RANK[s[1].upper()]
            suited = s[2].lower() == 's'

            if suited:
                return cls(Card(r1, Suit.SPADES), Card(r2, Suit.SPADES))
            else:
                return cls(Card(r1, Suit.SPADES), Card(r2, Suit.HEARTS))
        else:
            raise ValueError(f"Invalid hand string: {s}")

    def to_treys(self) -> list[int]:
        """Convert to treys library format."""
        return [self.card1.to_treys(), self.card2.to_treys()]


class Deck:
    """A standard 52-card deck."""

    def __init__(self):
        self.cards: list[Card] = []
        self.reset()

    def reset(self) -> None:
        """Reset deck to full 52 cards."""
        self.cards = [
            Card(rank, suit)
            for rank in range(2, 15)
            for suit in range(4)
        ]

    def shuffle(self) -> None:
        """Shuffle the deck."""
        random.shuffle(self.cards)

    def deal(self, n: int = 1) -> list[Card]:
        """Deal n cards from the deck."""
        if n > len(self.cards):
            raise ValueError(f"Cannot deal {n} cards, only {len(self.cards)} remaining")
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def remove(self, cards: list[Card]) -> None:
        """Remove specific cards from the deck."""
        for card in cards:
            if card in self.cards:
                self.cards.remove(card)

    def __len__(self) -> int:
        return len(self.cards)


def hand_to_treys(hand: Hand, board: list[Card]) -> tuple[list[int], list[int]]:
    """Convert hand and board to treys format."""
    hand_treys = hand.to_treys()
    board_treys = [c.to_treys() for c in board]
    return hand_treys, board_treys


# Hand ranking categories for abstraction
HAND_RANKINGS = {
    "premium_pairs": ["AA", "KK", "QQ", "JJ"],
    "medium_pairs": ["TT", "99", "88", "77"],
    "small_pairs": ["66", "55", "44", "33", "22"],
    "premium_broadway": ["AKs", "AKo", "AQs", "AQo", "AJs"],
    "suited_connectors": ["JTs", "T9s", "98s", "87s", "76s", "65s", "54s"],
    "suited_aces": ["A5s", "A4s", "A3s", "A2s"],
    "suited_kings": ["KQs", "KJs", "KTs"],
}


def get_all_hands() -> list[str]:
    """Generate all 169 unique starting hands in canonical form."""
    hands = []
    ranks = "AKQJT98765432"

    # Pairs
    for r in ranks:
        hands.append(f"{r}{r}")

    # Non-pairs
    for i, r1 in enumerate(ranks):
        for r2 in ranks[i+1:]:
            hands.append(f"{r1}{r2}s")  # Suited
            hands.append(f"{r1}{r2}o")  # Offsuit

    return hands


def parse_range(range_str: str) -> list[str]:
    """
    Parse a hand range string into list of hands.

    Examples:
        "AA" -> ["AA"]
        "AKs" -> ["AKs"]
        "TT+" -> ["TT", "JJ", "QQ", "KK", "AA"]
        "ATs+" -> ["ATs", "AJs", "AQs", "AKs"]
        "22-55" -> ["22", "33", "44", "55"]
    """
    hands = []
    range_str = range_str.strip()

    # Pair plus: "TT+"
    if len(range_str) == 3 and range_str[2] == "+" and range_str[0] == range_str[1]:
        start_rank = STR_RANK[range_str[0]]
        for rank in range(start_rank, 15):
            hands.append(f"{RANK_STR[rank]}{RANK_STR[rank]}")
        return hands

    # Pair range: "22-55"
    if "-" in range_str and len(range_str) == 5:
        low = STR_RANK[range_str[0]]
        high = STR_RANK[range_str[3]]
        for rank in range(low, high + 1):
            hands.append(f"{RANK_STR[rank]}{RANK_STR[rank]}")
        return hands

    # Suited/offsuit plus: "ATs+"
    if len(range_str) == 4 and range_str[3] == "+":
        high_rank = STR_RANK[range_str[0]]
        low_rank = STR_RANK[range_str[1]]
        suited = range_str[2] == "s"

        suffix = "s" if suited else "o"
        for rank in range(low_rank, high_rank):
            hands.append(f"{RANK_STR[high_rank]}{RANK_STR[rank]}{suffix}")
        return hands

    # Single hand
    return [range_str]
