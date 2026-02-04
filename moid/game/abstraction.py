"""Bet sizing and hand abstraction for tractable solving."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from treys import Evaluator

from .cards import Card, Hand, Deck


@dataclass
class BetSizeAbstraction:
    """
    Defines available bet sizes for game tree building.

    Abstracting bet sizes reduces the game tree complexity
    while maintaining strategic accuracy.
    """
    # Preflop bet sizes (in BBs)
    preflop_open: list[float] = field(default_factory=lambda: [2.5, 3.0])
    preflop_3bet: list[float] = field(default_factory=lambda: [9.0, 10.0])
    preflop_4bet: list[float] = field(default_factory=lambda: [22.0, 25.0])

    # Postflop bet sizes (as fraction of pot)
    flop_bet: list[float] = field(default_factory=lambda: [0.33, 0.5, 0.75, 1.0])
    turn_bet: list[float] = field(default_factory=lambda: [0.5, 0.75, 1.0])
    river_bet: list[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5])

    # Raise sizes (as multiple of previous bet)
    raise_sizes: list[float] = field(default_factory=lambda: [2.5, 3.0])

    def get_bet_sizes(self, street: int, pot: float, is_raise: bool = False) -> list[float]:
        """
        Get available bet sizes for a street.

        Args:
            street: 0=preflop, 1=flop, 2=turn, 3=river
            pot: Current pot size in BBs
            is_raise: Whether this is a raise (vs initial bet)

        Returns:
            List of bet sizes in BBs
        """
        if street == 0:
            return self.preflop_open
        elif street == 1:
            fractions = self.flop_bet
        elif street == 2:
            fractions = self.turn_bet
        else:
            fractions = self.river_bet

        # Convert fractions to actual amounts
        return [f * pot for f in fractions]

    def simplify(self, num_sizes: int = 2) -> "BetSizeAbstraction":
        """Return simplified abstraction with fewer bet sizes."""
        return BetSizeAbstraction(
            preflop_open=[2.5],
            preflop_3bet=[9.0],
            preflop_4bet=[22.0],
            flop_bet=self.flop_bet[:num_sizes],
            turn_bet=self.turn_bet[:num_sizes],
            river_bet=self.river_bet[:num_sizes],
            raise_sizes=self.raise_sizes[:1],
        )


@dataclass
class EquityBucket:
    """A bucket of hands grouped by equity."""
    min_equity: float
    max_equity: float
    hands: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.min_equity:.0%}-{self.max_equity:.0%}"


class HandAbstraction:
    """
    Groups hands by equity for tractable solving.

    Instead of solving for all 1326 hand combinations,
    we group hands into equity buckets on each board.
    """

    def __init__(self, num_buckets: int = 8):
        """
        Initialize hand abstraction.

        Args:
            num_buckets: Number of equity buckets (default 8)
        """
        self.num_buckets = num_buckets
        self.evaluator = Evaluator()
        self._cache: dict[str, list[EquityBucket]] = {}

    def compute_buckets(
        self,
        board: list[Card],
        num_simulations: int = 1000,
    ) -> list[EquityBucket]:
        """
        Compute equity buckets for all hands on a board.

        Args:
            board: Board cards
            num_simulations: Monte Carlo simulations per hand

        Returns:
            List of EquityBucket objects
        """
        board_key = "".join(str(c) for c in board)
        if board_key in self._cache:
            return self._cache[board_key]

        # Generate all possible hands (excluding board cards)
        deck = Deck()
        deck.remove(board)
        remaining_cards = deck.cards

        # Compute equity for each hand
        hand_equities: list[tuple[str, float]] = []

        for i, c1 in enumerate(remaining_cards):
            for c2 in remaining_cards[i + 1:]:
                hand = Hand(c1, c2)
                equity = self._compute_equity(hand, board, num_simulations)
                hand_equities.append((hand.canonical, equity))

        # Sort by equity
        hand_equities.sort(key=lambda x: x[1])

        # Create buckets
        bucket_size = len(hand_equities) / self.num_buckets
        buckets = []

        for i in range(self.num_buckets):
            start_idx = int(i * bucket_size)
            end_idx = int((i + 1) * bucket_size)
            if i == self.num_buckets - 1:
                end_idx = len(hand_equities)

            bucket_hands = hand_equities[start_idx:end_idx]
            if bucket_hands:
                min_eq = bucket_hands[0][1]
                max_eq = bucket_hands[-1][1]
                hands = [h[0] for h in bucket_hands]
                buckets.append(EquityBucket(min_eq, max_eq, hands))

        self._cache[board_key] = buckets
        return buckets

    def _compute_equity(
        self,
        hand: Hand,
        board: list[Card],
        num_simulations: int,
    ) -> float:
        """Compute hand equity via Monte Carlo."""
        # Convert to treys format
        hand_treys = hand.to_treys()
        board_treys = [c.to_treys() for c in board]

        # Remove used cards from deck
        used_cards = set(hand_treys + board_treys)
        deck = Deck()
        available = [c for c in deck.cards
                     if c.to_treys() not in used_cards]

        wins = 0
        ties = 0

        for _ in range(num_simulations):
            # Sample opponent hand and remaining board
            np.random.shuffle(available)

            # Sample opponent hand (2 cards)
            opp_hand = [available[0].to_treys(), available[1].to_treys()]

            # Complete board if needed
            remaining_board = 5 - len(board_treys)
            full_board = board_treys + [
                available[i + 2].to_treys()
                for i in range(remaining_board)
            ]

            # Evaluate hands
            hero_rank = self.evaluator.evaluate(hand_treys, full_board)
            opp_rank = self.evaluator.evaluate(opp_hand, full_board)

            if hero_rank < opp_rank:  # Lower is better in treys
                wins += 1
            elif hero_rank == opp_rank:
                ties += 0.5

        return (wins + ties) / num_simulations

    def get_bucket(self, hand: Hand, board: list[Card]) -> int:
        """
        Get the bucket index for a hand on a board.

        Args:
            hand: Player's hand
            board: Board cards

        Returns:
            Bucket index (0 to num_buckets-1)
        """
        buckets = self.compute_buckets(board)

        for i, bucket in enumerate(buckets):
            if hand.canonical in bucket.hands:
                return i

        # If not found, compute equity and find closest bucket
        equity = self._compute_equity(hand, board, 500)
        for i, bucket in enumerate(buckets):
            if bucket.min_equity <= equity <= bucket.max_equity:
                return i

        return self.num_buckets // 2  # Default to middle

    def get_bucket_hands(self, bucket_idx: int, board: list[Card]) -> list[str]:
        """Get all hands in a bucket."""
        buckets = self.compute_buckets(board)
        if 0 <= bucket_idx < len(buckets):
            return buckets[bucket_idx].hands
        return []


class SimpleAbstraction:
    """
    Simple hand strength abstraction for fast solving.

    Categorizes hands into strength tiers without Monte Carlo:
    - Monsters: Sets, two pair+
    - Strong: Top pair good kicker, overpairs
    - Medium: Top pair weak kicker, middle pair
    - Weak: Bottom pair, draws
    - Air: No pair, no draw
    """

    def __init__(self):
        self.evaluator = Evaluator()

    def categorize(self, hand: Hand, board: list[Card]) -> str:
        """
        Categorize hand strength on board.

        Args:
            hand: Player's hand
            board: Board cards (3-5)

        Returns:
            Category string
        """
        hand_treys = hand.to_treys()
        board_treys = [c.to_treys() for c in board]

        # Pad board to 5 cards if needed
        while len(board_treys) < 5:
            # Use placeholder - won't affect relative ranking
            board_treys.append(board_treys[0])

        rank = self.evaluator.evaluate(hand_treys, board_treys[:5])
        rank_class = self.evaluator.get_rank_class(rank)

        # Treys rank classes: 1=straight flush ... 9=high card
        if rank_class <= 3:  # Straight flush, quads, full house
            return "monster"
        elif rank_class <= 5:  # Flush, straight
            return "strong"
        elif rank_class <= 6:  # Three of a kind
            return "strong"
        elif rank_class == 7:  # Two pair
            return "strong"
        elif rank_class == 8:  # One pair
            return self._categorize_pair(hand, board)
        else:  # High card
            return self._categorize_air(hand, board)

    def _categorize_pair(self, hand: Hand, board: list[Card]) -> str:
        """Categorize one-pair hands."""
        board_ranks = sorted([c.rank for c in board], reverse=True)
        hand_ranks = [hand.card1.rank, hand.card2.rank]

        # Check if we have top pair
        if max(hand_ranks) == board_ranks[0]:
            if min(hand_ranks) >= 10:  # Good kicker
                return "strong"
            else:
                return "medium"
        elif max(hand_ranks) == board_ranks[1]:
            return "medium"
        else:
            return "weak"

    def _categorize_air(self, hand: Hand, board: list[Card]) -> str:
        """Categorize no-pair hands."""
        # Check for draws
        if self._has_flush_draw(hand, board):
            return "weak"
        if self._has_straight_draw(hand, board):
            return "weak"
        return "air"

    def _has_flush_draw(self, hand: Hand, board: list[Card]) -> bool:
        """Check for flush draw."""
        if len(board) < 3:
            return False

        suits = [c.suit for c in board] + [hand.card1.suit, hand.card2.suit]
        for suit in range(4):
            if suits.count(suit) >= 4:
                return True
        return False

    def _has_straight_draw(self, hand: Hand, board: list[Card]) -> bool:
        """Check for open-ended straight draw."""
        ranks = sorted(set(
            [c.rank for c in board] + [hand.card1.rank, hand.card2.rank]
        ))

        # Check for 4 consecutive ranks
        for i in range(len(ranks) - 3):
            if ranks[i + 3] - ranks[i] == 3:
                return True
        return False
