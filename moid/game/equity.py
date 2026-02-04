"""Equity calculation utilities."""

from typing import Optional

import numpy as np
from treys import Card as TreysCard, Evaluator

from .cards import Card, Hand, Deck


class EquityCalculator:
    """
    Fast equity calculations using treys library.

    Supports exact enumeration for small cases and
    Monte Carlo simulation for larger ones.
    """

    def __init__(self):
        self.evaluator = Evaluator()
        self._rank_cache: dict[tuple, int] = {}

    def hand_vs_hand(
        self,
        hand1: Hand,
        hand2: Hand,
        board: list[Card],
        num_simulations: int = 10000,
    ) -> tuple[float, float, float]:
        """
        Calculate equity of hand1 vs hand2 on a board.

        Args:
            hand1: First hand
            hand2: Second hand
            board: Board cards (0-5)
            num_simulations: Number of Monte Carlo simulations

        Returns:
            Tuple of (hand1_equity, hand2_equity, tie_equity)
        """
        h1_treys = hand1.to_treys()
        h2_treys = hand2.to_treys()
        board_treys = [c.to_treys() for c in board]

        # Check for card collisions
        all_cards = set(h1_treys + h2_treys + board_treys)
        if len(all_cards) != len(h1_treys) + len(h2_treys) + len(board_treys):
            raise ValueError("Duplicate cards detected")

        remaining = 5 - len(board_treys)

        if remaining == 0:
            # Exact evaluation
            r1 = self.evaluator.evaluate(h1_treys, board_treys)
            r2 = self.evaluator.evaluate(h2_treys, board_treys)
            if r1 < r2:
                return (1.0, 0.0, 0.0)
            elif r1 > r2:
                return (0.0, 1.0, 0.0)
            else:
                return (0.5, 0.5, 0.0)

        # Monte Carlo simulation
        deck = Deck()
        used = set(all_cards)
        available = [c for c in deck.cards if c.to_treys() not in used]

        wins1 = wins2 = ties = 0

        for _ in range(num_simulations):
            np.random.shuffle(available)
            runout = [available[i].to_treys() for i in range(remaining)]
            full_board = board_treys + runout

            r1 = self.evaluator.evaluate(h1_treys, full_board)
            r2 = self.evaluator.evaluate(h2_treys, full_board)

            if r1 < r2:
                wins1 += 1
            elif r1 > r2:
                wins2 += 1
            else:
                ties += 1

        total = num_simulations
        return (wins1 / total, wins2 / total, ties / total)

    def hand_vs_range(
        self,
        hand: Hand,
        range_hands: list[Hand],
        board: list[Card],
        num_simulations: int = 1000,
    ) -> float:
        """
        Calculate equity of hand vs a range of hands.

        Args:
            hand: Hero's hand
            range_hands: List of villain's possible hands
            board: Board cards
            num_simulations: Simulations per matchup

        Returns:
            Hero's equity (0-1)
        """
        if not range_hands:
            return 0.5

        total_equity = 0.0
        valid_matchups = 0

        for opp_hand in range_hands:
            # Skip if cards overlap with hero
            opp_cards = {opp_hand.card1.to_treys(), opp_hand.card2.to_treys()}
            hero_cards = {hand.card1.to_treys(), hand.card2.to_treys()}
            board_cards = {c.to_treys() for c in board}

            if opp_cards & (hero_cards | board_cards):
                continue

            eq, _, _ = self.hand_vs_hand(hand, opp_hand, board, num_simulations)
            total_equity += eq
            valid_matchups += 1

        if valid_matchups == 0:
            return 0.5

        return total_equity / valid_matchups

    def range_vs_range(
        self,
        range1: list[Hand],
        range2: list[Hand],
        board: list[Card],
        num_samples: int = 1000,
    ) -> float:
        """
        Calculate equity of range1 vs range2.

        Uses sampling for large ranges.

        Args:
            range1: First range
            range2: Second range
            board: Board cards
            num_samples: Number of matchups to sample

        Returns:
            range1's equity (0-1)
        """
        if not range1 or not range2:
            return 0.5

        total_equity = 0.0
        valid_samples = 0

        for _ in range(num_samples):
            h1 = range1[np.random.randint(len(range1))]
            h2 = range2[np.random.randint(len(range2))]

            # Check for card overlap
            h1_cards = {h1.card1.to_treys(), h1.card2.to_treys()}
            h2_cards = {h2.card1.to_treys(), h2.card2.to_treys()}
            board_cards = {c.to_treys() for c in board}

            if h1_cards & h2_cards or h1_cards & board_cards or h2_cards & board_cards:
                continue

            eq, _, _ = self.hand_vs_hand(h1, h2, board, 100)
            total_equity += eq
            valid_samples += 1

        if valid_samples == 0:
            return 0.5

        return total_equity / valid_samples


def calculate_equity(
    hand: Hand,
    board: list[Card],
    num_opponents: int = 1,
    num_simulations: int = 10000,
) -> float:
    """
    Calculate hand equity against random opponent(s).

    Args:
        hand: Hero's hand
        board: Board cards
        num_opponents: Number of opponents
        num_simulations: Number of simulations

    Returns:
        Equity (0-1)
    """
    evaluator = Evaluator()
    hand_treys = hand.to_treys()
    board_treys = [c.to_treys() for c in board]

    # Build available cards
    used = set(hand_treys + board_treys)
    deck = Deck()
    available = [c for c in deck.cards if c.to_treys() not in used]

    remaining_board = 5 - len(board_treys)
    cards_needed = num_opponents * 2 + remaining_board

    wins = ties = 0

    for _ in range(num_simulations):
        np.random.shuffle(available)

        # Deal remaining board
        full_board = board_treys + [
            available[i].to_treys() for i in range(remaining_board)
        ]

        # Deal opponent hands
        idx = remaining_board
        opp_ranks = []
        for _ in range(num_opponents):
            opp_hand = [available[idx].to_treys(), available[idx + 1].to_treys()]
            opp_ranks.append(evaluator.evaluate(opp_hand, full_board))
            idx += 2

        hero_rank = evaluator.evaluate(hand_treys, full_board)
        best_opp = min(opp_ranks)

        if hero_rank < best_opp:
            wins += 1
        elif hero_rank == best_opp:
            ties += 0.5

    return (wins + ties) / num_simulations


def calculate_hand_strength(hand: Hand, board: list[Card]) -> int:
    """
    Calculate absolute hand strength (treys rank).

    Lower is better.

    Args:
        hand: Hand to evaluate
        board: Board cards (must be 5)

    Returns:
        Hand rank (1 is best, 7462 is worst)
    """
    if len(board) != 5:
        raise ValueError("Board must have exactly 5 cards")

    evaluator = Evaluator()
    hand_treys = hand.to_treys()
    board_treys = [c.to_treys() for c in board]

    return evaluator.evaluate(hand_treys, board_treys)


def get_hand_class(hand: Hand, board: list[Card]) -> str:
    """
    Get the hand class (e.g., "Two Pair", "Flush").

    Args:
        hand: Hand to evaluate
        board: Board cards (5)

    Returns:
        Hand class string
    """
    if len(board) != 5:
        raise ValueError("Board must have exactly 5 cards")

    evaluator = Evaluator()
    hand_treys = hand.to_treys()
    board_treys = [c.to_treys() for c in board]

    rank = evaluator.evaluate(hand_treys, board_treys)
    return evaluator.class_to_string(evaluator.get_rank_class(rank))
