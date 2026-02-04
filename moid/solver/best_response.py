"""
Best response and exploitative strategy computation.

Given an opponent's (potentially suboptimal) strategy,
computes the maximally exploitative counter-strategy.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from moid.game.cards import Hand, Card
from moid.game.tree import GameTree, GameNode, Action, ActionType
from moid.game.abstraction import HandAbstraction
from moid.analysis.stats import PlayerStats
from .strategy import Strategy, StrategyProfile, make_info_set_key


@dataclass
class OpponentModel:
    """
    Model of opponent's strategy based on population stats.

    Converts population statistics into action probabilities
    for use in best response calculation.
    """
    stats: PlayerStats

    def get_preflop_strategy(self, position: str, facing_action: str) -> dict[str, float]:
        """
        Get preflop action probabilities.

        Args:
            position: Opponent's position
            facing_action: Action facing ("open", "raise", "3bet")

        Returns:
            Dict mapping actions to probabilities
        """
        if facing_action == "open":
            # Opening range based on VPIP/PFR
            fold_prob = 1 - self.stats.vpip / 100
            raise_prob = self.stats.pfr / 100
            limp_prob = max(0, (self.stats.vpip - self.stats.pfr) / 100)

            return {
                "fold": fold_prob,
                "raise": raise_prob,
                "limp": limp_prob,
            }

        elif facing_action == "raise":
            # Facing a raise: 3-bet, call, or fold
            three_bet_prob = self.stats.three_bet / 100
            fold_prob = self.stats.fold_to_3bet / 100 if self.stats.fold_to_3bet > 0 else 0.5
            call_prob = max(0, 1 - three_bet_prob - fold_prob)

            return {
                "fold": fold_prob,
                "call": call_prob,
                "raise": three_bet_prob,
            }

        return {"fold": 0.5, "call": 0.5}

    def get_postflop_strategy(
        self,
        is_aggressor: bool,
        facing_bet: bool,
        street: int,
    ) -> dict[str, float]:
        """
        Get postflop action probabilities.

        Args:
            is_aggressor: Was opponent the preflop aggressor
            facing_bet: Is opponent facing a bet
            street: 1=flop, 2=turn, 3=river

        Returns:
            Dict mapping actions to probabilities
        """
        if not facing_bet:
            if is_aggressor:
                # C-bet decision
                cbet_prob = self.stats.cbet / 100
                return {
                    "check": 1 - cbet_prob,
                    "bet": cbet_prob,
                }
            else:
                # Check or donk bet (rare)
                return {"check": 0.9, "bet": 0.1}

        else:
            # Facing a bet
            fold_prob = self.stats.fold_to_cbet / 100
            # Assume some call/raise split
            call_prob = (1 - fold_prob) * 0.85
            raise_prob = (1 - fold_prob) * 0.15

            return {
                "fold": fold_prob,
                "call": call_prob,
                "raise": raise_prob,
            }


class BestResponseSolver:
    """
    Computes best response to a fixed opponent strategy.

    Given opponent's strategy (from StrategyProfile or OpponentModel),
    finds the pure strategy that maximizes EV.
    """

    def __init__(
        self,
        tree: GameTree,
        board: list[Card],
        opponent_profile: Optional[StrategyProfile] = None,
        opponent_model: Optional[OpponentModel] = None,
        num_buckets: int = 8,
    ):
        """
        Initialize best response solver.

        Args:
            tree: Game tree
            board: Board cards
            opponent_profile: Opponent's explicit strategy profile
            opponent_model: Or opponent model from stats
            num_buckets: Hand abstraction buckets
        """
        self.tree = tree
        self.board = board
        self.opponent_profile = opponent_profile
        self.opponent_model = opponent_model
        self.abstraction = HandAbstraction(num_buckets=num_buckets)
        self.num_buckets = num_buckets

        if opponent_profile is None and opponent_model is None:
            raise ValueError("Must provide either opponent_profile or opponent_model")

    def compute_best_response(self, hero_player: int = 0) -> StrategyProfile:
        """
        Compute best response for hero.

        Args:
            hero_player: Hero's player index (0=IP, 1=OOP)

        Returns:
            Best response strategy profile
        """
        br_profile = StrategyProfile()

        # Compute BR for each hand bucket
        for bucket in range(self.num_buckets):
            self._compute_br_recursive(
                self.tree.root,
                hero_player,
                bucket,
                br_profile,
            )

        return br_profile

    def _compute_br_recursive(
        self,
        node: GameNode,
        hero_player: int,
        hero_bucket: int,
        br_profile: StrategyProfile,
    ) -> float:
        """
        Recursively compute best response values.

        Returns:
            Best response value at this node
        """
        if node.is_terminal:
            return self._terminal_value(node, hero_player, hero_bucket)

        if node.is_chance:
            # Average over possible runouts (simplified)
            return self._compute_br_recursive(
                list(node.children.values())[0],
                hero_player, hero_bucket, br_profile
            )

        actions = list(node.children.keys())

        if node.player == hero_player:
            # Hero's decision - maximize value
            info_set_key = make_info_set_key(hero_player, hero_bucket, node.action_history)

            action_values = np.zeros(len(actions))
            for i, action in enumerate(actions):
                child = node.children[action]
                action_values[i] = self._compute_br_recursive(
                    child, hero_player, hero_bucket, br_profile
                )

            # Best response: pure strategy on best action
            best_idx = np.argmax(action_values)
            probs = np.zeros(len(actions))
            probs[best_idx] = 1.0

            br_profile.strategies[info_set_key] = Strategy(
                info_set_key=info_set_key,
                actions=actions,
                probabilities=probs,
            )

            return action_values[best_idx]

        else:
            # Opponent's decision - use their strategy
            opp_probs = self._get_opponent_strategy(node)

            if len(opp_probs) != len(actions):
                # Default to uniform if mismatch
                opp_probs = np.ones(len(actions)) / len(actions)

            # Compute expected value
            ev = 0.0
            for i, action in enumerate(actions):
                child = node.children[action]
                child_value = self._compute_br_recursive(
                    child, hero_player, hero_bucket, br_profile
                )
                ev += opp_probs[i] * child_value

            return ev

    def _get_opponent_strategy(self, node: GameNode) -> np.ndarray:
        """Get opponent's strategy at a node."""
        actions = list(node.children.keys())

        if self.opponent_profile:
            # Use explicit profile
            info_set_key = f"P{node.player}|B0|{':'.join(str(a) for a in node.action_history)}"
            if info_set_key in self.opponent_profile.strategies:
                return self.opponent_profile.strategies[info_set_key].probabilities

        if self.opponent_model:
            # Convert model to action probabilities
            facing_bet = node.to_call > 0
            is_aggressor = False  # Simplified

            model_probs = self.opponent_model.get_postflop_strategy(
                is_aggressor, facing_bet, node.street
            )

            # Map to actual actions
            probs = np.zeros(len(actions))
            for i, action in enumerate(actions):
                if action.action_type == ActionType.FOLD:
                    probs[i] = model_probs.get("fold", 0)
                elif action.action_type == ActionType.CHECK:
                    probs[i] = model_probs.get("check", 0)
                elif action.action_type == ActionType.CALL:
                    probs[i] = model_probs.get("call", 0)
                elif action.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                    probs[i] = model_probs.get("bet", 0) + model_probs.get("raise", 0)

            # Normalize
            total = probs.sum()
            if total > 0:
                return probs / total

        # Default uniform
        return np.ones(len(actions)) / len(actions)

    def _terminal_value(
        self,
        node: GameNode,
        hero_player: int,
        hero_bucket: int,
    ) -> float:
        """Compute terminal node value for hero."""
        if node.winner is not None:
            pot = node.pot
            if node.winner == hero_player:
                return pot / 2
            else:
                return -pot / 2

        # Showdown - use bucket as proxy for equity
        hero_equity = (hero_bucket + 1) / (self.num_buckets + 1)
        pot = node.pot

        return hero_equity * pot - (1 - hero_equity) * pot


class ExploitativeSolver:
    """
    High-level interface for computing exploitative strategies.

    Combines population analysis with best response computation
    to find maximally exploitative plays against microstakes pools.
    """

    def __init__(
        self,
        stats: PlayerStats,
        board: list[Card],
        starting_pot: float = 6.5,
        effective_stack: float = 100,
    ):
        """
        Initialize exploitative solver.

        Args:
            stats: Population or player statistics
            board: Board cards
            starting_pot: Initial pot in BBs
            effective_stack: Effective stack in BBs
        """
        self.stats = stats
        self.board = board
        self.starting_pot = starting_pot
        self.effective_stack = effective_stack

        self.opponent_model = OpponentModel(stats)

    def solve(self, bet_sizes: list[float] = [0.5, 0.75, 1.0]) -> StrategyProfile:
        """
        Compute exploitative strategy.

        Args:
            bet_sizes: Available bet sizes as pot fractions

        Returns:
            Exploitative strategy profile
        """
        # Build game tree
        street = 1 if len(self.board) == 3 else 2 if len(self.board) == 4 else 3
        tree = GameTree(
            starting_pot=self.starting_pot,
            effective_stack=self.effective_stack,
            starting_street=street,
        )
        tree.build(bet_sizes)

        # Compute best response
        br_solver = BestResponseSolver(
            tree=tree,
            board=self.board,
            opponent_model=self.opponent_model,
        )

        return br_solver.compute_best_response(hero_player=0)

    def get_recommended_action(
        self,
        hand_strength: str,  # "strong", "medium", "weak", "air"
        facing_bet: bool,
        pot_odds: float = 0.0,
    ) -> tuple[str, str]:
        """
        Get recommended action and explanation.

        Args:
            hand_strength: Abstracted hand strength category
            facing_bet: Whether we're facing a bet
            pot_odds: Pot odds if facing bet (0-1)

        Returns:
            Tuple of (action, explanation)
        """
        fold_to_cbet = self.stats.fold_to_cbet / 100

        if not facing_bet:
            # Betting decision
            if hand_strength in ("strong", "medium"):
                return (
                    "bet",
                    f"Population folds {fold_to_cbet:.0%} to c-bets. "
                    "Bet for value and to deny equity."
                )
            elif hand_strength == "weak":
                if fold_to_cbet > 0.55:
                    return (
                        "bet",
                        f"Population overfolds ({fold_to_cbet:.0%}). "
                        "Bluff profitable even with weak hands."
                    )
                else:
                    return (
                        "check",
                        f"Population doesn't fold enough ({fold_to_cbet:.0%}). "
                        "Check to realize equity."
                    )
            else:  # air
                if fold_to_cbet > 0.60:
                    return (
                        "bet",
                        f"Population folds {fold_to_cbet:.0%}. "
                        "Profitable bluff spot."
                    )
                return (
                    "check",
                    "Don't have equity or fold equity. Give up."
                )

        else:
            # Facing a bet decision
            call_ev = pot_odds

            if hand_strength == "strong":
                return (
                    "raise",
                    "We have a strong hand. Raise for value."
                )
            elif hand_strength == "medium":
                return (
                    "call",
                    "Medium strength. Call and re-evaluate."
                )
            elif hand_strength == "weak":
                if call_ev > 0.35:  # Good pot odds
                    return (
                        "call",
                        f"Getting {call_ev:.0%} pot odds with draw. Call."
                    )
                return (
                    "fold",
                    "Weak hand without pot odds. Fold."
                )
            else:
                return (
                    "fold",
                    "No equity against betting range. Fold."
                )
