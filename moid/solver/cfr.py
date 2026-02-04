"""
Counterfactual Regret Minimization (CFR) solver.

CFR is an iterative algorithm for finding Nash equilibrium strategies
in extensive-form games. This implementation uses:
- Regret matching for strategy updates
- Monte Carlo sampling for efficiency (MCCFR)
- Chance sampling for board runouts
"""

from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from moid.game.cards import Hand, Card
from moid.game.tree import GameTree, GameNode, NodeType, Action
from moid.game.equity import EquityCalculator
from moid.game.abstraction import HandAbstraction
from .strategy import Strategy, StrategyProfile, make_info_set_key


@dataclass
class CFRConfig:
    """Configuration for CFR solver."""
    num_iterations: int = 10000
    num_buckets: int = 8           # Hand abstraction buckets
    use_monte_carlo: bool = True   # Use MCCFR
    sampling_prob: float = 0.3     # Exploration probability
    discount_interval: int = 0     # Iterations between discounting (0 = no discounting)
    discount_factor: float = 0.9   # Discount factor for older regrets

    # For alternating updates (can improve convergence)
    alternate_updates: bool = True


class CFRSolver:
    """
    Counterfactual Regret Minimization solver.

    Finds Nash equilibrium strategies through iterative self-play
    and regret minimization.

    Key concepts:
    - Regret: How much better we could have done by taking action a
    - Counterfactual value: EV at a node weighted by opponent's probability
    - Information set: What a player knows (their cards + action history)
    """

    def __init__(
        self,
        tree: GameTree,
        board: list[Card],
        config: Optional[CFRConfig] = None,
    ):
        """
        Initialize CFR solver.

        Args:
            tree: Game tree to solve
            board: Board cards for equity calculations
            config: Solver configuration
        """
        self.tree = tree
        self.board = board
        self.config = config or CFRConfig()

        self.profile = StrategyProfile()
        self.abstraction = HandAbstraction(num_buckets=self.config.num_buckets)
        self.equity_calc = EquityCalculator()

        # Track iterations
        self.iteration = 0

        # Build abstraction buckets for this board
        self.buckets = self.abstraction.compute_buckets(board)

    def solve(
        self,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> StrategyProfile:
        """
        Run CFR algorithm.

        Args:
            callback: Optional callback(iteration, exploitability) for progress

        Returns:
            Converged strategy profile
        """
        for i in range(self.config.num_iterations):
            self.iteration = i

            if self.config.use_monte_carlo:
                self._mccfr_iteration()
            else:
                self._vanilla_cfr_iteration()

            # Apply discounting periodically
            if (self.config.discount_interval > 0 and
                i > 0 and i % self.config.discount_interval == 0):
                self._apply_discounting()

            # Callback for progress tracking
            if callback and i % 100 == 0:
                exploit = self._estimate_exploitability()
                callback(i, exploit)

        return self.profile.get_average_strategies()

    def _vanilla_cfr_iteration(self) -> None:
        """Run one iteration of vanilla CFR."""
        # Sample hands for both players
        for bucket1 in range(self.config.num_buckets):
            for bucket2 in range(self.config.num_buckets):
                self._cfr(
                    self.tree.root,
                    bucket1,
                    bucket2,
                    1.0,  # Player 0 reach probability
                    1.0,  # Player 1 reach probability
                )

    def _mccfr_iteration(self) -> None:
        """
        Run one iteration of Monte Carlo CFR.

        Samples a single hand combination rather than iterating
        over all buckets.
        """
        # Sample hands
        bucket1 = np.random.randint(self.config.num_buckets)
        bucket2 = np.random.randint(self.config.num_buckets)

        # Alternate which player we update
        if self.config.alternate_updates:
            update_player = self.iteration % 2
            self._mccfr_traverse(
                self.tree.root,
                bucket1,
                bucket2,
                update_player,
            )
        else:
            # Update both players
            self._cfr(self.tree.root, bucket1, bucket2, 1.0, 1.0)

    def _cfr(
        self,
        node: GameNode,
        bucket0: int,  # Player 0's hand bucket
        bucket1: int,  # Player 1's hand bucket
        reach0: float,  # Player 0's reach probability
        reach1: float,  # Player 1's reach probability
    ) -> tuple[float, float]:
        """
        Recursive CFR traversal.

        Returns:
            Tuple of (player0_value, player1_value)
        """
        if node.is_terminal:
            return self._terminal_value(node, bucket0, bucket1)

        if node.is_chance:
            # For now, just continue (full impl would sample board cards)
            return self._cfr(
                list(node.children.values())[0],
                bucket0, bucket1, reach0, reach1
            )

        # Player decision node
        player = node.player
        my_bucket = bucket0 if player == 0 else bucket1
        my_reach = reach0 if player == 0 else reach1
        opp_reach = reach1 if player == 0 else reach0

        # Get or create strategy
        info_set_key = make_info_set_key(player, my_bucket, node.action_history)
        actions = list(node.children.keys())
        strategy = self.profile.get_strategy(info_set_key, actions)

        # Get current strategy (from regret matching)
        probs = strategy.get_strategy()
        strategy.probabilities = probs

        # Compute action values
        action_values = np.zeros(len(actions))

        for i, action in enumerate(actions):
            child = node.children[action]

            # Update reach probabilities
            if player == 0:
                new_reach0 = reach0 * probs[i]
                new_reach1 = reach1
            else:
                new_reach0 = reach0
                new_reach1 = reach1 * probs[i]

            child_values = self._cfr(child, bucket0, bucket1, new_reach0, new_reach1)
            action_values[i] = child_values[player]

        # Compute node value
        node_value = np.dot(probs, action_values)

        # Compute regrets
        regrets = action_values - node_value

        # Update regrets weighted by opponent's reach
        strategy.update_regrets(opp_reach * regrets)

        # Update strategy sum for average
        strategy.update_strategy_sum(my_reach)

        # Return values for both players
        if player == 0:
            return (node_value, -node_value)  # Zero-sum
        else:
            return (-node_value, node_value)

    def _mccfr_traverse(
        self,
        node: GameNode,
        bucket0: int,
        bucket1: int,
        update_player: int,
    ) -> float:
        """
        Monte Carlo CFR traversal.

        Only updates strategies for one player per iteration.

        Returns:
            Value for the update player
        """
        if node.is_terminal:
            values = self._terminal_value(node, bucket0, bucket1)
            return values[update_player]

        if node.is_chance:
            return self._mccfr_traverse(
                list(node.children.values())[0],
                bucket0, bucket1, update_player
            )

        player = node.player
        my_bucket = bucket0 if player == 0 else bucket1

        info_set_key = make_info_set_key(player, my_bucket, node.action_history)
        actions = list(node.children.keys())
        strategy = self.profile.get_strategy(info_set_key, actions)

        probs = strategy.get_strategy()
        strategy.probabilities = probs

        if player == update_player:
            # Traverse all actions, compute regrets
            action_values = np.zeros(len(actions))

            for i, action in enumerate(actions):
                child = node.children[action]
                action_values[i] = self._mccfr_traverse(
                    child, bucket0, bucket1, update_player
                )

            node_value = np.dot(probs, action_values)
            regrets = action_values - node_value

            strategy.update_regrets(regrets)
            strategy.update_strategy_sum(1.0)

            return node_value
        else:
            # Sample action according to strategy
            action_idx = np.random.choice(len(actions), p=probs)
            action = actions[action_idx]
            child = node.children[action]

            return self._mccfr_traverse(child, bucket0, bucket1, update_player)

    def _terminal_value(
        self,
        node: GameNode,
        bucket0: int,
        bucket1: int,
    ) -> tuple[float, float]:
        """
        Compute terminal node values.

        Values are computed as profit/loss from initial state:
        - When folding: Winner profits opponent's contribution
        - When showdown: EV = win_prob * pot - own_contribution

        Returns:
            Tuple of (player0_value, player1_value)
        """
        if node.winner is not None:
            # Someone folded - winner profits opponent's contribution
            # (which equals pot minus their own contribution)
            if node.winner == 0:
                # IP wins, OOP folded - IP profits OOP's contribution
                profit = node.contribution1
                return (profit, -profit)
            else:
                # OOP wins, IP folded - OOP profits IP's contribution
                profit = node.contribution0
                return (-profit, profit)

        # Showdown - compute equity-weighted value
        # Higher bucket = stronger hand (by our abstraction)
        eq0 = (bucket0 + 1) / (self.config.num_buckets + 1)
        eq1 = (bucket1 + 1) / (self.config.num_buckets + 1)

        # Normalize to win probabilities
        total = eq0 + eq1
        if total > 0:
            p0_wins = eq0 / total
        else:
            p0_wins = 0.5

        # EV = win_prob * pot - own_contribution
        pot = node.pot
        ev0 = p0_wins * pot - node.contribution0
        # ev1 = (1 - p0_wins) * pot - node.contribution1 = -ev0 (zero-sum)
        return (ev0, -ev0)

    def _apply_discounting(self) -> None:
        """Apply discount to older regrets and strategy sums."""
        factor = self.config.discount_factor
        for strategy in self.profile.strategies.values():
            strategy.regret_sum *= factor
            strategy.strategy_sum *= factor

    def _estimate_exploitability(self) -> float:
        """
        Estimate exploitability of current strategy.

        Exploitability = sum of best response values for each player.
        When it's 0, we've found a Nash equilibrium.
        """
        # Simplified exploitability estimate
        # Full implementation would compute exact best response

        total_regret = 0.0
        for strategy in self.profile.strategies.values():
            positive_regrets = np.maximum(strategy.regret_sum, 0)
            total_regret += positive_regrets.sum()

        # Normalize by number of info sets and iterations
        num_info_sets = max(1, len(self.profile.strategies))
        return total_regret / (num_info_sets * max(1, self.iteration))


def solve_spot(
    starting_pot: float,
    effective_stack: float,
    board: list[Card],
    bet_sizes: list[float],
    num_iterations: int = 10000,
) -> StrategyProfile:
    """
    Convenience function to solve a specific spot.

    Args:
        starting_pot: Initial pot in BBs
        effective_stack: Effective stack in BBs
        board: Board cards
        bet_sizes: Available bet sizes as pot fractions
        num_iterations: CFR iterations

    Returns:
        Solved strategy profile
    """
    tree = GameTree(
        starting_pot=starting_pot,
        effective_stack=effective_stack,
        starting_street=1 if len(board) == 3 else 2 if len(board) == 4 else 3,
    )
    tree.build(bet_sizes)

    config = CFRConfig(num_iterations=num_iterations)
    solver = CFRSolver(tree, board, config)

    return solver.solve()
