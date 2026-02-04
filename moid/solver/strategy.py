"""Strategy representation for poker solver."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from moid.game.tree import Action, GameNode


@dataclass
class Strategy:
    """
    Strategy for a single information set.

    An information set is identified by:
    - Player position
    - Hand bucket (abstracted)
    - Action history

    The strategy maps actions to probabilities.
    """
    info_set_key: str
    actions: list[Action]
    probabilities: np.ndarray  # Probability for each action

    # For regret matching
    regret_sum: np.ndarray = field(default=None)
    strategy_sum: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.regret_sum is None:
            self.regret_sum = np.zeros(len(self.actions))
        if self.strategy_sum is None:
            self.strategy_sum = np.zeros(len(self.actions))

    def get_action_probability(self, action: Action) -> float:
        """Get probability of taking an action."""
        try:
            idx = self.actions.index(action)
            return self.probabilities[idx]
        except ValueError:
            return 0.0

    def sample_action(self) -> Action:
        """Sample an action according to strategy."""
        idx = np.random.choice(len(self.actions), p=self.probabilities)
        return self.actions[idx]

    def update_regrets(self, regrets: np.ndarray) -> None:
        """
        Update cumulative regrets.

        Args:
            regrets: Regret for each action
        """
        self.regret_sum += regrets

    def get_strategy(self) -> np.ndarray:
        """
        Get current strategy from regrets using regret matching.

        Returns:
            Probability distribution over actions
        """
        positive_regrets = np.maximum(self.regret_sum, 0)
        total = positive_regrets.sum()

        if total > 0:
            return positive_regrets / total
        else:
            # Uniform random if no positive regrets
            return np.ones(len(self.actions)) / len(self.actions)

    def get_average_strategy(self) -> np.ndarray:
        """
        Get average strategy over all iterations.

        This converges to Nash equilibrium in two-player zero-sum games.

        Returns:
            Average probability distribution
        """
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.ones(len(self.actions)) / len(self.actions)

    def update_strategy_sum(self, realization_weight: float) -> None:
        """
        Update strategy sum for average computation.

        Args:
            realization_weight: Probability of reaching this info set
        """
        self.strategy_sum += realization_weight * self.probabilities

    def __repr__(self) -> str:
        action_strs = [
            f"{a}: {p:.2f}"
            for a, p in zip(self.actions, self.probabilities)
        ]
        return f"Strategy({self.info_set_key}, {{{', '.join(action_strs)}}})"


class StrategyProfile:
    """
    Complete strategy profile for all information sets.

    Maps information set keys to Strategy objects.
    """

    def __init__(self):
        self.strategies: dict[str, Strategy] = {}

    def get_strategy(
        self,
        info_set_key: str,
        actions: list[Action],
    ) -> Strategy:
        """
        Get or create strategy for an information set.

        Args:
            info_set_key: Unique identifier for info set
            actions: Available actions at this info set

        Returns:
            Strategy object
        """
        if info_set_key not in self.strategies:
            # Initialize with uniform random
            probs = np.ones(len(actions)) / len(actions)
            self.strategies[info_set_key] = Strategy(
                info_set_key=info_set_key,
                actions=actions,
                probabilities=probs,
            )
        return self.strategies[info_set_key]

    def set_strategy(self, info_set_key: str, strategy: Strategy) -> None:
        """Set strategy for an information set."""
        self.strategies[info_set_key] = strategy

    def get_action_probabilities(
        self,
        info_set_key: str,
    ) -> Optional[dict[Action, float]]:
        """
        Get action probabilities for an info set.

        Returns:
            Dict mapping actions to probabilities, or None if not found
        """
        if info_set_key not in self.strategies:
            return None

        strategy = self.strategies[info_set_key]
        return dict(zip(strategy.actions, strategy.probabilities))

    def num_info_sets(self) -> int:
        """Get number of information sets."""
        return len(self.strategies)

    def update_all_strategies(self) -> None:
        """Update all strategies from their regrets."""
        for strategy in self.strategies.values():
            strategy.probabilities = strategy.get_strategy()

    def get_average_strategies(self) -> "StrategyProfile":
        """
        Get a new profile with average strategies.

        Returns:
            StrategyProfile with averaged strategies
        """
        avg_profile = StrategyProfile()
        for key, strategy in self.strategies.items():
            avg_probs = strategy.get_average_strategy()
            avg_profile.strategies[key] = Strategy(
                info_set_key=key,
                actions=strategy.actions.copy(),
                probabilities=avg_probs,
            )
        return avg_profile

    def save(self, filepath: str) -> None:
        """Save strategy profile to file."""
        import json

        data = {}
        for key, strategy in self.strategies.items():
            data[key] = {
                "actions": [str(a) for a in strategy.actions],
                "probabilities": strategy.probabilities.tolist(),
                "regret_sum": strategy.regret_sum.tolist(),
                "strategy_sum": strategy.strategy_sum.tolist(),
            }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "StrategyProfile":
        """Load strategy profile from file."""
        import json
        from moid.game.tree import ActionType

        profile = cls()

        with open(filepath, "r") as f:
            data = json.load(f)

        for key, strat_data in data.items():
            # Parse actions from strings
            actions = []
            for action_str in strat_data["actions"]:
                parts = action_str.split("_")
                action_type = ActionType[parts[0]]
                amount = float(parts[1]) if len(parts) > 1 else 0.0
                actions.append(Action(action_type, amount))

            strategy = Strategy(
                info_set_key=key,
                actions=actions,
                probabilities=np.array(strat_data["probabilities"]),
                regret_sum=np.array(strat_data["regret_sum"]),
                strategy_sum=np.array(strat_data["strategy_sum"]),
            )
            profile.strategies[key] = strategy

        return profile


def make_info_set_key(
    player: int,
    hand_bucket: int,
    action_history: list[Action],
) -> str:
    """
    Create information set key.

    Args:
        player: Player index (0 or 1)
        hand_bucket: Abstracted hand bucket index
        action_history: List of actions taken

    Returns:
        Unique string key for this info set
    """
    history_str = ":".join(str(a) for a in action_history)
    return f"P{player}|B{hand_bucket}|{history_str}"
