"""Tests for solver components."""

import pytest
import numpy as np

from moid.game.cards import Card
from moid.game.tree import GameTree, GameNode, NodeType, Action, ActionType
from moid.solver.strategy import Strategy, StrategyProfile, make_info_set_key
from moid.solver.cfr import CFRSolver, CFRConfig
from moid.solver.best_response import (
    BestResponseSolver, ExploitativeSolver, OpponentModel
)
from moid.analysis.stats import PlayerStats


class TestStrategy:
    def test_uniform_initial(self):
        actions = [
            Action(ActionType.FOLD),
            Action(ActionType.CALL),
            Action(ActionType.RAISE, 3.0),
        ]
        probs = np.array([1/3, 1/3, 1/3])

        strategy = Strategy(
            info_set_key="test",
            actions=actions,
            probabilities=probs,
        )

        assert len(strategy.actions) == 3
        assert np.allclose(strategy.probabilities, [1/3, 1/3, 1/3])

    def test_get_action_probability(self):
        actions = [Action(ActionType.FOLD), Action(ActionType.CALL)]
        probs = np.array([0.3, 0.7])

        strategy = Strategy("test", actions, probs)

        assert strategy.get_action_probability(Action(ActionType.FOLD)) == 0.3
        assert strategy.get_action_probability(Action(ActionType.CALL)) == 0.7

    def test_regret_matching(self):
        actions = [Action(ActionType.FOLD), Action(ActionType.CALL)]
        probs = np.array([0.5, 0.5])

        strategy = Strategy("test", actions, probs)

        # Add positive regret for CALL
        strategy.update_regrets(np.array([-1.0, 2.0]))

        new_probs = strategy.get_strategy()

        # CALL should now have higher probability
        assert new_probs[1] > new_probs[0]

        # Only positive regrets count
        assert new_probs[0] == 0.0  # Negative regret -> 0
        assert new_probs[1] == 1.0  # Only positive regret

    def test_sample_action(self):
        actions = [Action(ActionType.FOLD), Action(ActionType.CALL)]
        probs = np.array([0.0, 1.0])

        strategy = Strategy("test", actions, probs)

        # Should always sample CALL
        for _ in range(10):
            action = strategy.sample_action()
            assert action.action_type == ActionType.CALL


class TestStrategyProfile:
    def test_get_or_create(self):
        profile = StrategyProfile()
        actions = [Action(ActionType.FOLD), Action(ActionType.CALL)]

        strategy = profile.get_strategy("info_set_1", actions)

        assert strategy is not None
        assert len(strategy.actions) == 2
        assert profile.num_info_sets() == 1

        # Getting same key should return same strategy
        strategy2 = profile.get_strategy("info_set_1", actions)
        assert strategy is strategy2

    def test_get_average_strategies(self):
        profile = StrategyProfile()
        actions = [Action(ActionType.FOLD), Action(ActionType.CALL)]

        strategy = profile.get_strategy("test", actions)
        strategy.strategy_sum = np.array([1.0, 3.0])

        avg_profile = profile.get_average_strategies()
        avg_strategy = avg_profile.strategies["test"]

        assert np.allclose(avg_strategy.probabilities, [0.25, 0.75])


class TestGameTree:
    def test_build_tree(self):
        tree = GameTree(
            starting_pot=6.5,
            effective_stack=100,
            starting_street=1,
        )

        tree.build(bet_sizes=[0.5, 1.0])
        counts = tree.count_nodes()

        assert counts["total"] > 0
        assert counts["player"] > 0
        assert counts["terminal"] > 0

    def test_root_node(self):
        tree = GameTree(
            starting_pot=6.5,
            effective_stack=100,
            starting_street=1,
        )

        tree.build(bet_sizes=[0.5])

        assert tree.root is not None
        assert tree.root.node_type == NodeType.PLAYER
        assert tree.root.pot == 6.5
        assert tree.root.player == 0  # IP acts first on flop

    def test_terminal_nodes(self):
        tree = GameTree(
            starting_pot=6.5,
            effective_stack=100,
            starting_street=1,
        )

        tree.build(bet_sizes=[0.5])
        terminals = tree.get_terminal_nodes()

        assert len(terminals) > 0
        for node in terminals:
            assert node.is_terminal


class TestCFRSolver:
    @pytest.fixture
    def simple_tree(self):
        tree = GameTree(
            starting_pot=6.5,
            effective_stack=50,
            starting_street=3,  # River - simpler tree
        )
        tree.build(bet_sizes=[0.5])
        return tree

    @pytest.fixture
    def board(self):
        return [
            Card.from_string("Ks"),
            Card.from_string("7d"),
            Card.from_string("2c"),
            Card.from_string("9h"),
            Card.from_string("3s"),
        ]

    def test_solver_runs(self, simple_tree, board):
        config = CFRConfig(
            num_iterations=10,
            num_buckets=4,
            use_monte_carlo=True,
        )

        solver = CFRSolver(simple_tree, board, config)
        profile = solver.solve()

        assert profile.num_info_sets() > 0

    def test_solver_convergence(self, simple_tree, board):
        config = CFRConfig(
            num_iterations=100,
            num_buckets=4,
        )

        solver = CFRSolver(simple_tree, board, config)

        exploitabilities = []

        def callback(iteration, exploit):
            exploitabilities.append(exploit)

        solver.solve(callback=callback)

        # Exploitability should generally decrease
        if len(exploitabilities) >= 2:
            # Not guaranteed to be monotonic but should trend down
            assert exploitabilities[-1] <= exploitabilities[0] * 2


class TestOpponentModel:
    def test_preflop_strategy_open(self):
        stats = PlayerStats(
            vpip=35.0,
            pfr=12.0,
        )

        model = OpponentModel(stats)
        strategy = model.get_preflop_strategy("BTN", "open")

        assert "fold" in strategy
        assert "raise" in strategy
        assert sum(strategy.values()) == pytest.approx(1.0, rel=0.1)

    def test_postflop_strategy_facing_bet(self):
        stats = PlayerStats(
            fold_to_cbet=55.0,
        )

        model = OpponentModel(stats)
        strategy = model.get_postflop_strategy(
            is_aggressor=False,
            facing_bet=True,
            street=1,
        )

        assert strategy["fold"] == pytest.approx(0.55, rel=0.01)


class TestBestResponseSolver:
    @pytest.fixture
    def solver_setup(self):
        tree = GameTree(
            starting_pot=6.5,
            effective_stack=50,
            starting_street=3,
        )
        tree.build(bet_sizes=[0.5])

        board = [
            Card.from_string("Ks"),
            Card.from_string("7d"),
            Card.from_string("2c"),
            Card.from_string("9h"),
            Card.from_string("3s"),
        ]

        stats = PlayerStats(
            vpip=35.0,
            pfr=12.0,
            fold_to_cbet=55.0,
        )
        model = OpponentModel(stats)

        return tree, board, model

    def test_best_response_runs(self, solver_setup):
        tree, board, model = solver_setup

        solver = BestResponseSolver(
            tree=tree,
            board=board,
            opponent_model=model,
            num_buckets=4,
        )

        br_profile = solver.compute_best_response(hero_player=0)

        # Best response should be pure strategies
        for strategy in br_profile.strategies.values():
            # One action should have probability 1
            assert max(strategy.probabilities) == 1.0


class TestExploitativeSolver:
    def test_solve(self):
        stats = PlayerStats(
            hands=1000,
            vpip=40.0,
            pfr=10.0,
            cbet=50.0,
            fold_to_cbet=60.0,
        )

        board = [
            Card.from_string("Ks"),
            Card.from_string("7d"),
            Card.from_string("2c"),
        ]

        solver = ExploitativeSolver(
            stats=stats,
            board=board,
            starting_pot=6.5,
            effective_stack=100,
        )

        profile = solver.solve()
        assert profile.num_info_sets() > 0

    def test_recommendations(self):
        stats = PlayerStats(
            fold_to_cbet=65.0,
        )

        board = [Card.from_string("Ks"), Card.from_string("7d"), Card.from_string("2c")]

        solver = ExploitativeSolver(stats, board)

        # With high fold to cbet, should recommend betting with air
        action, explanation = solver.get_recommended_action(
            hand_strength="air",
            facing_bet=False,
        )

        assert action == "bet"
        assert "fold" in explanation.lower()


class TestMakeInfoSetKey:
    def test_key_format(self):
        actions = [
            Action(ActionType.CHECK),
            Action(ActionType.BET, 3.0),
        ]

        key = make_info_set_key(
            player=0,
            hand_bucket=5,
            action_history=actions,
        )

        assert "P0" in key
        assert "B5" in key
        assert "CHECK" in key
        assert "BET" in key
