"""Game tree representation for poker."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class NodeType(Enum):
    """Types of nodes in a game tree."""
    ROOT = auto()
    PLAYER = auto()       # Player decision node
    CHANCE = auto()       # Chance node (card deal)
    TERMINAL = auto()     # End of hand (showdown or fold)


class ActionType(Enum):
    """Available actions at a decision node."""
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()
    ALL_IN = auto()


@dataclass
class Action:
    """An action in the game tree."""
    action_type: ActionType
    amount: float = 0.0  # Bet/raise amount in BBs

    def __str__(self) -> str:
        if self.amount > 0:
            return f"{self.action_type.name}_{self.amount:.1f}"
        return self.action_type.name


@dataclass
class GameNode:
    """
    A node in the poker game tree.

    Represents a point in the hand where either:
    - A player must make a decision (PLAYER)
    - A card is dealt (CHANCE)
    - The hand ends (TERMINAL)
    """
    node_type: NodeType
    player: int = 0              # 0 = IP (Button), 1 = OOP (BB)
    pot: float = 0.0             # Current pot in BBs
    stack: float = 100.0         # Effective stack in BBs
    to_call: float = 0.0         # Amount to call
    street: int = 0              # 0=preflop, 1=flop, 2=turn, 3=river

    # For terminal nodes
    winner: Optional[int] = None  # Player who won (if known)
    showdown: bool = False        # True if went to showdown

    # Tree structure
    parent: Optional["GameNode"] = None
    children: dict[Action, "GameNode"] = field(default_factory=dict)
    action_history: list[Action] = field(default_factory=list)

    # Information set identifier (for strategy indexing)
    info_set_key: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.node_type == NodeType.TERMINAL

    @property
    def is_chance(self) -> bool:
        return self.node_type == NodeType.CHANCE

    @property
    def is_player(self) -> bool:
        return self.node_type == NodeType.PLAYER

    def get_available_actions(self, bet_sizes: list[float]) -> list[Action]:
        """
        Get legal actions at this node.

        Args:
            bet_sizes: Available bet sizes as fractions of pot

        Returns:
            List of legal Action objects
        """
        if not self.is_player:
            return []

        actions = []

        # Can always fold if facing a bet
        if self.to_call > 0:
            actions.append(Action(ActionType.FOLD))
            actions.append(Action(ActionType.CALL, self.to_call))
        else:
            # Can check if not facing a bet
            actions.append(Action(ActionType.CHECK))

        # Can bet/raise if we have chips
        remaining_stack = self.stack - self.to_call
        if remaining_stack > 0:
            for size in bet_sizes:
                bet_amount = size * self.pot
                # Ensure minimum raise
                min_raise = max(self.to_call * 2, 1.0)
                if bet_amount >= min_raise and bet_amount < remaining_stack:
                    if self.to_call > 0:
                        actions.append(Action(ActionType.RAISE, bet_amount))
                    else:
                        actions.append(Action(ActionType.BET, bet_amount))

            # All-in option
            if remaining_stack > 0:
                actions.append(Action(ActionType.ALL_IN, remaining_stack))

        return actions

    def make_child(self, action: Action) -> "GameNode":
        """Create a child node after taking an action."""
        new_pot = self.pot
        new_to_call = 0.0
        new_player = 1 - self.player  # Switch players
        new_type = NodeType.PLAYER

        if action.action_type == ActionType.FOLD:
            new_type = NodeType.TERMINAL
            winner = 1 - self.player  # Opponent wins

            child = GameNode(
                node_type=new_type,
                player=new_player,
                pot=new_pot,
                stack=self.stack,
                to_call=0,
                street=self.street,
                winner=winner,
                showdown=False,
                parent=self,
                action_history=self.action_history + [action],
            )

        elif action.action_type == ActionType.CALL:
            new_pot = self.pot + self.to_call

            # Check if this ends the street
            if self._is_street_ending(action):
                if self.street == 3:  # River
                    new_type = NodeType.TERMINAL
                    child = GameNode(
                        node_type=new_type,
                        player=0,
                        pot=new_pot,
                        stack=self.stack - self.to_call,
                        to_call=0,
                        street=self.street,
                        showdown=True,
                        parent=self,
                        action_history=self.action_history + [action],
                    )
                else:
                    # Move to next street (chance node)
                    new_type = NodeType.CHANCE
                    child = GameNode(
                        node_type=new_type,
                        player=0,
                        pot=new_pot,
                        stack=self.stack - self.to_call,
                        to_call=0,
                        street=self.street + 1,
                        parent=self,
                        action_history=self.action_history + [action],
                    )
            else:
                child = GameNode(
                    node_type=new_type,
                    player=new_player,
                    pot=new_pot,
                    stack=self.stack - self.to_call,
                    to_call=0,
                    street=self.street,
                    parent=self,
                    action_history=self.action_history + [action],
                )

        elif action.action_type == ActionType.CHECK:
            # Check if this ends the street
            if self._is_street_ending(action):
                if self.street == 3:  # River
                    new_type = NodeType.TERMINAL
                    child = GameNode(
                        node_type=new_type,
                        player=0,
                        pot=new_pot,
                        stack=self.stack,
                        to_call=0,
                        street=self.street,
                        showdown=True,
                        parent=self,
                        action_history=self.action_history + [action],
                    )
                else:
                    new_type = NodeType.CHANCE
                    child = GameNode(
                        node_type=new_type,
                        player=0,
                        pot=new_pot,
                        stack=self.stack,
                        to_call=0,
                        street=self.street + 1,
                        parent=self,
                        action_history=self.action_history + [action],
                    )
            else:
                child = GameNode(
                    node_type=new_type,
                    player=new_player,
                    pot=new_pot,
                    stack=self.stack,
                    to_call=0,
                    street=self.street,
                    parent=self,
                    action_history=self.action_history + [action],
                )

        elif action.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            bet_amount = action.amount
            new_pot = self.pot + bet_amount
            new_to_call = bet_amount

            child = GameNode(
                node_type=new_type,
                player=new_player,
                pot=new_pot,
                stack=self.stack - bet_amount,
                to_call=new_to_call,
                street=self.street,
                parent=self,
                action_history=self.action_history + [action],
            )

        else:
            raise ValueError(f"Unknown action type: {action.action_type}")

        self.children[action] = child
        return child

    def _is_street_ending(self, action: Action) -> bool:
        """Check if this action ends the current street."""
        # Street ends when both players have acted and action is closed
        # Simplified: check if both players have had a chance to act
        if len(self.action_history) == 0:
            return False

        # Count actions on current street
        street_actions = [
            a for a in self.action_history
            if a.action_type not in (ActionType.FOLD,)
        ]

        # Both players have acted if we have at least one action from each
        return len(street_actions) >= 1


class GameTree:
    """
    Complete game tree for a poker scenario.

    Builds and traverses game trees for solver computations.
    """

    def __init__(
        self,
        starting_pot: float = 6.5,     # BBs (typical SRP pot)
        effective_stack: float = 100,   # BBs
        starting_street: int = 1,       # 1 = flop
    ):
        """
        Initialize game tree.

        Args:
            starting_pot: Initial pot size in BBs
            effective_stack: Effective stack size in BBs
            starting_street: Starting street (1=flop, 2=turn, 3=river)
        """
        self.starting_pot = starting_pot
        self.effective_stack = effective_stack
        self.starting_street = starting_street
        self.root: Optional[GameNode] = None

    def build(self, bet_sizes: list[float]) -> GameNode:
        """
        Build the game tree with given bet sizes.

        Args:
            bet_sizes: Available bet sizes as fractions of pot

        Returns:
            Root node of the game tree
        """
        self.root = GameNode(
            node_type=NodeType.PLAYER,
            player=0,  # IP acts first on flop
            pot=self.starting_pot,
            stack=self.effective_stack,
            to_call=0,
            street=self.starting_street,
        )

        self._build_subtree(self.root, bet_sizes, depth=0)
        return self.root

    def _build_subtree(
        self,
        node: GameNode,
        bet_sizes: list[float],
        depth: int,
        max_depth: int = 20,
    ) -> None:
        """Recursively build subtree from node."""
        if depth > max_depth or node.is_terminal:
            return

        if node.is_chance:
            # For now, just transition to next player decision
            # Full implementation would enumerate possible runouts
            next_node = GameNode(
                node_type=NodeType.PLAYER,
                player=0,  # OOP acts first postflop
                pot=node.pot,
                stack=node.stack,
                to_call=0,
                street=node.street,
                parent=node,
                action_history=node.action_history.copy(),
            )
            node.children[Action(ActionType.CHECK)] = next_node
            self._build_subtree(next_node, bet_sizes, depth + 1, max_depth)
            return

        # Get available actions
        actions = node.get_available_actions(bet_sizes)

        for action in actions:
            child = node.make_child(action)
            self._build_subtree(child, bet_sizes, depth + 1, max_depth)

    def count_nodes(self) -> dict[str, int]:
        """Count nodes by type."""
        counts = {
            "total": 0,
            "player": 0,
            "chance": 0,
            "terminal": 0,
        }

        def count_recursive(node: GameNode):
            counts["total"] += 1
            if node.is_player:
                counts["player"] += 1
            elif node.is_chance:
                counts["chance"] += 1
            elif node.is_terminal:
                counts["terminal"] += 1

            for child in node.children.values():
                count_recursive(child)

        if self.root:
            count_recursive(self.root)

        return counts

    def get_terminal_nodes(self) -> list[GameNode]:
        """Get all terminal nodes in the tree."""
        terminals = []

        def collect_terminals(node: GameNode):
            if node.is_terminal:
                terminals.append(node)
            for child in node.children.values():
                collect_terminals(child)

        if self.root:
            collect_terminals(self.root)

        return terminals
