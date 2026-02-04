"""Data models for parsed hand histories."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional


class Position(Enum):
    """Player positions at a 6-max table."""
    UTG = 0
    UTG1 = 1  # UTG+1
    CO = 2    # Cutoff
    BTN = 3   # Button
    SB = 4    # Small Blind
    BB = 5    # Big Blind

    @classmethod
    def from_string(cls, s: str) -> "Position":
        """Parse position from Ignition format string."""
        s = s.upper().strip()
        # Remove [ME] suffix if present
        s = s.replace("[ME]", "").strip()

        mapping = {
            "UTG": cls.UTG,
            "UTG+1": cls.UTG1,
            "CO": cls.CO,
            "CUTOFF": cls.CO,
            "BTN": cls.BTN,
            "BUTTON": cls.BTN,
            "DEALER": cls.BTN,
            "SB": cls.SB,
            "SMALL BLIND": cls.SB,
            "BB": cls.BB,
            "BIG BLIND": cls.BB,
        }
        if s in mapping:
            return mapping[s]
        raise ValueError(f"Unknown position: {s}")

    def __str__(self) -> str:
        return self.name


class Street(Enum):
    """Betting streets."""
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

    @classmethod
    def from_string(cls, s: str) -> "Street":
        """Parse street from string."""
        s = s.upper().strip()
        mapping = {
            "PREFLOP": cls.PREFLOP,
            "PRE-FLOP": cls.PREFLOP,
            "FLOP": cls.FLOP,
            "TURN": cls.TURN,
            "RIVER": cls.RIVER,
        }
        if s in mapping:
            return mapping[s]
        raise ValueError(f"Unknown street: {s}")


class ActionType(Enum):
    """Types of player actions."""
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()
    ALL_IN = auto()
    POST_SB = auto()
    POST_BB = auto()
    POST_ANTE = auto()

    @classmethod
    def from_string(cls, s: str) -> "ActionType":
        """Parse action type from string."""
        s = s.lower().strip()

        if "fold" in s:
            return cls.FOLD
        if "check" in s:
            return cls.CHECK
        if "call" in s:
            return cls.CALL
        if "bet" in s:
            return cls.BET
        if "raise" in s:
            return cls.RAISE
        if "all-in" in s or "allin" in s or "all in" in s:
            return cls.ALL_IN
        if "posts small blind" in s or "small blind" in s:
            return cls.POST_SB
        if "posts big blind" in s or "big blind" in s:
            return cls.POST_BB
        if "ante" in s:
            return cls.POST_ANTE

        raise ValueError(f"Unknown action type: {s}")


@dataclass
class Action:
    """A single player action."""
    position: Position
    street: Street
    action_type: ActionType
    amount: float = 0.0  # In big blinds or dollars depending on context
    is_all_in: bool = False

    def __repr__(self) -> str:
        if self.amount > 0:
            return f"{self.position.name} {self.action_type.name} {self.amount:.2f}"
        return f"{self.position.name} {self.action_type.name}"


@dataclass
class Player:
    """A player in a hand."""
    position: Position
    stack: float  # Starting stack in BBs
    hole_cards: Optional[tuple[str, str]] = None  # e.g., ("As", "Kh")
    result: float = 0.0  # Net result in BBs (+ for win, - for loss)
    is_hero: bool = False
    showed_cards: bool = False

    def __repr__(self) -> str:
        cards = f" [{self.hole_cards[0]}{self.hole_cards[1]}]" if self.hole_cards else ""
        return f"{self.position.name}({self.stack:.1f}bb){cards}"


@dataclass
class Hand:
    """A complete poker hand."""
    hand_id: str
    timestamp: datetime
    stakes: tuple[float, float]  # (small_blind, big_blind) in dollars
    table_name: str = ""

    # Board cards by street
    board: list[str] = field(default_factory=list)  # e.g., ["As", "Kh", "7d", "2c", "9s"]

    # Players in the hand
    players: list[Player] = field(default_factory=list)

    # All actions in order
    actions: list[Action] = field(default_factory=list)

    # Pot information
    total_pot: float = 0.0  # Final pot in BBs
    rake: float = 0.0

    # Hand result
    winners: list[Position] = field(default_factory=list)

    @property
    def flop(self) -> list[str]:
        """Get flop cards."""
        return self.board[:3] if len(self.board) >= 3 else []

    @property
    def turn(self) -> Optional[str]:
        """Get turn card."""
        return self.board[3] if len(self.board) >= 4 else None

    @property
    def river(self) -> Optional[str]:
        """Get river card."""
        return self.board[4] if len(self.board) >= 5 else None

    @property
    def num_players(self) -> int:
        """Number of players dealt into the hand."""
        return len(self.players)

    @property
    def bb(self) -> float:
        """Big blind amount in dollars."""
        return self.stakes[1]

    @property
    def sb(self) -> float:
        """Small blind amount in dollars."""
        return self.stakes[0]

    def get_player(self, position: Position) -> Optional[Player]:
        """Get player at a specific position."""
        for player in self.players:
            if player.position == position:
                return player
        return None

    def get_actions(self, street: Optional[Street] = None,
                    position: Optional[Position] = None) -> list[Action]:
        """Filter actions by street and/or position."""
        actions = self.actions
        if street is not None:
            actions = [a for a in actions if a.street == street]
        if position is not None:
            actions = [a for a in actions if a.position == position]
        return actions

    def went_to_showdown(self) -> bool:
        """Check if hand went to showdown."""
        # If we have river actions and multiple players, likely went to showdown
        river_actions = self.get_actions(street=Street.RIVER)
        if not river_actions:
            return False

        # Check if any player showed cards
        return any(p.showed_cards for p in self.players)

    def is_heads_up_postflop(self) -> bool:
        """Check if hand was heads-up on the flop."""
        if not self.flop:
            return False

        # Count players who didn't fold preflop
        preflop_actions = self.get_actions(street=Street.PREFLOP)
        folders = {a.position for a in preflop_actions if a.action_type == ActionType.FOLD}
        active_players = len(self.players) - len(folders)
        return active_players == 2

    def __repr__(self) -> str:
        board_str = " ".join(self.board) if self.board else "no board"
        return f"Hand({self.hand_id}, ${self.stakes[0]}/{self.stakes[1]}, {board_str})"
