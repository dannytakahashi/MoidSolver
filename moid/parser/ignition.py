"""Parser for Ignition/Bovada hand history format."""

import re
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .models import Action, ActionType, Hand, Player, Position, Street


class IgnitionParser:
    """
    Parser for Ignition/Bovada hand history files.

    Bovada format characteristics:
    - Players identified by position (Small Blind, Big Blind, UTG, etc.)
    - [ME] suffix indicates hero
    - All amounts in dollars
    - Hands separated by blank lines
    """

    # Regex patterns for parsing
    HAND_START = re.compile(
        r"Bovada Hand #(\d+)"
        r".*?"
        r"HOLDEM\s+No Limit\s*-\s*"
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
    )

    SEAT_INFO = re.compile(
        r"Seat\s+(\d+):\s*(.+?)\s*\(\$?([\d,]+\.?\d*)\s*in chips\)"
    )

    HOLE_CARDS = re.compile(
        r"(.+?)\s*:\s*Card dealt to a spot\s*\[([2-9TJQKA][cdhs])\s+([2-9TJQKA][cdhs])\]"
    )

    BLIND_POST = re.compile(
        r"(.+?)\s*:\s*(?:Small Blind|Big blind|Posts chip)\s*\$?([\d,]+\.?\d*)"
    )

    ACTION_PATTERN = re.compile(
        r"(.+?)\s*:\s*(Folds?|Checks?|Calls?|Bets?|Raises?|All-in)"
        r"(?:\s*\$?([\d,]+\.?\d*))?"
        r"(?:\s*to\s*\$?([\d,]+\.?\d*))?"
    )

    BOARD_PATTERN = re.compile(
        r"\*\*\*\s*(FLOP|TURN|RIVER)\s*\*\*\*\s*\[([^\]]+)\]"
    )

    SHOWDOWN_PATTERN = re.compile(
        r"(.+?)\s*:\s*(?:Showdown|Shows|Does not show)\s*\[?([^\]]*)\]?"
    )

    HAND_RESULT = re.compile(
        r"(.+?)\s*:\s*Hand result\s*\$?([\d,]+\.?\d*)"
    )

    UNCALLED_BET = re.compile(
        r"(.+?)\s*:\s*Return uncalled portion of bet\s*\$?([\d,]+\.?\d*)"
    )

    TOTAL_POT = re.compile(r"Total Pot\s*\(\s*\$?([\d,]+\.?\d*)\s*\)")

    # Position mapping
    POSITION_NAMES = {
        "Dealer": Position.BTN,
        "Small Blind": Position.SB,
        "Big Blind": Position.BB,
        "UTG": Position.UTG,
        "UTG+1": Position.UTG1,
        "UTG+2": Position.CO,  # In 6-max, UTG+2 is CO
    }

    def __init__(self):
        self._current_street = Street.PREFLOP
        self._bb_amount = 0.05  # Default, will be detected

    def parse_file(self, filepath: str | Path) -> Iterator[Hand]:
        """Parse a hand history file and yield Hand objects."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Split into individual hands
        hand_texts = self._split_hands(content)

        for hand_text in hand_texts:
            if not hand_text.strip():
                continue
            try:
                hand = self.parse_hand(hand_text)
                if hand is not None:
                    yield hand
            except Exception as e:
                # Log parsing errors but continue
                continue

    def _split_hands(self, content: str) -> list[str]:
        """Split file content into individual hand texts."""
        # Split on "Bovada Hand #" but keep the delimiter
        parts = re.split(r'(?=Bovada Hand #)', content)
        return [p.strip() for p in parts if p.strip() and 'Bovada Hand #' in p]

    def parse_hand(self, text: str) -> Optional[Hand]:
        """Parse a single hand from text."""
        lines = text.strip().split("\n")
        if not lines:
            return None

        # Parse hand header
        hand = self._parse_header(lines[0])
        if hand is None:
            return None

        self._current_street = Street.PREFLOP
        position_map: dict[str, Position] = {}
        seat_to_position: dict[int, str] = {}

        # First pass: collect seat info to determine positions
        for line in lines:
            line = line.strip()
            if line.startswith("Seat"):
                match = self.SEAT_INFO.match(line)
                if match:
                    seat_num = int(match.group(1))
                    player_name = match.group(2).replace("[ME]", "").strip()
                    seat_to_position[seat_num] = player_name

        # Detect BB amount from blind posting
        for line in lines:
            if "Big blind $" in line or "Big blind $" in line.lower():
                match = re.search(r'\$?([\d.]+)', line.split("Big blind")[1])
                if match:
                    self._bb_amount = float(match.group(1))
                    break

        # Set stakes based on detected BB
        hand.stakes = (self._bb_amount / 2, self._bb_amount)

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            # Parse seat info
            if line.startswith("Seat") and "in chips" in line:
                self._parse_seat(line, hand, position_map)
                continue

            # Parse hole cards
            if "Card dealt to a spot" in line:
                self._parse_hole_cards(line, hand, position_map)
                continue

            # Parse board cards
            if line.startswith("***"):
                self._parse_board(line, hand)
                continue

            # Parse blind posts (skip, just for info)
            if "Small Blind $" in line or "Big blind $" in line or "Posts chip $" in line:
                continue

            # Parse actions
            if ":" in line and not line.startswith("Seat"):
                self._parse_action(line, hand, position_map)
                continue

            # Parse total pot
            if "Total Pot" in line:
                match = self.TOTAL_POT.search(line)
                if match:
                    hand.total_pot = self._parse_amount(match.group(1)) / self._bb_amount
                continue

        return hand

    def _parse_header(self, line: str) -> Optional[Hand]:
        """Parse hand header line."""
        match = self.HAND_START.search(line)
        if not match:
            return None

        hand_id = match.group(1)
        timestamp_str = match.group(2)

        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            timestamp = datetime.now()

        return Hand(
            hand_id=hand_id,
            timestamp=timestamp,
            stakes=(0.02, 0.05),  # Default, will be updated
        )

    def _parse_seat(self, line: str, hand: Hand, position_map: dict[str, Position]) -> None:
        """Parse seat information."""
        match = self.SEAT_INFO.match(line)
        if not match:
            return

        seat_num = int(match.group(1))
        player_name_raw = match.group(2)
        stack_dollars = self._parse_amount(match.group(3))

        # Check for [ME] marker
        is_hero = "[ME]" in player_name_raw
        player_name = player_name_raw.replace("[ME]", "").strip()

        # Map player name to position
        position = self._name_to_position(player_name)
        if position is None:
            return

        position_map[player_name] = position
        # Also map with [ME] suffix if hero
        if is_hero:
            position_map[player_name_raw.strip()] = position
            position_map[player_name + " [ME]"] = position
            position_map[player_name + "  [ME]"] = position  # Sometimes double space

        stack_bb = stack_dollars / self._bb_amount

        player = Player(
            position=position,
            stack=stack_bb,
            is_hero=is_hero,
        )
        hand.players.append(player)

    def _name_to_position(self, name: str) -> Optional[Position]:
        """Convert player name to Position."""
        name = name.strip()

        # Direct mapping
        if name in self.POSITION_NAMES:
            return self.POSITION_NAMES[name]

        # Try variations
        name_lower = name.lower()
        for key, pos in self.POSITION_NAMES.items():
            if key.lower() == name_lower:
                return pos

        # Handle CO specifically
        if name in ("CO", "Cutoff"):
            return Position.CO

        return None

    def _parse_hole_cards(self, line: str, hand: Hand, position_map: dict[str, Position]) -> None:
        """Parse hole cards for a player."""
        match = self.HOLE_CARDS.match(line)
        if not match:
            return

        player_name_raw = match.group(1).strip()
        card1 = self._normalize_card(match.group(2))
        card2 = self._normalize_card(match.group(3))

        # Try to find position
        position = self._lookup_position(player_name_raw, position_map)
        if position is None:
            return

        player = hand.get_player(position)
        if player:
            player.hole_cards = (card1, card2)

    def _lookup_position(self, name: str, position_map: dict[str, Position]) -> Optional[Position]:
        """Look up position from player name, handling variations."""
        name = name.strip()

        # Direct lookup
        if name in position_map:
            return position_map[name]

        # Try without [ME]
        clean_name = name.replace("[ME]", "").strip()
        if clean_name in position_map:
            return position_map[clean_name]

        # Try base name to position mapping
        return self._name_to_position(clean_name)

    def _parse_board(self, line: str, hand: Hand) -> None:
        """Parse board cards and update current street."""
        match = self.BOARD_PATTERN.search(line)
        if not match:
            return

        street_name = match.group(1)
        cards_str = match.group(2)

        # Update current street
        if street_name == "FLOP":
            self._current_street = Street.FLOP
        elif street_name == "TURN":
            self._current_street = Street.TURN
        elif street_name == "RIVER":
            self._current_street = Street.RIVER

        # Parse and add board cards
        cards = self._parse_cards(cards_str)
        for card in cards:
            if card not in hand.board:
                hand.board.append(card)

    def _parse_action(self, line: str, hand: Hand, position_map: dict[str, Position]) -> None:
        """Parse a player action."""
        # Skip non-action lines
        skip_patterns = [
            "Card dealt", "Set dealer", "Showdown", "Hand result",
            "Return uncalled", "Does not show", "Table leave",
            "Seat stand", "Table deposit", "Seat re-join",
            "Small Blind $", "Big blind $", "Posts chip $"
        ]
        for pattern in skip_patterns:
            if pattern in line:
                # Handle hand result
                if "Hand result" in line:
                    match = self.HAND_RESULT.match(line)
                    if match:
                        player_name = match.group(1).strip()
                        amount = self._parse_amount(match.group(2))
                        position = self._lookup_position(player_name, position_map)
                        if position:
                            if position not in hand.winners:
                                hand.winners.append(position)
                            player = hand.get_player(position)
                            if player:
                                player.result = amount / self._bb_amount
                # Handle showdown
                if "Showdown" in line or "Does not show" in line:
                    match = self.SHOWDOWN_PATTERN.match(line)
                    if match:
                        player_name = match.group(1).strip()
                        position = self._lookup_position(player_name, position_map)
                        if position:
                            player = hand.get_player(position)
                            if player:
                                player.showed_cards = "Showdown" in line
                return

        # Parse regular actions
        match = self.ACTION_PATTERN.match(line)
        if not match:
            return

        player_name = match.group(1).strip()
        action_str = match.group(2)
        amount_str = match.group(3) or match.group(4) or "0"

        position = self._lookup_position(player_name, position_map)
        if position is None:
            return

        try:
            action_type = self._parse_action_type(action_str)
        except ValueError:
            return

        amount = self._parse_amount(amount_str) / self._bb_amount
        is_all_in = "All-in" in action_str or "All-in" in line

        action = Action(
            position=position,
            street=self._current_street,
            action_type=action_type,
            amount=amount,
            is_all_in=is_all_in,
        )
        hand.actions.append(action)

    def _parse_action_type(self, action_str: str) -> ActionType:
        """Parse action type from string."""
        action_str = action_str.lower().strip()

        if "fold" in action_str:
            return ActionType.FOLD
        if "check" in action_str:
            return ActionType.CHECK
        if "call" in action_str:
            return ActionType.CALL
        if "bet" in action_str:
            return ActionType.BET
        if "raise" in action_str:
            return ActionType.RAISE
        if "all-in" in action_str or "allin" in action_str:
            return ActionType.ALL_IN

        raise ValueError(f"Unknown action: {action_str}")

    def _parse_cards(self, cards_str: str) -> list[str]:
        """Parse a string of cards into a list."""
        pattern = re.compile(r"([2-9TJQKA][cdhs])")
        matches = pattern.findall(cards_str)
        return [self._normalize_card(c) for c in matches]

    def _normalize_card(self, card: str) -> str:
        """Normalize card representation."""
        if len(card) != 2:
            return card
        rank = card[0].upper()
        suit = card[1].lower()
        return rank + suit

    def _parse_amount(self, amount_str: str) -> float:
        """Parse dollar amount string to float."""
        if not amount_str:
            return 0.0
        cleaned = amount_str.replace(",", "").replace("$", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0


def parse_ignition_file(filepath: str | Path) -> list[Hand]:
    """Convenience function to parse an Ignition/Bovada hand history file."""
    parser = IgnitionParser()
    return list(parser.parse_file(filepath))
