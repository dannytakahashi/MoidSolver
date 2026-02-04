"""Parser for Ignition/Bovada hand history format."""

import re
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .models import Action, ActionType, Hand, Player, Position, Street


class IgnitionParser:
    """
    Parser for Ignition/Bovada hand history files.

    Ignition format characteristics:
    - Players are anonymous (positions like "UTG [ME]", "BTN")
    - Hole cards shown for hero and at showdown
    - All amounts in dollars
    - Hands separated by blank lines
    """

    # Regex patterns for parsing
    HAND_START = re.compile(
        r"Ignition Hand #(\d+)\s*:?\s*"
        r"(?:Zone Poker\s+)?(?:HOLDEM|Hold'?em)\s+"
        r"No Limit\s*-?\s*"
        r"\$?([\d.]+)/\$?([\d.]+)\s*-?\s*"
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
    )

    TABLE_INFO = re.compile(r"Table\s+(?:#?\d+\s+)?(?:Zone\s+)?(?:TBL#)?(\d+)")

    SEAT_INFO = re.compile(
        r"Seat\s+(\d+):\s*(\S+)(?:\s*\[ME\])?\s*"
        r"\(\s*\$?([\d,]+\.?\d*)\s*(?:in chips)?\)"
    )

    DEALT_CARDS = re.compile(
        r"(\S+)(?:\s*\[ME\])?\s*:\s*Card dealt to a]spot\s*\[([2-9TJQKA][cdhs])\s+([2-9TJQKA][cdhs])\]"
    )

    HOLE_CARDS = re.compile(
        r"(?:Dealt to\s+)?(\S+)(?:\s*\[ME\])?\s*\[([2-9TJQKA][cdhs])\s+([2-9TJQKA][cdhs])\]"
    )

    ACTION_PATTERN = re.compile(
        r"(\S+)(?:\s*\[ME\])?\s*:\s*"
        r"(Folds?|Checks?|Calls?|Bets?|Raises?|All-in(?:\(raise\))?)"
        r"(?:\s*\(?\$?([\d,]+\.?\d*)\)?)?"
        r"(?:\s*to\s*\$?([\d,]+\.?\d*))?"
    )

    BOARD_PATTERN = re.compile(
        r"\*\*\*\s*(FLOP|TURN|RIVER)\s*\*\*\*\s*\[([^\]]+)\]"
    )

    SHOWS_PATTERN = re.compile(
        r"(\S+)(?:\s*\[ME\])?\s*:\s*(?:Shows|Showdown)\s*\[([2-9TJQKA][cdhs])\s+([2-9TJQKA][cdhs])\]"
    )

    WINS_PATTERN = re.compile(
        r"(\S+)(?:\s*\[ME\])?\s*:\s*(?:Hand result(?:-Loss)?|Wins|wins)\s*\$?([\d,]+\.?\d*)"
    )

    TOTAL_POT = re.compile(r"Total Pot\s*\(\s*\$?([\d,]+\.?\d*)\s*\)")

    RAKE_PATTERN = re.compile(r"Rake\s*\(\s*\$?([\d,]+\.?\d*)\s*\)")

    # Position mapping from seat names to Position enum
    POSITION_MAP = {
        "UTG": Position.UTG,
        "UTG+1": Position.UTG1,
        "CO": Position.CO,
        "Dealer": Position.BTN,
        "Small Blind": Position.SB,
        "Big Blind": Position.BB,
    }

    def __init__(self):
        self._current_street = Street.PREFLOP

    def parse_file(self, filepath: str | Path) -> Iterator[Hand]:
        """Parse a hand history file and yield Hand objects."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Split into individual hands
        hand_texts = self._split_hands(content)

        for hand_text in hand_texts:
            try:
                hand = self.parse_hand(hand_text)
                if hand is not None:
                    yield hand
            except Exception as e:
                # Log parsing errors but continue
                print(f"Error parsing hand: {e}")
                continue

    def _split_hands(self, content: str) -> list[str]:
        """Split file content into individual hand texts."""
        # Hands are typically separated by multiple blank lines
        hands = re.split(r"\n\s*\n\s*\n", content)
        return [h.strip() for h in hands if h.strip()]

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

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            # Parse table info
            if "Table" in line:
                match = self.TABLE_INFO.search(line)
                if match:
                    hand.table_name = match.group(1)
                continue

            # Parse seat info
            if line.startswith("Seat"):
                self._parse_seat(line, hand, position_map)
                continue

            # Parse hole cards
            if "Card dealt" in line or "[" in line and "]" in line:
                self._parse_hole_cards(line, hand, position_map)
                continue

            # Parse board cards
            if "***" in line:
                self._parse_board(line, hand)
                continue

            # Parse actions
            if ":" in line:
                self._parse_action(line, hand, position_map)
                continue

            # Parse total pot
            if "Total Pot" in line:
                match = self.TOTAL_POT.search(line)
                if match:
                    hand.total_pot = self._parse_amount(match.group(1)) / hand.bb
                continue

            # Parse rake
            if "Rake" in line:
                match = self.RAKE_PATTERN.search(line)
                if match:
                    hand.rake = self._parse_amount(match.group(1))
                continue

        return hand

    def _parse_header(self, line: str) -> Optional[Hand]:
        """Parse hand header line."""
        match = self.HAND_START.search(line)
        if not match:
            return None

        hand_id = match.group(1)
        sb = float(match.group(2))
        bb = float(match.group(3))
        timestamp_str = match.group(4)

        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            timestamp = datetime.now()

        return Hand(
            hand_id=hand_id,
            timestamp=timestamp,
            stakes=(sb, bb),
        )

    def _parse_seat(self, line: str, hand: Hand, position_map: dict[str, Position]) -> None:
        """Parse seat information."""
        match = self.SEAT_INFO.search(line)
        if not match:
            return

        seat_num = int(match.group(1))
        player_name = match.group(2)
        stack_dollars = self._parse_amount(match.group(3))
        stack_bb = stack_dollars / hand.bb

        # Map seat name to position
        position = self._seat_name_to_position(player_name)
        if position is None:
            return

        position_map[player_name] = position

        is_hero = "[ME]" in line

        player = Player(
            position=position,
            stack=stack_bb,
            is_hero=is_hero,
        )
        hand.players.append(player)

    def _seat_name_to_position(self, name: str) -> Optional[Position]:
        """Convert Ignition seat name to Position."""
        name = name.strip()
        if name in self.POSITION_MAP:
            return self.POSITION_MAP[name]

        # Try parsing as position string
        try:
            return Position.from_string(name)
        except ValueError:
            return None

    def _parse_hole_cards(self, line: str, hand: Hand, position_map: dict[str, Position]) -> None:
        """Parse hole cards for a player."""
        match = self.HOLE_CARDS.search(line)
        if not match:
            return

        player_name = match.group(1)
        card1 = self._normalize_card(match.group(2))
        card2 = self._normalize_card(match.group(3))

        position = position_map.get(player_name)
        if position is None:
            return

        player = hand.get_player(position)
        if player:
            player.hole_cards = (card1, card2)

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
        # Check for showdown
        shows_match = self.SHOWS_PATTERN.search(line)
        if shows_match:
            player_name = shows_match.group(1)
            position = position_map.get(player_name)
            if position:
                player = hand.get_player(position)
                if player:
                    player.showed_cards = True
                    card1 = self._normalize_card(shows_match.group(2))
                    card2 = self._normalize_card(shows_match.group(3))
                    player.hole_cards = (card1, card2)
            return

        # Check for wins
        wins_match = self.WINS_PATTERN.search(line)
        if wins_match:
            player_name = wins_match.group(1)
            amount = self._parse_amount(wins_match.group(2))
            position = position_map.get(player_name)
            if position:
                if position not in hand.winners:
                    hand.winners.append(position)
                player = hand.get_player(position)
                if player:
                    player.result = amount / hand.bb
            return

        # Parse regular actions
        action_match = self.ACTION_PATTERN.search(line)
        if not action_match:
            return

        player_name = action_match.group(1)
        action_str = action_match.group(2)
        amount_str = action_match.group(3) or action_match.group(4) or "0"

        position = position_map.get(player_name)
        if position is None:
            return

        try:
            action_type = ActionType.from_string(action_str)
        except ValueError:
            return

        amount = self._parse_amount(amount_str) / hand.bb
        is_all_in = "All-in" in action_str

        action = Action(
            position=position,
            street=self._current_street,
            action_type=action_type,
            amount=amount,
            is_all_in=is_all_in,
        )
        hand.actions.append(action)

    def _parse_cards(self, cards_str: str) -> list[str]:
        """Parse a string of cards into a list."""
        # Match card patterns
        pattern = re.compile(r"([2-9TJQKA][cdhs])")
        matches = pattern.findall(cards_str)
        return [self._normalize_card(c) for c in matches]

    def _normalize_card(self, card: str) -> str:
        """Normalize card representation (e.g., 'Th' -> 'Th')."""
        if len(card) != 2:
            return card
        rank = card[0].upper()
        suit = card[1].lower()
        return rank + suit

    def _parse_amount(self, amount_str: str) -> float:
        """Parse dollar amount string to float."""
        if not amount_str:
            return 0.0
        # Remove commas and dollar signs
        cleaned = amount_str.replace(",", "").replace("$", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0


def parse_ignition_file(filepath: str | Path) -> list[Hand]:
    """Convenience function to parse an Ignition hand history file."""
    parser = IgnitionParser()
    return list(parser.parse_file(filepath))
