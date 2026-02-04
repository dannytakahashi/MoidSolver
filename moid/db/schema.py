"""SQLite schema definitions for poker hand storage."""

import sqlite3
from pathlib import Path
from typing import Optional

# Schema version for migrations
SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Hands table: core hand metadata
CREATE TABLE IF NOT EXISTS hands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id TEXT UNIQUE NOT NULL,          -- Original hand ID from site
    timestamp DATETIME NOT NULL,
    sb REAL NOT NULL,                       -- Small blind in dollars
    bb REAL NOT NULL,                       -- Big blind in dollars
    table_name TEXT,
    board TEXT,                             -- Space-separated board cards
    total_pot REAL,                         -- Total pot in BBs
    rake REAL,                              -- Rake in dollars
    num_players INTEGER,
    went_to_showdown BOOLEAN DEFAULT 0,
    is_heads_up_postflop BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Players table: player data for each hand
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id INTEGER NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    position TEXT NOT NULL,                 -- UTG, UTG1, CO, BTN, SB, BB
    stack REAL NOT NULL,                    -- Starting stack in BBs
    hole_cards TEXT,                        -- Space-separated hole cards (if shown)
    result REAL DEFAULT 0,                  -- Net result in BBs
    is_hero BOOLEAN DEFAULT 0,
    showed_cards BOOLEAN DEFAULT 0,
    is_winner BOOLEAN DEFAULT 0
);

-- Actions table: all actions in all hands
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id INTEGER NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    position TEXT NOT NULL,
    street TEXT NOT NULL,                   -- PREFLOP, FLOP, TURN, RIVER
    action_type TEXT NOT NULL,              -- FOLD, CHECK, CALL, BET, RAISE, ALL_IN
    amount REAL DEFAULT 0,                  -- Amount in BBs
    is_all_in BOOLEAN DEFAULT 0,
    action_order INTEGER NOT NULL           -- Order within the hand
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_hands_timestamp ON hands(timestamp);
CREATE INDEX IF NOT EXISTS idx_hands_stakes ON hands(sb, bb);
CREATE INDEX IF NOT EXISTS idx_hands_showdown ON hands(went_to_showdown);
CREATE INDEX IF NOT EXISTS idx_hands_hu_postflop ON hands(is_heads_up_postflop);

CREATE INDEX IF NOT EXISTS idx_players_hand_id ON players(hand_id);
CREATE INDEX IF NOT EXISTS idx_players_position ON players(position);
CREATE INDEX IF NOT EXISTS idx_players_hero ON players(is_hero);

CREATE INDEX IF NOT EXISTS idx_actions_hand_id ON actions(hand_id);
CREATE INDEX IF NOT EXISTS idx_actions_position ON actions(position);
CREATE INDEX IF NOT EXISTS idx_actions_street ON actions(street);
CREATE INDEX IF NOT EXISTS idx_actions_type ON actions(action_type);
CREATE INDEX IF NOT EXISTS idx_actions_position_street ON actions(position, street);
CREATE INDEX IF NOT EXISTS idx_actions_hand_position ON actions(hand_id, position);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """
    Get a database connection with optimized settings.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLite connection object
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")

    # Performance optimizations
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
    conn.execute("PRAGMA temp_store = MEMORY")

    return conn


def create_database(db_path: str | Path, force: bool = False) -> sqlite3.Connection:
    """
    Create the database schema.

    Args:
        db_path: Path to SQLite database file
        force: If True, drop existing tables and recreate

    Returns:
        SQLite connection object
    """
    db_path = Path(db_path)

    # Create parent directory if needed
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection(db_path)

    if force:
        # Drop all existing tables
        conn.executescript("""
            DROP TABLE IF EXISTS actions;
            DROP TABLE IF EXISTS players;
            DROP TABLE IF EXISTS hands;
            DROP TABLE IF EXISTS schema_version;
        """)

    # Create schema
    conn.executescript(SCHEMA_SQL)

    # Set schema version
    conn.execute(
        "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
        (SCHEMA_VERSION,)
    )

    conn.commit()
    return conn


def get_schema_version(conn: sqlite3.Connection) -> Optional[int]:
    """Get the current schema version."""
    try:
        cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None
