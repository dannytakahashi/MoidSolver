"""Core poker statistics calculations."""

import sqlite3
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlayerStats:
    """
    Standard poker HUD statistics.

    All percentages are stored as values 0-100.
    """
    # Sample size
    hands: int = 0

    # Preflop stats
    vpip: float = 0.0         # Voluntarily Put $ In Pot %
    pfr: float = 0.0          # Preflop Raise %
    three_bet: float = 0.0    # 3-bet %
    fold_to_3bet: float = 0.0 # Fold to 3-bet %
    four_bet: float = 0.0     # 4-bet %
    cold_call: float = 0.0    # Cold call %

    # Postflop stats
    cbet: float = 0.0         # Continuation bet %
    fold_to_cbet: float = 0.0 # Fold to c-bet %
    cbet_turn: float = 0.0    # Turn c-bet %
    cbet_river: float = 0.0   # River c-bet %

    # Aggression
    af: float = 0.0           # Aggression Factor (bets+raises) / calls
    afq: float = 0.0          # Aggression Frequency %

    # Showdown stats
    wtsd: float = 0.0         # Went to Showdown %
    wsd: float = 0.0          # Won $ at Showdown %

    # Position-specific (optional)
    vpip_by_position: dict[str, float] = field(default_factory=dict)
    pfr_by_position: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"PlayerStats(hands={self.hands}, "
            f"VPIP={self.vpip:.1f}, PFR={self.pfr:.1f}, "
            f"3bet={self.three_bet:.1f}, AF={self.af:.1f})"
        )


def compute_stats(
    conn: sqlite3.Connection,
    position: Optional[str] = None,
    min_stack: Optional[float] = None,
    max_stack: Optional[float] = None,
) -> PlayerStats:
    """
    Compute player statistics from database.

    Args:
        conn: Database connection
        position: Optional position filter (e.g., "BTN")
        min_stack: Minimum stack size in BBs
        max_stack: Maximum stack size in BBs

    Returns:
        PlayerStats object with computed statistics
    """
    # If no position specified, compute average across all positions
    if not position:
        return _compute_average_stats(conn, min_stack, max_stack)

    stats = PlayerStats()

    # Build common filters
    filters = ["p.position = ?"]
    params = [position]

    if min_stack is not None:
        filters.append("p.stack >= ?")
        params.append(min_stack)
    if max_stack is not None:
        filters.append("p.stack <= ?")
        params.append(max_stack)

    where_clause = " AND ".join(filters)

    # Count total hands
    stats.hands = _count_hands(conn, where_clause, params)
    if stats.hands == 0:
        return stats

    # Compute VPIP and PFR
    vpip_pfr = _compute_vpip_pfr(conn, where_clause, params)
    stats.vpip = vpip_pfr["vpip"]
    stats.pfr = vpip_pfr["pfr"]

    # Compute 3-bet stats
    three_bet_stats = _compute_3bet_stats(conn, where_clause, params)
    stats.three_bet = three_bet_stats["three_bet"]
    stats.fold_to_3bet = three_bet_stats["fold_to_3bet"]

    # Compute c-bet stats
    cbet_stats = _compute_cbet_stats(conn, where_clause, params)
    stats.cbet = cbet_stats["cbet"]
    stats.fold_to_cbet = cbet_stats["fold_to_cbet"]

    # Compute aggression
    aggression = _compute_aggression(conn, where_clause, params)
    stats.af = aggression["af"]
    stats.afq = aggression["afq"]

    # Compute showdown stats
    showdown = _compute_showdown_stats(conn, where_clause, params)
    stats.wtsd = showdown["wtsd"]
    stats.wsd = showdown["wsd"]

    return stats


def _compute_average_stats(
    conn: sqlite3.Connection,
    min_stack: Optional[float] = None,
    max_stack: Optional[float] = None,
) -> PlayerStats:
    """
    Compute average stats across all positions.

    This gives a meaningful "overall" population stat by averaging
    the per-position stats rather than computing a nonsensical aggregate.
    """
    positions = ["UTG", "UTG1", "CO", "BTN", "SB", "BB"]
    position_stats: dict[str, PlayerStats] = {}

    for pos in positions:
        pos_stats = compute_stats(conn, position=pos, min_stack=min_stack, max_stack=max_stack)
        if pos_stats.hands > 0:
            position_stats[pos] = pos_stats

    if not position_stats:
        return PlayerStats()

    stats_list = list(position_stats.values())
    n = len(stats_list)

    # Average the stats across positions
    avg = PlayerStats()
    avg.hands = sum(s.hands for s in stats_list)

    avg.vpip = sum(s.vpip for s in stats_list) / n
    avg.pfr = sum(s.pfr for s in stats_list) / n
    avg.three_bet = sum(s.three_bet for s in stats_list) / n
    avg.fold_to_3bet = sum(s.fold_to_3bet for s in stats_list) / n
    avg.cbet = sum(s.cbet for s in stats_list) / n
    avg.fold_to_cbet = sum(s.fold_to_cbet for s in stats_list) / n
    avg.af = sum(s.af for s in stats_list) / n
    avg.afq = sum(s.afq for s in stats_list) / n
    avg.wtsd = sum(s.wtsd for s in stats_list) / n
    avg.wsd = sum(s.wsd for s in stats_list) / n

    # Store per-position breakdowns
    avg.vpip_by_position = {pos: s.vpip for pos, s in position_stats.items()}
    avg.pfr_by_position = {pos: s.pfr for pos, s in position_stats.items()}

    return avg


def _count_hands(conn: sqlite3.Connection, where_clause: str, params: list) -> int:
    """Count distinct hands matching filters."""
    query = f"""
        SELECT COUNT(DISTINCT h.id)
        FROM hands h
        JOIN players p ON h.id = p.hand_id
        WHERE {where_clause}
    """
    cursor = conn.execute(query, params)
    return cursor.fetchone()[0]


def _compute_vpip_pfr(
    conn: sqlite3.Connection, where_clause: str, params: list
) -> dict[str, float]:
    """
    Compute VPIP and PFR.

    VPIP: Player voluntarily put money in preflop (call or raise, excluding blinds)
    PFR: Player raised preflop
    """
    query = f"""
        SELECT
            COUNT(DISTINCT h.id) as total_hands,
            COUNT(DISTINCT CASE
                WHEN a.street = 'PREFLOP'
                AND a.action_type IN ('CALL', 'RAISE', 'BET', 'ALL_IN')
                THEN h.id
            END) as vpip_hands,
            COUNT(DISTINCT CASE
                WHEN a.street = 'PREFLOP'
                AND a.action_type IN ('RAISE', 'BET', 'ALL_IN')
                THEN h.id
            END) as pfr_hands
        FROM hands h
        JOIN players p ON h.id = p.hand_id
        LEFT JOIN actions a ON h.id = a.hand_id AND a.position = p.position
        WHERE {where_clause}
    """
    cursor = conn.execute(query, params)
    row = cursor.fetchone()

    total = row[0] or 1
    vpip = (row[1] or 0) / total * 100
    pfr = (row[2] or 0) / total * 100

    return {"vpip": vpip, "pfr": pfr}


def _compute_3bet_stats(
    conn: sqlite3.Connection, where_clause: str, params: list
) -> dict[str, float]:
    """
    Compute 3-bet percentage and fold to 3-bet.

    3-bet: Raised after facing a raise preflop
    """
    # 3-bet opportunities: hands where player faced a raise preflop
    # 3-bet: hands where player re-raised that raise

    query = f"""
        WITH player_hands AS (
            SELECT DISTINCT h.id as hand_id, p.position
            FROM hands h
            JOIN players p ON h.id = p.hand_id
            WHERE {where_clause}
        ),
        facing_raise AS (
            -- Hands where there was a preflop raise before player acted
            SELECT ph.hand_id, ph.position,
                   MAX(CASE WHEN a2.action_type = 'RAISE' THEN 1 ELSE 0 END) as reraised
            FROM player_hands ph
            JOIN actions a1 ON ph.hand_id = a1.hand_id
                AND a1.street = 'PREFLOP'
                AND a1.action_type = 'RAISE'
                AND a1.position != ph.position
            LEFT JOIN actions a2 ON ph.hand_id = a2.hand_id
                AND a2.position = ph.position
                AND a2.street = 'PREFLOP'
                AND a2.action_type = 'RAISE'
            GROUP BY ph.hand_id, ph.position
        )
        SELECT
            COUNT(*) as opportunities,
            SUM(reraised) as three_bets
        FROM facing_raise
    """
    cursor = conn.execute(query, params)
    row = cursor.fetchone()

    opportunities = row[0] or 1
    three_bet = (row[1] or 0) / opportunities * 100 if opportunities > 0 else 0

    # Fold to 3-bet: when player raised and then folded to a re-raise
    fold_query = f"""
        WITH player_raises AS (
            SELECT DISTINCT h.id as hand_id, p.position
            FROM hands h
            JOIN players p ON h.id = p.hand_id
            JOIN actions a ON h.id = a.hand_id
                AND a.position = p.position
                AND a.street = 'PREFLOP'
                AND a.action_type = 'RAISE'
            WHERE {where_clause}
        ),
        faced_3bet AS (
            SELECT pr.hand_id, pr.position,
                   MAX(CASE WHEN a2.action_type = 'FOLD' THEN 1 ELSE 0 END) as folded
            FROM player_raises pr
            JOIN actions a1 ON pr.hand_id = a1.hand_id
                AND a1.street = 'PREFLOP'
                AND a1.action_type = 'RAISE'
                AND a1.position != pr.position
            LEFT JOIN actions a2 ON pr.hand_id = a2.hand_id
                AND a2.position = pr.position
                AND a2.street = 'PREFLOP'
            GROUP BY pr.hand_id, pr.position
        )
        SELECT
            COUNT(*) as opportunities,
            SUM(folded) as folds
        FROM faced_3bet
    """
    cursor = conn.execute(fold_query, params)
    row = cursor.fetchone()

    fold_opps = row[0] or 1
    fold_to_3bet = (row[1] or 0) / fold_opps * 100 if fold_opps > 0 else 0

    return {"three_bet": three_bet, "fold_to_3bet": fold_to_3bet}


def _compute_cbet_stats(
    conn: sqlite3.Connection, where_clause: str, params: list
) -> dict[str, float]:
    """
    Compute c-bet and fold to c-bet percentages.

    C-bet: Bet the flop after being the preflop aggressor
    """
    # C-bet opportunities: player was PFR and saw the flop
    query = f"""
        WITH pfr_hands AS (
            -- Hands where player was the last preflop raiser
            SELECT h.id as hand_id, p.position
            FROM hands h
            JOIN players p ON h.id = p.hand_id
            JOIN actions a ON h.id = a.hand_id
                AND a.position = p.position
                AND a.street = 'PREFLOP'
                AND a.action_type IN ('RAISE', 'BET')
            WHERE {where_clause}
            AND h.board IS NOT NULL
            AND h.board != ''
        ),
        cbet_opps AS (
            SELECT ph.hand_id, ph.position,
                   MAX(CASE
                       WHEN a.action_type IN ('BET', 'RAISE', 'ALL_IN') THEN 1
                       ELSE 0
                   END) as cbetted
            FROM pfr_hands ph
            LEFT JOIN actions a ON ph.hand_id = a.hand_id
                AND a.position = ph.position
                AND a.street = 'FLOP'
            GROUP BY ph.hand_id, ph.position
        )
        SELECT
            COUNT(*) as opportunities,
            SUM(cbetted) as cbets
        FROM cbet_opps
    """
    cursor = conn.execute(query, params)
    row = cursor.fetchone()

    cbet_opps = row[0] or 1
    cbet = (row[1] or 0) / cbet_opps * 100 if cbet_opps > 0 else 0

    # Fold to c-bet: folded to flop bet after calling preflop
    fold_query = f"""
        WITH called_pf AS (
            SELECT h.id as hand_id, p.position
            FROM hands h
            JOIN players p ON h.id = p.hand_id
            JOIN actions a ON h.id = a.hand_id
                AND a.position = p.position
                AND a.street = 'PREFLOP'
                AND a.action_type = 'CALL'
            WHERE {where_clause}
            AND h.board IS NOT NULL
        ),
        faced_cbet AS (
            SELECT cp.hand_id, cp.position,
                   MAX(CASE WHEN a2.action_type = 'FOLD' THEN 1 ELSE 0 END) as folded
            FROM called_pf cp
            JOIN actions a1 ON cp.hand_id = a1.hand_id
                AND a1.street = 'FLOP'
                AND a1.action_type IN ('BET', 'RAISE')
                AND a1.position != cp.position
            LEFT JOIN actions a2 ON cp.hand_id = a2.hand_id
                AND a2.position = cp.position
                AND a2.street = 'FLOP'
            GROUP BY cp.hand_id, cp.position
        )
        SELECT
            COUNT(*) as opportunities,
            SUM(folded) as folds
        FROM faced_cbet
    """
    cursor = conn.execute(fold_query, params)
    row = cursor.fetchone()

    fold_opps = row[0] or 1
    fold_to_cbet = (row[1] or 0) / fold_opps * 100 if fold_opps > 0 else 0

    return {"cbet": cbet, "fold_to_cbet": fold_to_cbet}


def _compute_aggression(
    conn: sqlite3.Connection, where_clause: str, params: list
) -> dict[str, float]:
    """
    Compute aggression factor and aggression frequency.

    AF = (bets + raises) / calls
    AFq = (bets + raises) / (bets + raises + calls + folds) * 100
    """
    query = f"""
        SELECT
            SUM(CASE WHEN a.action_type IN ('BET', 'RAISE', 'ALL_IN') THEN 1 ELSE 0 END) as aggressive,
            SUM(CASE WHEN a.action_type = 'CALL' THEN 1 ELSE 0 END) as calls,
            SUM(CASE WHEN a.action_type = 'FOLD' THEN 1 ELSE 0 END) as folds,
            SUM(CASE WHEN a.action_type = 'CHECK' THEN 1 ELSE 0 END) as checks
        FROM hands h
        JOIN players p ON h.id = p.hand_id
        JOIN actions a ON h.id = a.hand_id AND a.position = p.position
        WHERE {where_clause}
        AND a.street != 'PREFLOP'
    """
    cursor = conn.execute(query, params)
    row = cursor.fetchone()

    aggressive = row[0] or 0
    calls = row[1] or 1  # Avoid division by zero
    folds = row[2] or 0
    checks = row[3] or 0

    af = aggressive / calls if calls > 0 else 0
    total_actions = aggressive + calls + folds + checks
    afq = aggressive / total_actions * 100 if total_actions > 0 else 0

    return {"af": af, "afq": afq}


def _compute_showdown_stats(
    conn: sqlite3.Connection, where_clause: str, params: list
) -> dict[str, float]:
    """
    Compute showdown statistics.

    WTSD: Went to showdown when saw flop
    W$SD: Won money at showdown

    Uses hands.went_to_showdown flag instead of players.showed_cards
    to avoid selection bias (players can muck losing hands without showing).
    """
    query = f"""
        WITH saw_flop AS (
            SELECT DISTINCT h.id as hand_id, p.position,
                   h.went_to_showdown, p.is_winner
            FROM hands h
            JOIN players p ON h.id = p.hand_id
            WHERE {where_clause}
            AND h.board IS NOT NULL
            AND h.board != ''
            AND NOT EXISTS (
                SELECT 1 FROM actions a
                WHERE a.hand_id = h.id
                AND a.position = p.position
                AND a.street = 'PREFLOP'
                AND a.action_type = 'FOLD'
            )
        )
        SELECT
            COUNT(*) as saw_flop,
            SUM(CASE WHEN went_to_showdown = 1 THEN 1 ELSE 0 END) as showdowns,
            SUM(CASE WHEN went_to_showdown = 1 AND is_winner = 1 THEN 1 ELSE 0 END) as won_sd
        FROM saw_flop
    """
    cursor = conn.execute(query, params)
    row = cursor.fetchone()

    saw_flop = row[0] or 1
    showdowns = row[1] or 0
    won_sd = row[2] or 0

    wtsd = showdowns / saw_flop * 100 if saw_flop > 0 else 0
    wsd = won_sd / showdowns * 100 if showdowns > 0 else 0

    return {"wtsd": wtsd, "wsd": wsd}


def _compute_stat_by_position(
    conn: sqlite3.Connection,
    stat: str,
    min_stack: Optional[float] = None,
    max_stack: Optional[float] = None,
) -> dict[str, float]:
    """Compute a stat broken down by position."""
    positions = ["UTG", "UTG1", "CO", "BTN", "SB", "BB"]
    result = {}

    for pos in positions:
        filters = [f"p.position = '{pos}'"]
        params = []

        if min_stack is not None:
            filters.append("p.stack >= ?")
            params.append(min_stack)
        if max_stack is not None:
            filters.append("p.stack <= ?")
            params.append(max_stack)

        where_clause = " AND ".join(filters)

        if stat == "vpip":
            stats = _compute_vpip_pfr(conn, where_clause, params)
            result[pos] = stats["vpip"]
        elif stat == "pfr":
            stats = _compute_vpip_pfr(conn, where_clause, params)
            result[pos] = stats["pfr"]

    return result
