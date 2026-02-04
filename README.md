# MoidSolver

A microstakes NLHE solver for 6-max no-limit hold'em, focused on identifying optimal strategies against microstakes populations using Ignition/Bovada hand history data.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Tools](#cli-tools)
  - [Importing Hand Histories](#importing-hand-histories)
  - [Analyzing Population](#analyzing-population)
  - [Solving Spots](#solving-spots)
- [Python API](#python-api)
  - [Parsing Hands](#parsing-hands)
  - [Database Operations](#database-operations)
  - [Computing Statistics](#computing-statistics)
  - [Player Classification](#player-classification)
  - [Equity Calculations](#equity-calculations)
  - [Running the Solver](#running-the-solver)
  - [Visualizing Ranges](#visualizing-ranges)
- [Example Workflows](#example-workflows)
- [Architecture](#architecture)

## Installation

### Requirements

- Python 3.10 or higher
- pip

### Install from source

```bash
# Clone the repository
cd MoidSolver

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Dependencies

Core dependencies are installed automatically:
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scipy` - Scientific computing
- `treys` - Fast poker hand evaluation
- `matplotlib` - Plotting (optional, for graphical range display)
- `rich` - Terminal formatting

## Quick Start

```bash
# 1. Import your hand histories
python scripts/import_hands.py /path/to/hand/histories/ -d hands.db

# 2. Analyze the population
python scripts/analyze_population.py -d hands.db --exploits

# 3. Solve a specific spot
python scripts/solve_spot.py -b "AsKh7d" -p 6.5 -s 100 --mode adaptive -d hands.db
```

## CLI Tools

### Importing Hand Histories

Import Ignition/Bovada hand history files into an SQLite database.

```bash
python scripts/import_hands.py [FILES...] [OPTIONS]
```

**Arguments:**
- `FILES` - Hand history files or directories to import

**Options:**
- `-d, --database PATH` - Database file path (default: `hands.db`)
- `--force` - Recreate database, deleting existing data
- `-v, --verbose` - Show detailed output

**Examples:**

```bash
# Import a single file
python scripts/import_hands.py session_2024_01_15.txt

# Import all files in a directory
python scripts/import_hands.py ./hand_histories/

# Import to a specific database
python scripts/import_hands.py ./hands/ -d my_database.db

# Recreate database from scratch
python scripts/import_hands.py ./hands/ -d hands.db --force
```

### Analyzing Population

Compute and display population statistics.

```bash
python scripts/analyze_population.py [OPTIONS]
```

**Options:**
- `-d, --database PATH` - Database file path (default: `hands.db`)
- `-p, --position POS` - Filter by position (UTG, UTG1, CO, BTN, SB, BB)
- `--min-stack BBs` - Minimum stack size filter
- `--max-stack BBs` - Maximum stack size filter
- `--exploits` - Show exploitation recommendations
- `--by-position` - Show breakdown by position
- `--by-stack` - Show breakdown by stack depth

**Examples:**

```bash
# Basic population analysis
python scripts/analyze_population.py -d hands.db

# Show exploitation recommendations
python scripts/analyze_population.py -d hands.db --exploits

# Analyze BTN play specifically
python scripts/analyze_population.py -d hands.db -p BTN

# Analyze short-stack play
python scripts/analyze_population.py -d hands.db --max-stack 50

# Full breakdown
python scripts/analyze_population.py -d hands.db --by-position --by-stack --exploits
```

**Sample Output:**

```
                    Overall Population Statistics
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Statistic      ┃   Value ┃ Notes                 ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ Hands          │  12,450 │ sample size           │
│                │         │                       │
│ Preflop        │         │                       │
│ VPIP           │  38.2%  │ very loose            │
│ PFR            │  11.5%  │ passive               │
│ 3-Bet          │   4.2%  │ low (exploitable)     │
│ Fold to 3-Bet  │  62.1%  │                       │
│                │         │                       │
│ Postflop       │         │                       │
│ C-Bet          │  52.3%  │                       │
│ Fold to C-Bet  │  58.7%  │ overfolds (exploit!)  │
...
```

### Solving Spots

Solve specific poker spots using CFR or population-based methods.

```bash
python scripts/solve_spot.py [OPTIONS]
```

**Options:**
- `-b, --board CARDS` - Board cards (required), e.g., "AsKh7d"
- `-p, --pot BBs` - Starting pot in big blinds (default: 6.5)
- `-s, --stack BBs` - Effective stack in big blinds (default: 100)
- `-i, --iterations N` - CFR iterations (default: 1000)
- `--bet-sizes SIZES` - Bet sizes as pot fractions (default: "0.33,0.5,0.75,1.0")
- `-d, --database PATH` - Use population stats for adaptive solving
- `--mode MODE` - Solving mode: "nash" or "adaptive" (default: nash)
- `-o, --output PATH` - Save strategy to file
- `-v, --verbose` - Verbose output

**Examples:**

```bash
# Solve for Nash equilibrium on a flop
python scripts/solve_spot.py -b "AsKh7d" -i 5000

# Solve adaptively using population data
python scripts/solve_spot.py -b "AsKh7d" --mode adaptive -d hands.db

# Solve a turn spot with specific stack/pot
python scripts/solve_spot.py -b "AsKh7d2c" -p 15 -s 85

# Use custom bet sizes
python scripts/solve_spot.py -b "AsKh7d" --bet-sizes "0.5,1.0,1.5"

# Save strategy for later analysis
python scripts/solve_spot.py -b "AsKh7d" -o strategy.json
```

## Python API

### Parsing Hands

```python
from moid.parser import IgnitionParser, Hand, Action, Position, Street

# Parse a hand history file
parser = IgnitionParser()
for hand in parser.parse_file("session.txt"):
    print(f"Hand {hand.hand_id}: {hand.board}")

    # Access players
    for player in hand.players:
        print(f"  {player.position}: {player.stack}bb", end="")
        if player.hole_cards:
            print(f" [{player.hole_cards[0]}{player.hole_cards[1]}]")
        else:
            print()

    # Access actions
    for action in hand.get_actions(street=Street.PREFLOP):
        print(f"  {action.position} {action.action_type.name} {action.amount}")
```

### Database Operations

```python
from moid.db import create_database, get_connection, HandRepository

# Create a new database
conn = create_database("hands.db")
repo = HandRepository(conn)

# Import hands
from moid.parser import IgnitionParser
parser = IgnitionParser()
hands = parser.parse_file("session.txt")
count = repo.insert_hands(hands)
conn.commit()
print(f"Imported {count} hands")

# Query hands
print(f"Total hands: {repo.count_hands()}")
print(f"Preflop raises: {repo.count_actions(action_type='RAISE', street='PREFLOP')}")

# Retrieve a specific hand
hand = repo.get_hand("4561234567")
if hand:
    print(f"Board: {hand.board}")
```

### Computing Statistics

```python
from moid.db import get_connection
from moid.analysis import compute_stats, PopulationAnalyzer

conn = get_connection("hands.db")

# Compute overall stats
stats = compute_stats(conn)
print(f"VPIP: {stats.vpip:.1f}%")
print(f"PFR: {stats.pfr:.1f}%")
print(f"3-Bet: {stats.three_bet:.1f}%")
print(f"C-Bet: {stats.cbet:.1f}%")
print(f"AF: {stats.af:.2f}")

# Filter by position
btn_stats = compute_stats(conn, position="BTN")
print(f"BTN VPIP: {btn_stats.vpip:.1f}%")

# Filter by stack depth
short_stats = compute_stats(conn, max_stack=50)
deep_stats = compute_stats(conn, min_stack=100)

# Full population analysis
analyzer = PopulationAnalyzer(conn)
pop_stats = analyzer.analyze()

# Get exploitation recommendations
exploits = analyzer.get_exploits()
for exploit in exploits:
    print(f"- {exploit}")
```

### Player Classification

```python
from moid.classifier import ArchetypeClassifier, BayesianClassifier, PlayerArchetype
from moid.analysis import compute_stats

# Rule-based classification
classifier = ArchetypeClassifier(min_hands=30)
stats = compute_stats(conn, position="BTN")
archetype = classifier.classify(stats)

print(f"Archetype: {archetype.name}")
print(f"Description: {archetype.description()}")

# Get exploitation strategies
exploits = classifier.get_exploits(archetype)
for exploit in exploits:
    print(f"  - {exploit}")

# Real-time Bayesian classification (during a session)
bayesian = BayesianClassifier()

# Observe actions as they happen
bayesian.observe_preflop(vpip=True, pfr=False)   # Called preflop
bayesian.observe_preflop(vpip=True, pfr=False)   # Called again
bayesian.observe_preflop(vpip=True, pfr=True)    # Raised
bayesian.observe_postflop(aggressive=False)       # Check/called postflop

# Get current classification
archetype = bayesian.get_classification()
probabilities = bayesian.get_probabilities()

print(f"Most likely: {archetype.name}")
for arch, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
    print(f"  {arch.name}: {prob:.1%}")
```

### Equity Calculations

```python
from moid.game.cards import Card, Hand
from moid.game.equity import EquityCalculator, calculate_equity

# Create hands and board
hero = Hand.from_string("AsKh")
villain = Hand.from_string("QsQd")
board = [Card.from_string(c) for c in ["Ks", "7d", "2c"]]

# Hand vs hand equity
calc = EquityCalculator()
hero_eq, villain_eq, tie = calc.hand_vs_hand(hero, villain, board)
print(f"Hero: {hero_eq:.1%}, Villain: {villain_eq:.1%}, Tie: {tie:.1%}")

# Equity vs random hand
equity = calculate_equity(hero, board, num_opponents=1)
print(f"Hero vs random: {equity:.1%}")

# Equity multiway
equity_3way = calculate_equity(hero, board, num_opponents=2)
print(f"Hero vs 2 randoms: {equity_3way:.1%}")
```

### Running the Solver

```python
from moid.game.cards import Card
from moid.game.tree import GameTree
from moid.solver import CFRSolver, CFRConfig, AdaptiveSolver
from moid.analysis.stats import PlayerStats

board = [Card.from_string(c) for c in ["Ks", "7d", "2c"]]

# Nash equilibrium via CFR
tree = GameTree(starting_pot=6.5, effective_stack=100, starting_street=1)
tree.build(bet_sizes=[0.33, 0.5, 0.75, 1.0])

config = CFRConfig(num_iterations=10000, num_buckets=8)
solver = CFRSolver(tree, board, config)

def progress(iteration, exploitability):
    if iteration % 1000 == 0:
        print(f"Iteration {iteration}, exploitability: {exploitability:.4f}")

strategy = solver.solve(callback=progress)
print(f"Solved {strategy.num_info_sets()} information sets")

# Population-based solving
population_stats = PlayerStats(
    hands=10000,
    vpip=38.0,
    pfr=12.0,
    fold_to_cbet=58.0,
    cbet=52.0,
    af=1.4,
)

exploit_solver = AdaptiveSolver(
    stats=population_stats,
    board=board,
    starting_pot=6.5,
    effective_stack=100,
)

exploit_strategy = exploit_solver.solve()

# Get specific recommendations
action, explanation = exploit_solver.get_recommended_action(
    hand_strength="medium",  # "strong", "medium", "weak", "air"
    facing_bet=False,
)
print(f"Recommendation: {action}")
print(f"Reason: {explanation}")
```

### Visualizing Ranges

```python
from moid.viz import RangeDisplay, display_range

# Display a simple range
display_range(["AA", "KK", "QQ", "AKs", "AKo"], title="Premium Hands")

# Display with frequencies
hands = ["AA", "KK", "QQ", "JJ", "TT", "AKs", "AQs", "AKo"]
freqs = [1.0, 1.0, 1.0, 0.8, 0.5, 1.0, 0.7, 0.6]
display_range(hands, freqs, title="Opening Range")

# Full control with RangeDisplay
display = RangeDisplay()

# Load from range string
display.load_from_range_string("AA,KK,QQ,JJ,TT+,AKs,AQs:0.5")

# Set action breakdowns
display.set_action_breakdown("AA", {"raise": 1.0})
display.set_action_breakdown("AKs", {"raise": 0.7, "call": 0.3})

# Display in terminal
display.display_terminal(title="3-Bet Range")

# Display with action colors
display.display_action_breakdown(title="Strategy by Hand")

# Plot with matplotlib (if available)
display.plot(title="Range Heatmap", save_path="range.png")
```

## Example Workflows

### Workflow 1: Analyze Your Database and Find Leaks

```python
from moid.db import get_connection
from moid.analysis import PopulationAnalyzer

conn = get_connection("hands.db")
analyzer = PopulationAnalyzer(conn)

# Get full analysis
stats = analyzer.analyze()

print("=== Population Tendencies ===")
print(f"VPIP: {stats.overall.vpip:.1f}% ({stats.overall.get_tendency('vpip')})")
print(f"Fold to C-Bet: {stats.overall.fold_to_cbet:.1f}%")

print("\n=== Exploits to Use ===")
for exploit in analyzer.get_exploits():
    print(f"• {exploit}")

print("\n=== Position Breakdown ===")
for pos, pos_stats in stats.by_position.items():
    print(f"{pos}: VPIP={pos_stats.vpip:.1f}%, PFR={pos_stats.pfr:.1f}%")
```

### Workflow 2: Solve a Specific Spot

```python
from moid.game.cards import Card
from moid.solver import AdaptiveSolver
from moid.analysis.stats import PlayerStats
from moid.db import get_connection
from moid.analysis import compute_stats

# Load population stats from database
conn = get_connection("hands.db")
stats = compute_stats(conn)
conn.close()

# Define the spot: BTN vs BB single raised pot, flop
board = [Card.from_string(c) for c in ["Qs", "8d", "3c"]]

solver = AdaptiveSolver(
    stats=stats,
    board=board,
    starting_pot=6.5,    # Typical SRP
    effective_stack=97,  # After preflop betting
)

# Get recommendations for different hand strengths
for strength in ["strong", "medium", "weak", "air"]:
    action, reason = solver.get_recommended_action(strength, facing_bet=False)
    print(f"{strength.upper()}: {action} - {reason}")
```

### Workflow 3: Real-Time Session Tracking

```python
from moid.classifier import BayesianClassifier

# Track a specific villain during a session
villain_tracker = BayesianClassifier()

# As you observe their actions...
villain_tracker.observe_preflop(vpip=True, pfr=False)   # Limped
villain_tracker.observe_preflop(vpip=True, pfr=False)   # Called raise
villain_tracker.observe_postflop(aggressive=False)       # Check-called
villain_tracker.observe_preflop(vpip=True, pfr=False)   # Called again

# Check classification
archetype = villain_tracker.get_classification()
print(f"Villain appears to be: {archetype.name}")
print(f"Description: {archetype.description()}")

# Get probabilities
probs = villain_tracker.get_probabilities()
for arch, prob in sorted(probs.items(), key=lambda x: -x[1])[:3]:
    print(f"  {arch.name}: {prob:.0%}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Scripts                          │
│  import_hands.py │ analyze_population.py │ solve_spot.py   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│    Parser     │     │   Analysis    │     │    Solver     │
│───────────────│     │───────────────│     │───────────────│
│ • IgnitionPar │     │ • PlayerStats │     │ • CFRSolver   │
│ • Hand/Action │     │ • Population  │     │ • BestResp    │
│ • Position    │     │   Analyzer    │     │ • Strategy    │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        ▼                     │                     ▼
┌───────────────┐             │             ┌───────────────┐
│   Database    │◄────────────┘             │     Game      │
│───────────────│                           │───────────────│
│ • Schema      │                           │ • Cards/Hand  │
│ • Repository  │                           │ • GameTree    │
│ • SQLite      │                           │ • Equity      │
└───────────────┘                           │ • Abstraction │
                                            └───────────────┘
        ┌─────────────────────┬─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Classifier   │     │ Visualization │     │    treys      │
│───────────────│     │───────────────│     │  (external)   │
│ • Archetypes  │     │ • RangeDisp   │     │───────────────│
│ • Bayesian    │     │ • 13x13 grid  │     │ • Hand eval   │
└───────────────┘     └───────────────┘     └───────────────┘
```

## Key Concepts

### Information Sets
In poker, an information set represents what a player knows at a decision point: their hole cards and the action history. The solver computes strategies for each information set.

### CFR (Counterfactual Regret Minimization)
CFR iteratively improves strategies by tracking "regret" - how much better we could have done by taking different actions. Over many iterations, it converges to Nash equilibrium.

### Adaptive vs GTO
- **GTO (Nash)**: Unexploitable strategy that doesn't lose to any counter-strategy
- **Adaptive**: Maximizes EV against a specific opponent model, but may itself be exploitable

### Hand Abstraction
Groups similar hands together to reduce computational complexity. Hands are bucketed by equity on each board.

### Population Stats
Since Ignition/Bovada uses anonymous tables, we can't track individual players. Instead, we model aggregate population tendencies and exploit systematic leaks.

## License

MIT License
