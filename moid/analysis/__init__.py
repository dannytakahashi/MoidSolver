"""Population analysis module."""

from .stats import PlayerStats, compute_stats
from .population import PopulationAnalyzer, PopulationStats
from .benchmarks import (
    GTOBenchmarks,
    GTO_BENCHMARKS,
    PositionBenchmarks,
    BoardTextureBenchmarks,
    MicrostakesAdjustments,
    MICROSTAKES_ADJUSTMENTS,
    get_benchmark,
    get_adjusted_target,
    classify_board_texture,
)
from .hero import HeroAnalyzer, HeroStats, Leak
from .spots import SpotAnalyzer, SpotType, SpotStats, SpotBreakdown
from .flagger import HandFlagger, FlaggedHand, FlagReason, FlagCriteria

__all__ = [
    # Stats
    "PlayerStats",
    "compute_stats",
    # Population
    "PopulationAnalyzer",
    "PopulationStats",
    # Benchmarks
    "GTOBenchmarks",
    "GTO_BENCHMARKS",
    "PositionBenchmarks",
    "BoardTextureBenchmarks",
    "MicrostakesAdjustments",
    "MICROSTAKES_ADJUSTMENTS",
    "get_benchmark",
    "get_adjusted_target",
    "classify_board_texture",
    # Hero analysis
    "HeroAnalyzer",
    "HeroStats",
    "Leak",
    # Spot analysis
    "SpotAnalyzer",
    "SpotType",
    "SpotStats",
    "SpotBreakdown",
    # Hand flagger
    "HandFlagger",
    "FlaggedHand",
    "FlagReason",
    "FlagCriteria",
]
