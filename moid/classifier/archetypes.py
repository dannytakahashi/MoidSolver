"""Player archetype classification system."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from moid.analysis.stats import PlayerStats


class PlayerArchetype(Enum):
    """
    Standard player archetypes based on VPIP/PFR grid.

    These archetypes capture the main playing styles seen at microstakes:
    - Fish: Loose-passive players who play too many hands passively
    - Nit: Tight-passive players who only play premium hands
    - TAG: Tight-aggressive players with balanced, solid play
    - LAG: Loose-aggressive players with wide ranges and aggression
    - Maniac: Extremely aggressive players (rare but exploitable)
    - Calling Station: Players who call too much but rarely raise
    """
    UNKNOWN = auto()
    FISH = auto()
    NIT = auto()
    TAG = auto()
    LAG = auto()
    MANIAC = auto()
    CALLING_STATION = auto()

    def description(self) -> str:
        """Human-readable description of archetype."""
        descriptions = {
            self.UNKNOWN: "Unknown - insufficient data",
            self.FISH: "Fish - loose-passive, plays too many hands",
            self.NIT: "Nit - tight-passive, only premium hands",
            self.TAG: "TAG - tight-aggressive, solid player",
            self.LAG: "LAG - loose-aggressive, wide ranges",
            self.MANIAC: "Maniac - extremely aggressive, overbluffs",
            self.CALLING_STATION: "Calling Station - calls too much",
        }
        return descriptions.get(self, "Unknown")


@dataclass
class ArchetypeProfile:
    """Statistical profile for an archetype."""
    archetype: PlayerArchetype
    vpip_range: tuple[float, float]  # (min, max)
    pfr_range: tuple[float, float]
    af_range: tuple[float, float]
    three_bet_range: tuple[float, float] = (0, 100)

    def matches(self, stats: PlayerStats) -> bool:
        """Check if stats match this profile."""
        vpip_ok = self.vpip_range[0] <= stats.vpip <= self.vpip_range[1]
        pfr_ok = self.pfr_range[0] <= stats.pfr <= self.pfr_range[1]
        af_ok = self.af_range[0] <= stats.af <= self.af_range[1]
        return vpip_ok and pfr_ok and af_ok

    def distance(self, stats: PlayerStats) -> float:
        """Calculate distance from stats to profile center."""
        vpip_center = (self.vpip_range[0] + self.vpip_range[1]) / 2
        pfr_center = (self.pfr_range[0] + self.pfr_range[1]) / 2
        af_center = (self.af_range[0] + self.af_range[1]) / 2

        # Normalize by range widths
        vpip_width = max(self.vpip_range[1] - self.vpip_range[0], 1)
        pfr_width = max(self.pfr_range[1] - self.pfr_range[0], 1)
        af_width = max(self.af_range[1] - self.af_range[0], 0.1)

        vpip_dist = ((stats.vpip - vpip_center) / vpip_width) ** 2
        pfr_dist = ((stats.pfr - pfr_center) / pfr_width) ** 2
        af_dist = ((stats.af - af_center) / af_width) ** 2

        return np.sqrt(vpip_dist + pfr_dist + af_dist)


# Standard archetype profiles
ARCHETYPE_PROFILES = [
    ArchetypeProfile(
        archetype=PlayerArchetype.NIT,
        vpip_range=(0, 18),
        pfr_range=(0, 15),
        af_range=(0.5, 3.0),
    ),
    ArchetypeProfile(
        archetype=PlayerArchetype.TAG,
        vpip_range=(18, 28),
        pfr_range=(14, 24),
        af_range=(2.0, 4.0),
    ),
    ArchetypeProfile(
        archetype=PlayerArchetype.LAG,
        vpip_range=(26, 40),
        pfr_range=(20, 35),
        af_range=(2.5, 5.0),
    ),
    ArchetypeProfile(
        archetype=PlayerArchetype.FISH,
        vpip_range=(35, 100),
        pfr_range=(0, 15),
        af_range=(0, 2.0),
    ),
    ArchetypeProfile(
        archetype=PlayerArchetype.CALLING_STATION,
        vpip_range=(35, 100),
        pfr_range=(0, 10),
        af_range=(0, 1.0),
    ),
    ArchetypeProfile(
        archetype=PlayerArchetype.MANIAC,
        vpip_range=(40, 100),
        pfr_range=(30, 100),
        af_range=(4.0, 100),
    ),
]


class ArchetypeClassifier:
    """
    Rule-based player archetype classifier.

    Classifies players based on their VPIP, PFR, and aggression stats
    using predefined archetype profiles.
    """

    def __init__(self, min_hands: int = 30):
        """
        Initialize classifier.

        Args:
            min_hands: Minimum hands required for classification
        """
        self.min_hands = min_hands
        self.profiles = ARCHETYPE_PROFILES

    def classify(self, stats: PlayerStats) -> PlayerArchetype:
        """
        Classify player based on their stats.

        Args:
            stats: Player statistics

        Returns:
            Most likely player archetype
        """
        if stats.hands < self.min_hands:
            return PlayerArchetype.UNKNOWN

        # Find best matching profile
        best_archetype = PlayerArchetype.UNKNOWN
        best_distance = float("inf")

        for profile in self.profiles:
            if profile.matches(stats):
                distance = profile.distance(stats)
                if distance < best_distance:
                    best_distance = distance
                    best_archetype = profile.archetype

        # If no exact match, find closest
        if best_archetype == PlayerArchetype.UNKNOWN:
            for profile in self.profiles:
                distance = profile.distance(stats)
                if distance < best_distance:
                    best_distance = distance
                    best_archetype = profile.archetype

        return best_archetype

    def get_exploits(self, archetype: PlayerArchetype) -> list[str]:
        """
        Get exploitation strategies for an archetype.

        Args:
            archetype: Player archetype

        Returns:
            List of exploitation recommendations
        """
        exploits = {
            PlayerArchetype.FISH: [
                "Value bet wider - they call too much",
                "Don't bluff rivers - they call down light",
                "Isolate preflop with wider range",
                "Bet larger for value",
            ],
            PlayerArchetype.NIT: [
                "Steal blinds aggressively",
                "Fold to their 3-bets (they have it)",
                "Give up on bluffs if they continue",
                "Respect their postflop aggression",
            ],
            PlayerArchetype.TAG: [
                "Play straightforward",
                "Avoid marginal spots",
                "3-bet light occasionally to balance",
                "They're harder to exploit - minimize variance",
            ],
            PlayerArchetype.LAG: [
                "Call down lighter",
                "Let them bluff into you",
                "Don't 4-bet bluff (they're sticky)",
                "Check strong hands to induce bluffs",
            ],
            PlayerArchetype.MANIAC: [
                "Trap with strong hands",
                "Call down very wide",
                "Don't try to bluff",
                "Let them hang themselves",
            ],
            PlayerArchetype.CALLING_STATION: [
                "Never bluff",
                "Value bet thinly and often",
                "Overbet for value",
                "Don't worry about being balanced",
            ],
        }

        return exploits.get(archetype, ["Insufficient data - gather more hands"])


@dataclass
class BayesianClassifier:
    """
    Bayesian classifier for real-time player classification.

    Uses observed actions to update probability distribution
    over player archetypes during a session.
    """

    # Prior probabilities (microstakes population estimates)
    priors: dict[PlayerArchetype, float] = field(default_factory=lambda: {
        PlayerArchetype.FISH: 0.40,
        PlayerArchetype.NIT: 0.15,
        PlayerArchetype.TAG: 0.20,
        PlayerArchetype.LAG: 0.10,
        PlayerArchetype.MANIAC: 0.05,
        PlayerArchetype.CALLING_STATION: 0.10,
    })

    # Current posteriors
    posteriors: dict[PlayerArchetype, float] = field(default_factory=dict)

    # Observed actions
    vpip_count: int = 0
    pfr_count: int = 0
    hands_count: int = 0
    aggressive_count: int = 0
    passive_count: int = 0

    def __post_init__(self):
        """Initialize posteriors from priors."""
        if not self.posteriors:
            self.posteriors = self.priors.copy()

    def observe_preflop(self, vpip: bool, pfr: bool) -> None:
        """
        Update beliefs based on preflop action.

        Args:
            vpip: Did player voluntarily put money in?
            pfr: Did player raise preflop?
        """
        self.hands_count += 1
        if vpip:
            self.vpip_count += 1
        if pfr:
            self.pfr_count += 1

        self._update_posteriors()

    def observe_postflop(self, aggressive: bool) -> None:
        """
        Update beliefs based on postflop action.

        Args:
            aggressive: Was the action aggressive (bet/raise)?
        """
        if aggressive:
            self.aggressive_count += 1
        else:
            self.passive_count += 1

        self._update_posteriors()

    def _update_posteriors(self) -> None:
        """Recompute posterior probabilities."""
        if self.hands_count == 0:
            return

        observed_vpip = (self.vpip_count / self.hands_count) * 100
        observed_pfr = (self.pfr_count / self.hands_count) * 100

        total_postflop = self.aggressive_count + self.passive_count
        observed_af = (
            self.aggressive_count / max(self.passive_count, 1)
            if total_postflop > 0
            else 1.5
        )

        # Compute likelihoods for each archetype
        likelihoods = {}
        for profile in ARCHETYPE_PROFILES:
            likelihood = self._compute_likelihood(
                profile, observed_vpip, observed_pfr, observed_af
            )
            likelihoods[profile.archetype] = likelihood

        # Apply Bayes rule
        total = sum(
            self.priors[arch] * likelihoods.get(arch, 0.01)
            for arch in self.priors
        )

        if total > 0:
            for arch in self.priors:
                self.posteriors[arch] = (
                    self.priors[arch] * likelihoods.get(arch, 0.01)
                ) / total

    def _compute_likelihood(
        self,
        profile: ArchetypeProfile,
        vpip: float,
        pfr: float,
        af: float,
    ) -> float:
        """Compute likelihood of observations given archetype profile."""
        # Use Gaussian likelihood centered on profile midpoints
        vpip_mid = (profile.vpip_range[0] + profile.vpip_range[1]) / 2
        pfr_mid = (profile.pfr_range[0] + profile.pfr_range[1]) / 2
        af_mid = (profile.af_range[0] + profile.af_range[1]) / 2

        vpip_std = (profile.vpip_range[1] - profile.vpip_range[0]) / 2 + 5
        pfr_std = (profile.pfr_range[1] - profile.pfr_range[0]) / 2 + 5
        af_std = (profile.af_range[1] - profile.af_range[0]) / 2 + 0.5

        vpip_ll = np.exp(-((vpip - vpip_mid) ** 2) / (2 * vpip_std**2))
        pfr_ll = np.exp(-((pfr - pfr_mid) ** 2) / (2 * pfr_std**2))
        af_ll = np.exp(-((af - af_mid) ** 2) / (2 * af_std**2))

        return vpip_ll * pfr_ll * af_ll

    def get_classification(self) -> PlayerArchetype:
        """
        Get most likely archetype based on current posteriors.

        Returns:
            Most probable player archetype
        """
        if self.hands_count < 5:
            return PlayerArchetype.UNKNOWN

        return max(self.posteriors, key=lambda a: self.posteriors[a])

    def get_probabilities(self) -> dict[PlayerArchetype, float]:
        """
        Get current probability distribution over archetypes.

        Returns:
            Dict mapping archetypes to probabilities
        """
        return self.posteriors.copy()

    def reset(self) -> None:
        """Reset classifier to prior state."""
        self.posteriors = self.priors.copy()
        self.vpip_count = 0
        self.pfr_count = 0
        self.hands_count = 0
        self.aggressive_count = 0
        self.passive_count = 0
