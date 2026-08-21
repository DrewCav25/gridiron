"""League scoring and roster configuration.

Everything downstream reads from these objects. Nothing in this project
hardcodes PPR — that is the single most common design flaw in public
fantasy football tooling and it makes projections useless to anyone whose
league is set up differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import json


@dataclass(frozen=True)
class ScoringConfig:
    """Points awarded per statistical event.

    Defaults are standard (non-PPR) scoring. Use the ``ppr``, ``half_ppr``
    and ``standard`` constructors for the common presets.
    """

    # Passing
    pass_yards: float = 0.04  # 1 pt / 25 yds
    pass_td: float = 4.0
    pass_int: float = -2.0
    pass_2pt: float = 2.0

    # Rushing
    rush_yards: float = 0.1  # 1 pt / 10 yds
    rush_td: float = 6.0
    rush_2pt: float = 2.0

    # Receiving
    rec_yards: float = 0.1
    rec_td: float = 6.0
    rec_2pt: float = 2.0
    reception: float = 0.0  # 0.5 = half PPR, 1.0 = full PPR

    # Position-specific reception bonus (TE premium)
    te_reception_bonus: float = 0.0

    # Misc
    fumble_lost: float = -2.0
    special_teams_td: float = 6.0

    # Bonuses (0 disables). Applied once if the threshold is met in a game.
    bonus_pass_300: float = 0.0
    bonus_rush_100: float = 0.0
    bonus_rec_100: float = 0.0

    name: str = "standard"

    @classmethod
    def standard(cls) -> "ScoringConfig":
        return cls(reception=0.0, name="standard")

    @classmethod
    def half_ppr(cls) -> "ScoringConfig":
        return cls(reception=0.5, name="half_ppr")

    @classmethod
    def ppr(cls) -> "ScoringConfig":
        return cls(reception=1.0, name="ppr")

    @classmethod
    def te_premium(cls, base: float = 1.0, bonus: float = 0.5) -> "ScoringConfig":
        return cls(reception=base, te_reception_bonus=bonus, name="te_premium")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScoringConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class LeagueConfig:
    """Roster structure and league size.

    Replacement level for Value Over Replacement is derived from these
    numbers, so they matter as much as the scoring settings do.
    """

    teams: int = 12
    qb: int = 1
    rb: int = 2
    wr: int = 2
    te: int = 1
    flex: int = 1  # RB/WR/TE
    superflex: int = 0  # QB/RB/WR/TE
    k: int = 1
    dst: int = 1
    bench: int = 6

    flex_positions: tuple[str, ...] = ("RB", "WR", "TE")
    superflex_positions: tuple[str, ...] = ("QB", "RB", "WR", "TE")

    scoring: ScoringConfig = field(default_factory=ScoringConfig.half_ppr)

    @property
    def starters_per_team(self) -> int:
        return (
            self.qb + self.rb + self.wr + self.te
            + self.flex + self.superflex + self.k + self.dst
        )

    @property
    def roster_size(self) -> int:
        return self.starters_per_team + self.bench

    @property
    def total_drafted(self) -> int:
        return self.roster_size * self.teams

    def base_starters(self, position: str) -> int:
        """Dedicated (non-flex) starting slots for a position, league-wide."""
        per_team = {
            "QB": self.qb, "RB": self.rb, "WR": self.wr,
            "TE": self.te, "K": self.k, "DST": self.dst,
        }.get(position.upper(), 0)
        return per_team * self.teams

    def flex_slots(self, position: str) -> int:
        """League-wide flex slots a position is eligible for."""
        n = 0
        if position.upper() in self.flex_positions:
            n += self.flex * self.teams
        if position.upper() in self.superflex_positions:
            n += self.superflex * self.teams
        return n


# Convenience presets
STANDARD_12 = LeagueConfig(scoring=ScoringConfig.standard())
HALF_PPR_12 = LeagueConfig(scoring=ScoringConfig.half_ppr())
PPR_12 = LeagueConfig(scoring=ScoringConfig.ppr())
PPR_10 = LeagueConfig(teams=10, scoring=ScoringConfig.ppr())
SUPERFLEX_12 = LeagueConfig(superflex=1, scoring=ScoringConfig.half_ppr())
