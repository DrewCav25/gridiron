"""Sleeper league sync.

Projections are only useful if they match the league you actually play in.
Sleeper's read API is public, unauthenticated and well documented, so a
league ID is enough to pull real scoring settings and roster structure
instead of guessing.

Find your league ID in the Sleeper web app URL:
``https://sleeper.com/leagues/<LEAGUE_ID>/team``

Note: this module is written against the documented API but is exercised
by unit tests against recorded payloads rather than live calls, so that
the test suite stays offline and deterministic.
"""

from __future__ import annotations

from typing import Any

from .config import LeagueConfig, ScoringConfig

API = "https://api.sleeper.app/v1"
TIMEOUT = 15

# Sleeper's scoring keys -> our ScoringConfig fields. Sleeper exposes far
# more keys than this (defensive, kicking, IDP); we map the offensive ones
# the projection models actually cover and ignore the rest.
SCORING_MAP = {
    "pass_yd": "pass_yards",
    "pass_td": "pass_td",
    "pass_int": "pass_int",
    "pass_2pt": "pass_2pt",
    "rush_yd": "rush_yards",
    "rush_td": "rush_td",
    "rush_2pt": "rush_2pt",
    "rec_yd": "rec_yards",
    "rec_td": "rec_td",
    "rec_2pt": "rec_2pt",
    "rec": "reception",
    "fum_lost": "fumble_lost",
    "bonus_pass_yd_300": "bonus_pass_300",
    "bonus_rush_yd_100": "bonus_rush_100",
    "bonus_rec_yd_100": "bonus_rec_100",
}


def fetch_league(league_id: str) -> dict[str, Any]:
    """GET the raw league payload. Requires network access."""
    import requests

    resp = requests.get(f"{API}/league/{league_id}", timeout=TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if not payload:
        raise ValueError(f"Sleeper returned no league for id {league_id!r}")
    return payload


def parse_scoring(payload: dict[str, Any]) -> ScoringConfig:
    """Build a ScoringConfig from a Sleeper league payload.

    Unmapped keys fall back to our defaults rather than erroring, so an
    unusual league still produces a usable config instead of a stack trace.
    """
    settings = payload.get("scoring_settings") or {}
    kwargs: dict[str, float] = {}
    for sleeper_key, our_field in SCORING_MAP.items():
        if sleeper_key in settings:
            kwargs[our_field] = float(settings[sleeper_key])

    # TE premium is expressed as a position-specific reception bonus.
    te_bonus = settings.get("bonus_rec_te")
    if te_bonus:
        kwargs["te_reception_bonus"] = float(te_bonus)

    rec = kwargs.get("reception", 0.0)
    name = {0.0: "standard", 0.5: "half_ppr", 1.0: "ppr"}.get(rec, f"custom_{rec}")
    return ScoringConfig(**kwargs, name=name)


def parse_league(payload: dict[str, Any]) -> LeagueConfig:
    """Build a LeagueConfig from a Sleeper league payload.

    Sleeper describes rosters as an ordered list of slot labels, e.g.
    ``["QB","RB","RB","WR","WR","TE","FLEX","K","DEF","BN","BN",...]``.
    Counting the labels gives the roster structure directly.
    """
    positions = payload.get("roster_positions") or []
    counts: dict[str, int] = {}
    for slot in positions:
        counts[slot] = counts.get(slot, 0) + 1

    return LeagueConfig(
        teams=int(payload.get("total_rosters") or 12),
        qb=counts.get("QB", 1),
        rb=counts.get("RB", 2),
        wr=counts.get("WR", 2),
        te=counts.get("TE", 1),
        flex=counts.get("FLEX", 0) + counts.get("WRRB_FLEX", 0) + counts.get("REC_FLEX", 0),
        superflex=counts.get("SUPER_FLEX", 0),
        k=counts.get("K", 0),
        dst=counts.get("DEF", 0),
        bench=counts.get("BN", 6),
        scoring=parse_scoring(payload),
    )


def load_league(league_id: str) -> LeagueConfig:
    """Fetch and parse a league in one call."""
    return parse_league(fetch_league(league_id))


def describe(league: LeagueConfig) -> str:
    """Human-readable summary, for confirming the sync did what you expect."""
    s = league.scoring
    lines = [
        f"{league.teams}-team league, {league.roster_size}-man rosters",
        f"  starters: QB {league.qb} | RB {league.rb} | WR {league.wr} | "
        f"TE {league.te} | FLEX {league.flex} | SFLEX {league.superflex} | "
        f"K {league.k} | DST {league.dst}",
        f"  scoring: {s.name} ({s.reception} per reception, "
        f"{s.pass_td} pass TD, {s.rec_td} rec TD)",
    ]
    if s.te_reception_bonus:
        lines.append(f"  TE premium: +{s.te_reception_bonus} per TE reception")
    return "\n".join(lines)
