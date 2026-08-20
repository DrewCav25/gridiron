"""Data loading with local parquet caching.

Every raw nflverse pull goes through :func:`cached`, so re-running the
pipeline (which you will do hundreds of times) hits disk instead of the
network. Delete ``data/cache/`` to force a refresh.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable

import polars as pl

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# nflverse coverage. Play-by-play, weekly stats, rosters and schedules go
# back to 1999; snap counts start in 2012; ff_opportunity in 2006.
FIRST_SEASON = 1999
FIRST_SNAP_COUNT_SEASON = 2012
FIRST_FF_OPPORTUNITY_SEASON = 2006

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")


def cached(name: str, fn: Callable[[], pl.DataFrame], refresh: bool = False) -> pl.DataFrame:
    """Return ``fn()``, memoised to ``data/cache/<name>.parquet``."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists() and not refresh:
        log.debug("cache hit: %s", path.name)
        return pl.read_parquet(path)
    log.info("fetching %s ...", name)
    df = fn()
    df.write_parquet(path)
    return df


def _seasons(seasons: Iterable[int]) -> list[int]:
    return sorted(set(int(s) for s in seasons))


def load_weekly_stats(seasons: Iterable[int], refresh: bool = False) -> pl.DataFrame:
    """Weekly offensive player stats."""
    import nflreadpy as nfl

    yrs = _seasons(seasons)
    key = f"weekly_{yrs[0]}_{yrs[-1]}"
    return cached(key, lambda: nfl.load_player_stats(seasons=yrs), refresh)


def load_snap_counts(seasons: Iterable[int], refresh: bool = False) -> pl.DataFrame:
    """Snap counts (2012+). Snap share is one of the stickiest signals we have."""
    import nflreadpy as nfl

    yrs = [y for y in _seasons(seasons) if y >= FIRST_SNAP_COUNT_SEASON]
    if not yrs:
        return pl.DataFrame()
    key = f"snaps_{yrs[0]}_{yrs[-1]}"
    return cached(key, lambda: nfl.load_snap_counts(seasons=yrs), refresh)


def load_ff_opportunity(seasons: Iterable[int], refresh: bool = False) -> pl.DataFrame:
    """Expected fantasy points from the ffopportunity model (2006+).

    This is the strongest opportunity-only baseline available: expected
    yards, receptions and touchdowns given the plays a player was actually
    involved in. Great features, and a baseline worth beating.
    """
    import nflreadpy as nfl

    yrs = [y for y in _seasons(seasons) if y >= FIRST_FF_OPPORTUNITY_SEASON]
    if not yrs:
        return pl.DataFrame()
    key = f"ffopp_{yrs[0]}_{yrs[-1]}"
    return cached(key, lambda: nfl.load_ff_opportunity(seasons=yrs, stat_type="weekly"), refresh)


def load_rosters(seasons: Iterable[int], refresh: bool = False) -> pl.DataFrame:
    """Season rosters — source of age, experience and cross-platform IDs."""
    import nflreadpy as nfl

    yrs = _seasons(seasons)
    key = f"rosters_{yrs[0]}_{yrs[-1]}"
    return cached(key, lambda: nfl.load_rosters(seasons=yrs), refresh)


def load_draft_picks(refresh: bool = False) -> pl.DataFrame:
    """NFL draft capital — the main prior available for rookies."""
    import nflreadpy as nfl

    return cached("draft_picks", lambda: nfl.load_draft_picks(), refresh)


def load_players(refresh: bool = False) -> pl.DataFrame:
    """Player master table (birth dates, rookie season, ID crosswalk)."""
    import nflreadpy as nfl

    return cached("players", lambda: nfl.load_players(), refresh)


def load_ff_rankings(refresh: bool = False) -> pl.DataFrame:
    """FantasyPros ECR / ADP via DynastyProcess.

    NOTE: this hits github.com/dynastyprocess directly and can 403 from
    sandboxed or proxied networks. It works from a normal home connection.
    Everything else in this module comes from nflverse-data releases and is
    unaffected, so a failure here degrades gracefully — you lose the
    consensus baseline and ADP, not the pipeline.
    """
    import nflreadpy as nfl

    return cached("ff_rankings", lambda: nfl.load_ff_rankings(type="draft"), refresh)
