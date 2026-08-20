"""Fantasy point computation from nflverse weekly player stats.

Takes a polars DataFrame of weekly stats (from ``nflreadpy.load_player_stats``)
and a :class:`ScoringConfig`, and returns the same frame with a
``fantasy_points_custom`` column.

The engine is validated against nflverse's own ``fantasy_points`` and
``fantasy_points_ppr`` columns in ``tests/test_scoring.py`` — if our standard
and PPR presets don't reproduce theirs, something is wrong.
"""

from __future__ import annotations

import polars as pl

from .config import ScoringConfig

# Columns we need; missing ones are filled with 0 so the engine works on
# partial frames (e.g. a single position group).
_STAT_COLS = [
    "passing_yards", "passing_tds", "passing_interceptions", "passing_2pt_conversions",
    "carries", "rushing_yards", "rushing_tds", "rushing_2pt_conversions",
    "receptions", "receiving_yards", "receiving_tds", "receiving_2pt_conversions",
    "targets", "special_teams_tds",
    "sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost",
]


def _col(df: pl.DataFrame, name: str) -> pl.Expr:
    """Reference a stat column, or a literal 0 if it isn't present."""
    if name in df.columns:
        return pl.col(name).fill_null(0.0).cast(pl.Float64)
    return pl.lit(0.0)


def add_fantasy_points(
    df: pl.DataFrame,
    scoring: ScoringConfig,
    out_col: str = "fantasy_points_custom",
) -> pl.DataFrame:
    """Add a fantasy points column computed under ``scoring``."""
    s = scoring

    pass_yds = _col(df, "passing_yards")
    rush_yds = _col(df, "rushing_yards")
    rec_yds = _col(df, "receiving_yards")
    recs = _col(df, "receptions")

    fumbles_lost = (
        _col(df, "sack_fumbles_lost")
        + _col(df, "rushing_fumbles_lost")
        + _col(df, "receiving_fumbles_lost")
    )

    base = (
        pass_yds * s.pass_yards
        + _col(df, "passing_tds") * s.pass_td
        + _col(df, "passing_interceptions") * s.pass_int
        + _col(df, "passing_2pt_conversions") * s.pass_2pt
        + rush_yds * s.rush_yards
        + _col(df, "rushing_tds") * s.rush_td
        + _col(df, "rushing_2pt_conversions") * s.rush_2pt
        + rec_yds * s.rec_yards
        + _col(df, "receiving_tds") * s.rec_td
        + _col(df, "receiving_2pt_conversions") * s.rec_2pt
        + recs * s.reception
        + _col(df, "special_teams_tds") * s.special_teams_td
        + fumbles_lost * s.fumble_lost
    )

    # TE premium: extra points per reception for tight ends only.
    if s.te_reception_bonus and "position" in df.columns:
        base = base + pl.when(pl.col("position") == "TE").then(
            recs * s.te_reception_bonus
        ).otherwise(0.0)

    # Yardage bonuses, applied per game when the threshold is met.
    if s.bonus_pass_300:
        base = base + pl.when(pass_yds >= 300).then(s.bonus_pass_300).otherwise(0.0)
    if s.bonus_rush_100:
        base = base + pl.when(rush_yds >= 100).then(s.bonus_rush_100).otherwise(0.0)
    if s.bonus_rec_100:
        base = base + pl.when(rec_yds >= 100).then(s.bonus_rec_100).otherwise(0.0)

    return df.with_columns(base.alias(out_col))


def weekly_to_season(
    df: pl.DataFrame,
    points_col: str = "fantasy_points_custom",
    regular_season_only: bool = True,
) -> pl.DataFrame:
    """Collapse weekly rows into one row per player-season.

    Produces both total and per-game points, plus games played. Keeping
    these separate matters: availability and productivity are different
    prediction problems and conflating them is why projections are
    systematically wrong on injury-prone players.
    """
    if regular_season_only and "season_type" in df.columns:
        df = df.filter(pl.col("season_type") == "REG")

    # A game counts as played if the player recorded any offensive opportunity.
    opportunity = (
        _col(df, "attempts") + _col(df, "carries") + _col(df, "targets")
    )
    df = df.with_columns((opportunity > 0).alias("_played"))

    agg = [
        pl.col(points_col).sum().alias("fantasy_points_total"),
        pl.col("_played").sum().alias("games_played"),
        pl.col(points_col).mean().alias("ppg_all_weeks"),
    ]

    # Carry through the counting stats we need for feature engineering.
    for c in _STAT_COLS:
        if c in df.columns:
            agg.append(pl.col(c).fill_null(0.0).sum().alias(c))

    # Rate stats: average over weeks where the player was on the field.
    for c in ["target_share", "air_yards_share", "wopr", "racr",
              "passing_epa", "rushing_epa", "receiving_epa"]:
        if c in df.columns:
            agg.append(pl.col(c).mean().alias(f"{c}_mean"))

    keys = ["player_id", "season"]
    firsts = [
        pl.col(c).drop_nulls().first().alias(c)
        for c in ["player_display_name", "position", "position_group", "team"]
        if c in df.columns
    ]

    out = df.group_by(keys).agg(firsts + agg)

    return out.with_columns(
        pl.when(pl.col("games_played") > 0)
        .then(pl.col("fantasy_points_total") / pl.col("games_played"))
        .otherwise(0.0)
        .alias("fantasy_points_per_game")
    ).sort(["season", "fantasy_points_total"], descending=[False, True])
