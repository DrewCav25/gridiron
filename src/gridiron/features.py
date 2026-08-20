"""Build the modelling panel: one row per player-season, features lagged.

The central rule of this module: **to predict season N, use only information
available before season N started.** Every feature is drawn from season N-1
or earlier. The target comes from season N. Violating this is the leakage
bug that makes most public fantasy models look better than they are.
"""

from __future__ import annotations

import polars as pl

from . import data as D
from .config import ScoringConfig
from .scoring import add_fantasy_points, weekly_to_season

# Opportunity metrics: stable year over year. These are what we predict from.
STICKY_FEATURES = [
    "target_share_mean", "air_yards_share_mean", "wopr_mean",
    "targets", "carries", "snap_pct_mean", "attempts",
]

# Efficiency metrics: regress hard to the mean. Included so the model can
# learn *how much* to discount them, not because they carry forward.
VOLATILE_FEATURES = [
    "td_rate", "yards_per_carry", "yards_per_target", "catch_rate",
    "passing_epa_mean", "rushing_epa_mean", "receiving_epa_mean",
]


def build_season_table(
    seasons: list[int],
    scoring: ScoringConfig,
    refresh: bool = False,
) -> pl.DataFrame:
    """One row per player-season with fantasy points and raw counting stats."""
    weekly = D.load_weekly_stats(seasons, refresh=refresh)
    weekly = weekly.filter(pl.col("position").is_in(list(D.FANTASY_POSITIONS)))
    weekly = add_fantasy_points(weekly, scoring)
    season = weekly_to_season(weekly)
    season = _attach_snap_share(season, seasons, refresh=refresh)
    season = _attach_age(season, refresh=refresh)
    return _derive_rates(season)


def _attach_snap_share(
    season: pl.DataFrame, seasons: list[int], refresh: bool = False
) -> pl.DataFrame:
    """Join mean offensive snap share (2012+).

    Snap counts key on Pro-Football-Reference IDs, so this routes through
    the players table to get back to gsis_id.
    """
    snaps = D.load_snap_counts(seasons, refresh=refresh)
    if snaps.is_empty():
        return season.with_columns(pl.lit(None, dtype=pl.Float64).alias("snap_pct_mean"))

    snaps = (
        snaps.filter(pl.col("game_type") == "REG")
        .group_by(["pfr_player_id", "season"])
        .agg(pl.col("offense_pct").mean().alias("snap_pct_mean"))
    )

    players = D.load_players(refresh=refresh).select(
        pl.col("gsis_id").alias("player_id"), pl.col("pfr_id").alias("pfr_player_id")
    ).drop_nulls()

    snaps = snaps.join(players, on="pfr_player_id", how="inner").drop("pfr_player_id")
    return season.join(snaps, on=["player_id", "season"], how="left")


def _attach_age(season: pl.DataFrame, refresh: bool = False) -> pl.DataFrame:
    """Age at the start of the season, and years of experience.

    Age curves differ sharply by position — running backs fall off a cliff
    that wide receivers don't — so the model needs both age and position.
    """
    players = D.load_players(refresh=refresh).select(
        pl.col("gsis_id").alias("player_id"), "birth_date", "rookie_season"
    ).drop_nulls(subset=["player_id"])

    # birth_date arrives as a string in some nflverse releases.
    if players.schema["birth_date"] == pl.String:
        players = players.with_columns(
            pl.col("birth_date").str.to_date("%Y-%m-%d", strict=False)
        )

    out = season.join(players, on="player_id", how="left")
    return out.with_columns(
        (pl.col("season") - pl.col("birth_date").dt.year()).cast(pl.Float64).alias("age"),
        (pl.col("season") - pl.col("rookie_season")).cast(pl.Float64).alias("years_exp"),
    ).drop(["birth_date", "rookie_season"])


def _derive_rates(season: pl.DataFrame) -> pl.DataFrame:
    """Efficiency rates — the metrics that regress to the mean."""
    def safe(num: str, den: str) -> pl.Expr:
        return (
            pl.when(pl.col(den) > 0)
            .then(pl.col(num) / pl.col(den))
            .otherwise(None)
        )

    touches = pl.col("carries") + pl.col("receptions")
    total_td = pl.col("rushing_tds") + pl.col("receiving_tds")

    return season.with_columns(
        pl.when(touches > 0).then(total_td / touches).otherwise(None).alias("td_rate"),
        safe("rushing_yards", "carries").alias("yards_per_carry"),
        safe("receiving_yards", "targets").alias("yards_per_target"),
        safe("receptions", "targets").alias("catch_rate"),
        (pl.col("carries") + pl.col("targets")).alias("opportunities"),
    )


def build_panel(
    seasons: list[int],
    scoring: ScoringConfig,
    lags: int = 2,
    min_games_prior: int = 1,
    refresh: bool = False,
) -> pl.DataFrame:
    """Supervised panel: features from seasons N-1..N-lags, target from N.

    Rookies are dropped (no prior NFL season). They need a separate model
    built on draft capital and college production — a genuinely harder
    problem, deliberately out of scope for v1.
    """
    season = build_season_table(seasons, scoring, refresh=refresh)

    target = season.select(
        "player_id", "season", "position", "player_display_name", "team",
        pl.col("fantasy_points_total").alias("y_points"),
        pl.col("fantasy_points_per_game").alias("y_ppg"),
        pl.col("games_played").alias("y_games"),
        "age", "years_exp",
    )

    feature_cols = [
        c for c in season.columns
        if c not in {"player_id", "season", "position", "player_display_name",
                     "team", "position_group", "age", "years_exp"}
    ]

    panel = target
    for lag in range(1, lags + 1):
        prior = season.select(
            ["player_id", "season"] + feature_cols
        ).with_columns((pl.col("season") + lag).alias("season")).rename(
            {c: f"{c}_lag{lag}" for c in feature_cols}
        )
        panel = panel.join(prior, on=["player_id", "season"], how="left")

    panel = panel.filter(pl.col("games_played_lag1") >= min_games_prior)

    # Persistence baseline lives in the panel so every downstream model is
    # scored against it automatically.
    return panel.with_columns(
        pl.col("fantasy_points_total_lag1").alias("baseline_persistence")
    ).sort(["season", "y_points"], descending=[False, True])


def stickiness_report(
    season: pl.DataFrame,
    metrics: list[str] | None = None,
    min_games: int = 8,
) -> pl.DataFrame:
    """Year-over-year correlation for each metric — the project's thesis.

    Opportunity metrics should show high correlation; efficiency metrics
    should show much lower. This table is the argument for the whole feature
    strategy, and it belongs at the top of the README.
    """
    metrics = metrics or [
        "target_share_mean", "air_yards_share_mean", "wopr_mean", "snap_pct_mean",
        "targets", "carries", "opportunities",
        "fantasy_points_per_game", "td_rate", "yards_per_carry",
        "yards_per_target", "catch_rate",
    ]
    metrics = [m for m in metrics if m in season.columns]

    cur = season.filter(pl.col("games_played") >= min_games)
    nxt = cur.select(["player_id", "season"] + metrics).with_columns(
        (pl.col("season") - 1).alias("season")
    ).rename({m: f"{m}_next" for m in metrics})

    paired = cur.join(nxt, on=["player_id", "season"], how="inner")

    rows = []
    for m in metrics:
        sub = paired.select(m, f"{m}_next").drop_nulls()
        if sub.height < 30:
            continue
        rows.append({
            "metric": m,
            "yoy_correlation": sub.select(pl.corr(m, f"{m}_next")).item(),
            "n_pairs": sub.height,
        })

    return pl.DataFrame(rows).sort("yoy_correlation", descending=True)
