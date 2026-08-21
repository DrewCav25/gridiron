"""Projections for an upcoming season.

Everything else in this project is backtesting: predict a season that has
already happened, then check. This module does the forward version —
project a season that has not been played yet, which is what you actually
need on draft day.

The mechanics are the same as ``features.build_panel`` with one difference:
there is no season-N row to build the target frame from, because no games
have been played. The player list comes from preseason rosters instead, and
the target columns are left null.

Depth chart features are **off by default here**. Week-1 depth charts post
just before the season, often after August fantasy drafts, and the schema
changes between seasons. Finding #3 reports both variants; the strict
draft-day one is the honest choice for a forward projection.
"""

from __future__ import annotations

import polars as pl

from . import data as D
from .config import ScoringConfig
from .features import build_season_table
from .models import GBMProjector
from .offseason import attach_offseason_features

HISTORY_START = 2012


def build_future_panel(
    target_season: int,
    scoring: ScoringConfig,
    history_start: int = HISTORY_START,
    include_depth_chart: bool = False,
    min_games_prior: int = 1,
    refresh: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(train_panel, predict_panel)`` for ``target_season``.

    ``train_panel`` is every completed season with known outcomes.
    ``predict_panel`` is the upcoming season with features but no target.
    """
    completed = list(range(history_start, target_season))
    table = build_season_table(completed, scoring, refresh=refresh)

    feature_cols = [
        c for c in table.columns
        if c not in {"player_id", "season", "position", "player_display_name",
                     "team", "position_group", "age", "years_exp"}
    ]

    # --- historical target frame (same as features.build_panel) -----------
    hist_target = table.select(
        "player_id", "season", "position", "player_display_name", "team",
        pl.col("fantasy_points_total").alias("y_points"),
        pl.col("fantasy_points_per_game").alias("y_ppg"),
        pl.col("games_played").alias("y_games"),
        "age", "years_exp",
    )

    # --- upcoming-season target frame from preseason rosters --------------
    roster = (
        D.load_rosters([target_season], refresh=refresh)
        .filter(pl.col("position").is_in(list(D.FANTASY_POSITIONS)))
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.lit(target_season).cast(pl.Int32).alias("season"),
            "position",
            pl.col("full_name").alias("player_display_name"),
            "team",
            "birth_date",
            "years_exp",
        )
        .drop_nulls("player_id")
        .unique(subset=["player_id"], keep="first")
    )
    if roster.schema["birth_date"] == pl.String:
        roster = roster.with_columns(
            pl.col("birth_date").str.to_date("%Y-%m-%d", strict=False)
        )

    future_target = roster.with_columns(
        (pl.lit(target_season) - pl.col("birth_date").dt.year())
        .cast(pl.Float64).alias("age"),
        pl.col("years_exp").cast(pl.Float64),
        pl.lit(None, dtype=pl.Float64).alias("y_points"),
        pl.lit(None, dtype=pl.Float64).alias("y_ppg"),
        pl.lit(None, dtype=pl.UInt32).alias("y_games"),
    ).drop("birth_date").select(hist_target.columns)

    combined = pl.concat(
        [hist_target.with_columns(pl.col("season").cast(pl.Int32)), future_target],
        how="vertical_relaxed",
    )

    # --- lags apply identically to both ----------------------------------
    for lag in (1, 2):
        prior = (
            table.select(["player_id", "season"] + feature_cols)
            .with_columns((pl.col("season") + lag).cast(pl.Int32).alias("season"))
            .rename({c: f"{c}_lag{lag}" for c in feature_cols})
        )
        combined = combined.join(prior, on=["player_id", "season"], how="left")

    combined = combined.filter(pl.col("games_played_lag1") >= min_games_prior)
    combined = combined.with_columns(
        pl.col("fantasy_points_total_lag1").alias("baseline_persistence")
    )

    all_seasons = completed + [target_season]
    combined = attach_offseason_features(
        combined, all_seasons,
        include_depth_chart=include_depth_chart, refresh=refresh,
    )

    train = combined.filter(pl.col("season") < target_season).drop_nulls("y_points")
    predict = combined.filter(pl.col("season") == target_season)
    return train, predict


def project_season(
    target_season: int,
    scoring: ScoringConfig | None = None,
    history_start: int = HISTORY_START,
    include_depth_chart: bool = False,
    refresh: bool = False,
) -> pl.DataFrame:
    """Point projections for every rostered skill player in an upcoming season."""
    scoring = scoring or ScoringConfig.half_ppr()
    train, predict = build_future_panel(
        target_season, scoring,
        history_start=history_start,
        include_depth_chart=include_depth_chart,
        refresh=refresh,
    )
    if predict.is_empty():
        raise ValueError(
            f"no rostered players with prior-season data for {target_season}"
        )

    model = GBMProjector().fit(train)
    preds = model.predict(predict)

    return predict.select(
        "player_id", "player_display_name", "position", "team", "season",
        pl.col("fantasy_points_total_lag1").alias("last_season_points"),
        pl.col("age"), pl.col("changed_team"), pl.col("new_head_coach"),
        pl.col("draft_best_pick_at_pos"),
    ).with_columns(
        # The GBM is unbounded regression, so it extrapolates slightly
        # below zero for deep-bench players with almost no prior
        # production. A negative season projection is meaningless — clamp
        # rather than ship a ranking with nonsense at the bottom.
        pl.Series("projected_points", preds).clip(lower_bound=0.0).round(1)
    ).sort("projected_points", descending=True)
