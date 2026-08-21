"""Offseason features — the information lagged box scores cannot see.

Finding #2 in the README: a gradient boosted model fed only prior-season
statistics does not beat naive persistence on rank correlation. The
diagnosis was that the information which actually moves projections year
over year happens *between* seasons — free agency, the draft, coaching
changes, depth chart competition — and none of it appears in last year's
stat line. A model has no way to know a team just spent a second-round
pick on a running back.

This module adds that information. Every feature here is knowable before
week 1 of the season being predicted; ``tests/test_leakage.py`` enforces it.

Timing caveat, stated plainly: the NFL draft (April) and free agency
(March) are settled well before fantasy drafts in late August, so those
features are honestly available. Week-1 depth charts are published just
before the season and may post *after* some fantasy drafts, so
``include_depth_chart`` defaults to True for modelling accuracy but can be
switched off to simulate strict draft-day information.
"""

from __future__ import annotations

import polars as pl

from . import data as D

# Pro-Football-Reference team codes (used by load_draft_picks) mapped to
# nflverse codes (used by rosters and team stats). Relocations collapse to
# the current franchise code, which is what the roster tables use.
PFR_TO_NFLVERSE = {
    "GNB": "GB", "KAN": "KC", "NOR": "NO", "NWE": "NE", "SFO": "SF",
    "TAM": "TB", "LVR": "LV", "OAK": "LV", "RAI": "LV",
    "SDG": "LAC", "STL": "LA", "LAR": "LA", "RAM": "LA",
    "CRD": "ARI", "RAV": "BAL", "HTX": "HOU", "OTI": "TEN", "CLT": "IND",
    "JAC": "JAX",
}


def _normalize_team(col: str) -> pl.Expr:
    return pl.col(col).replace(PFR_TO_NFLVERSE)


def preseason_team(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """The team each player is rostered on entering the season.

    Uses the week-1 roster, which is set before any game is played. Taking
    a mid-season or end-of-season team would leak: it would encode that the
    player survived roster cuts or was traded to a contender.
    """
    # rosters_weekly only covers completed seasons. For an upcoming season
    # (no games played yet) fall back to the preseason season-roster table,
    # which is exactly the right source for a draft-day projection.
    past = [s for s in seasons if s <= D.LAST_COMPLETED_SEASON]
    future = [s for s in seasons if s > D.LAST_COMPLETED_SEASON]

    frames = []
    if future:
        frames.append(
            D.load_rosters(future, refresh=refresh).select(
                pl.col("gsis_id").alias("player_id"),
                "season",
                pl.col("team").alias("team_current"),
            ).drop_nulls("player_id")
        )
    if not past:
        return pl.concat(frames).unique(subset=["player_id", "season"], keep="first")

    rosters = D.load_rosters_weekly(past, refresh=refresh)
    wk1 = (
        rosters.filter((pl.col("week") == 1) & (pl.col("game_type") == "REG"))
        .select(
            pl.col("gsis_id").alias("player_id"),
            "season",
            pl.col("team").alias("team_current"),
        )
        .drop_nulls("player_id")
    )
    frames.append(wk1)
    return pl.concat(frames).unique(subset=["player_id", "season"], keep="first")


def team_context(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Prior-season offensive profile for each team.

    Joined onto a player's *new* team, this is what tells the model a
    receiver moved from a run-heavy offense to a pass-heavy one — a change
    that lagged personal stats cannot express.
    """
    import nflreadpy as nfl

    # Context for season N is season N-1's team profile, so only completed
    # seasons are ever needed. Requesting an upcoming season here 404s —
    # those stats do not exist yet by definition.
    yrs = [s for s in sorted(set(seasons)) if s <= D.LAST_COMPLETED_SEASON]
    if not yrs:
        return pl.DataFrame()
    key = f"teamstats_{yrs[0]}_{yrs[-1]}"
    ts = D.cached(
        key,
        lambda: nfl.load_team_stats(seasons=yrs, summary_level="reg"),
        refresh,
    )

    games = pl.col("games").cast(pl.Float64)
    ctx = ts.select(
        "season",
        "team",
        (pl.col("attempts") / games).alias("tm_pass_att_pg"),
        (pl.col("carries") / games).alias("tm_rush_att_pg"),
        (
            pl.col("attempts")
            / (pl.col("attempts") + pl.col("carries")).cast(pl.Float64)
        ).alias("tm_pass_rate"),
        (pl.col("passing_epa") / games).alias("tm_pass_epa_pg"),
        (pl.col("rushing_epa") / games).alias("tm_rush_epa_pg"),
        (
            (pl.col("passing_tds") + pl.col("rushing_tds")) / games
        ).alias("tm_off_td_pg"),
        (pl.col("targets") / games).alias("tm_targets_pg"),
    )

    # Shift forward one season: context for predicting season N is the
    # team's season N-1 profile.
    return ctx.with_columns((pl.col("season") + 1).alias("season")).rename(
        {c: f"{c}_prior" for c in ctx.columns if c not in ("season", "team")}
    )


def incoming_draft_capital(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Draft capital a team spent at each position, in the season's own draft.

    This is the single most direct measure of incoming competition. A
    running back whose team just used pick 46 on another running back is a
    different asset than he was in December, and nothing in his own stat
    line reflects that.

    The NFL draft happens in April, so for predicting season N this uses
    season N's draft — correctly available, not leakage.
    """
    picks = D.load_draft_picks(refresh=refresh)
    yrs = sorted(set(seasons))

    picks = (
        picks.filter(pl.col("season").is_in(yrs))
        .filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
        .select(
            "season",
            _normalize_team("team").alias("team_current"),
            "position",
            "round",
            "pick",
        )
        .drop_nulls(["team_current", "position", "pick"])
    )

    return picks.group_by(["season", "team_current", "position"]).agg(
        pl.col("pick").min().alias("draft_best_pick_at_pos"),
        (pl.col("round") <= 3).sum().cast(pl.Int32).alias("draft_early_picks_at_pos"),
        pl.len().cast(pl.Int32).alias("draft_any_picks_at_pos"),
    )


def coaching_change(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Flag teams with a new head coach entering the season.

    A scheme change can move a player's usage far more than anything in his
    prior stat line. Derived from week-1 head coach in the schedule table
    versus the previous season's.
    """
    import nflreadpy as nfl

    sched = D.cached("schedules", lambda: nfl.load_schedules(), refresh)

    home = sched.select(
        "season", pl.col("home_team").alias("team"), pl.col("home_coach").alias("coach"), "week"
    )
    away = sched.select(
        "season", pl.col("away_team").alias("team"), pl.col("away_coach").alias("coach"), "week"
    )
    coaches = (
        pl.concat([home, away])
        .drop_nulls("coach")
        .sort("week")
        .group_by(["season", "team"])
        .agg(pl.col("coach").first().alias("coach"))
    )

    prior = coaches.with_columns((pl.col("season") + 1).alias("season")).rename(
        {"coach": "coach_prior"}
    )

    joined = coaches.join(prior, on=["season", "team"], how="left")
    return joined.select(
        "season",
        pl.col("team").alias("team_current"),
        pl.when(pl.col("coach_prior").is_null())
        .then(None)
        .otherwise((pl.col("coach") != pl.col("coach_prior")).cast(pl.Int32))
        .alias("new_head_coach"),
    )


def preseason_depth_chart(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Week-1 depth chart position (1 = starter).

    Published before the season opens, so it is not outcome leakage — but
    it can post after early fantasy drafts, which is why it is optional.
    """
    import nflreadpy as nfl

    yrs = sorted(set(seasons))
    key = f"depth_{yrs[0]}_{yrs[-1]}"
    dc = D.cached(
        key, lambda: nfl.load_depth_charts(seasons=yrs), refresh
    )

    return (
        dc.filter((pl.col("week") == 1) & (pl.col("game_type") == "REG"))
        .filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
        .select(
            pl.col("gsis_id").alias("player_id"),
            "season",
            pl.col("depth_team").cast(pl.Int32, strict=False).alias("depth_position"),
        )
        .drop_nulls("player_id")
        .group_by(["player_id", "season"])
        .agg(pl.col("depth_position").min().alias("depth_position"))
    )


def attach_offseason_features(
    panel: pl.DataFrame,
    seasons: list[int],
    include_depth_chart: bool = True,
    refresh: bool = False,
) -> pl.DataFrame:
    """Join every offseason signal onto the modelling panel."""
    out = panel

    # --- current team, and whether the player moved -----------------------
    teams = preseason_team(seasons, refresh=refresh)
    prior_teams = teams.with_columns((pl.col("season") + 1).alias("season")).rename(
        {"team_current": "team_prior"}
    )
    out = out.join(teams, on=["player_id", "season"], how="left")
    out = out.join(prior_teams, on=["player_id", "season"], how="left")
    out = out.with_columns(
        pl.when(pl.col("team_prior").is_null() | pl.col("team_current").is_null())
        .then(None)
        .otherwise((pl.col("team_current") != pl.col("team_prior")).cast(pl.Int32))
        .alias("changed_team")
    )

    # --- new team's prior offensive profile -------------------------------
    ctx = team_context(seasons, refresh=refresh)
    out = out.join(
        ctx.rename({"team": "team_current"}),
        on=["season", "team_current"],
        how="left",
    )

    # Delta in offensive environment: what the player's situation actually
    # changed by. For a player who stayed put this is zero by construction.
    ctx_old = ctx.rename(
        {"team": "team_prior"}
        | {c: f"{c}_oldteam" for c in ctx.columns if c.endswith("_prior")}
    )
    out = out.join(ctx_old, on=["season", "team_prior"], how="left")
    for c in ["tm_pass_rate_prior", "tm_pass_att_pg_prior", "tm_off_td_pg_prior"]:
        out = out.with_columns(
            (pl.col(c) - pl.col(f"{c}_oldteam")).alias(f"{c}_delta")
        )

    # --- incoming competition from the draft ------------------------------
    draft = incoming_draft_capital(seasons, refresh=refresh)
    out = out.join(draft, on=["season", "team_current", "position"], how="left")
    out = out.with_columns(
        pl.col("draft_early_picks_at_pos").fill_null(0),
        pl.col("draft_any_picks_at_pos").fill_null(0),
        # No pick spent at the position is best encoded as "very late",
        # not as a missing value.
        pl.col("draft_best_pick_at_pos").fill_null(300),
    )

    # --- coaching change --------------------------------------------------
    coach = coaching_change(seasons, refresh=refresh)
    out = out.join(coach, on=["season", "team_current"], how="left")

    # --- depth chart ------------------------------------------------------
    if include_depth_chart:
        dc = preseason_depth_chart(seasons, refresh=refresh)
        out = out.join(dc, on=["player_id", "season"], how="left")

    return out
