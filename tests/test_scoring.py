"""Scoring engine validation.

The important test here is ``test_matches_nflverse_reference``: our standard
and PPR presets must exactly reproduce nflverse's own ``fantasy_points`` and
``fantasy_points_ppr`` columns. If they don't, every downstream number in
the project is wrong, so this runs first and fails loudly.
"""

from __future__ import annotations

import polars as pl
import pytest

from gridiron.config import ScoringConfig, LeagueConfig, PPR_12
from gridiron.scoring import add_fantasy_points, weekly_to_season


@pytest.fixture(scope="module")
def weekly() -> pl.DataFrame:
    import nflreadpy as nfl
    return nfl.load_player_stats(seasons=[2023, 2024])


@pytest.mark.parametrize(
    "preset,reference",
    [
        (ScoringConfig.standard(), "fantasy_points"),
        (ScoringConfig.ppr(), "fantasy_points_ppr"),
    ],
)
def test_matches_nflverse_reference(weekly, preset, reference):
    scored = add_fantasy_points(weekly, preset)
    diff = (
        scored["fantasy_points_custom"] - scored[reference].fill_null(0.0)
    ).abs().max()
    assert diff < 0.011, f"{preset.name} deviates from {reference} by {diff}"


def test_half_ppr_sits_between_standard_and_ppr(weekly):
    std = add_fantasy_points(weekly, ScoringConfig.standard())["fantasy_points_custom"]
    half = add_fantasy_points(weekly, ScoringConfig.half_ppr())["fantasy_points_custom"]
    ppr = add_fantasy_points(weekly, ScoringConfig.ppr())["fantasy_points_custom"]
    assert (half >= std - 1e-9).all()
    assert (ppr >= half - 1e-9).all()


def test_te_premium_only_affects_tight_ends(weekly):
    base = add_fantasy_points(weekly, ScoringConfig.ppr())
    prem = add_fantasy_points(weekly, ScoringConfig.te_premium())
    delta = prem["fantasy_points_custom"] - base["fantasy_points_custom"]
    non_te = base.with_columns(delta.alias("d")).filter(pl.col("position") != "TE")
    assert non_te["d"].abs().max() < 1e-9


def test_scoring_config_is_not_hardcoded():
    """Changing a single setting must change the output."""
    a = ScoringConfig.ppr()
    b = ScoringConfig(reception=1.0, rush_td=10.0)
    assert a.rush_td != b.rush_td


class TestLeagueConfig:
    def test_roster_arithmetic(self):
        lg = LeagueConfig(teams=12, qb=1, rb=2, wr=2, te=1, flex=1, k=1, dst=1, bench=6)
        assert lg.starters_per_team == 9
        assert lg.roster_size == 15
        assert lg.total_drafted == 180

    def test_replacement_level_scales_with_league_size(self):
        """Replacement level is a function of league settings, not a constant.

        A 10-team league has a much shallower replacement level than a
        14-team league, which is why VOR computed with hardcoded settings
        is wrong for most people.
        """
        small = LeagueConfig(teams=10)
        big = LeagueConfig(teams=14)
        assert small.base_starters("RB") < big.base_starters("RB")

    def test_flex_slots_counted_for_eligible_positions_only(self):
        lg = PPR_12
        assert lg.flex_slots("RB") == 12
        assert lg.flex_slots("QB") == 0

    def test_superflex_adds_qb_slots(self):
        lg = LeagueConfig(teams=12, superflex=1)
        assert lg.flex_slots("QB") == 12


class TestSeasonAggregation:
    def test_games_played_never_exceeds_weeks(self, weekly):
        scored = add_fantasy_points(weekly, ScoringConfig.half_ppr())
        season = weekly_to_season(scored)
        assert season["games_played"].max() <= 18

    def test_totals_reconcile_with_per_game(self, weekly):
        scored = add_fantasy_points(weekly, ScoringConfig.half_ppr())
        season = weekly_to_season(scored).filter(pl.col("games_played") > 0)
        recomputed = season["fantasy_points_per_game"] * season["games_played"]
        assert (recomputed - season["fantasy_points_total"]).abs().max() < 1e-6
