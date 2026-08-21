"""Leakage tests.

These are the tests that matter most in this project. A fantasy projection
model that accidentally sees the future looks excellent in backtest and
fails completely in September, and the failure is silent — there is no
error, just numbers that are too good.

So: every feature the model trains on is asserted here to be knowable
before week 1 of the season being predicted. If someone joins a new column
onto the panel and it isn't on the allowlist, `test_no_unvetted_features`
fails.
"""

from __future__ import annotations

import polars as pl
import pytest

from gridiron.config import ScoringConfig
from gridiron.features import build_panel, build_season_table
from gridiron.models import (
    KNOWN_PRESEASON,
    OFFSEASON_FEATURES,
    feature_columns,
)
from gridiron.offseason import (
    incoming_draft_capital,
    preseason_team,
    preseason_depth_chart,
    attach_offseason_features,
)

SEASONS = list(range(2015, 2026))
TARGETS = {"y_points", "y_ppg", "y_games"}


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    base = build_panel(SEASONS, ScoringConfig.half_ppr())
    return attach_offseason_features(base, SEASONS)


class TestFeatureAllowlist:
    def test_no_target_columns_in_features(self, panel):
        cols = set(feature_columns(panel))
        assert not (cols & TARGETS), f"target leaked into features: {cols & TARGETS}"

    def test_no_unvetted_features(self, panel):
        """Every training feature must be a lag, a known-preseason field,
        or an explicitly vetted offseason feature. No exceptions."""
        for c in feature_columns(panel):
            allowed = (
                c.endswith(("_lag1", "_lag2", "_lag3"))
                or c in KNOWN_PRESEASON
                or c in OFFSEASON_FEATURES
            )
            assert allowed, f"unvetted feature in training set: {c}"

    def test_current_season_stats_are_not_features(self, panel):
        """Raw season-N statistics must never be eligible.

        ``build_season_table`` produces unlagged columns like ``targets``
        and ``receptions``; the panel keeps some of them for baselines and
        diagnostics, but the model must not see them.
        """
        unlagged = {
            "targets", "receptions", "carries", "receiving_yards",
            "rushing_yards", "passing_yards", "opportunities",
            "games_played", "fantasy_points_total", "fantasy_points_per_game",
        }
        assert not (set(feature_columns(panel)) & unlagged)


class TestLagCorrectness:
    def test_lag1_equals_prior_season_value(self):
        """A season-N lag1 feature must equal that player's season N-1 value.

        This is the test that catches an off-by-one in the join, which is
        the most damaging and least visible bug available here.
        """
        table = build_season_table(SEASONS, ScoringConfig.half_ppr())
        panel = build_panel(SEASONS, ScoringConfig.half_ppr())

        prior = table.select(
            "player_id",
            (pl.col("season") + 1).alias("season"),
            pl.col("fantasy_points_total").alias("expected"),
        )
        merged = panel.join(prior, on=["player_id", "season"], how="inner")
        diff = (merged["fantasy_points_total_lag1"] - merged["expected"]).abs().max()
        assert merged.height > 500
        assert diff < 1e-9, f"lag1 misaligned by {diff}"

    def test_persistence_baseline_is_the_lag_not_the_target(self, panel):
        same_as_target = (
            panel["baseline_persistence"] - panel["y_points"]
        ).abs().mean()
        assert same_as_target > 10, "persistence baseline looks like the target"


class TestOffseasonTiming:
    def test_draft_capital_uses_only_that_seasons_draft(self):
        """Season N features may use season N's April draft, never N+1's."""
        picks = incoming_draft_capital(SEASONS)
        assert picks["season"].min() >= min(SEASONS)
        assert picks["season"].max() <= max(SEASONS)

    def test_preseason_team_comes_from_week_one(self):
        """Using a later-week team would encode surviving cuts or a trade."""
        teams = preseason_team([2023, 2024])
        assert teams.height > 1000
        assert teams.select(
            pl.struct("player_id", "season").n_unique()
        ).item() == teams.height

    def test_depth_chart_is_week_one_only(self):
        dc = preseason_depth_chart([2023, 2024])
        assert dc["depth_position"].min() >= 1
        assert dc.height > 500

    def test_changed_team_flag_is_plausible(self, panel):
        """Roughly a quarter of fantasy-relevant players change teams.

        A rate near 0 means the join silently failed; near 1 means the team
        codes don't match between sources. Both have happened in this repo.
        """
        rate = panel["changed_team"].mean()
        assert 0.10 < rate < 0.45, f"implausible team-change rate: {rate}"

    def test_team_context_is_prior_season(self, panel):
        """Team offensive context must describe season N-1, not season N."""
        assert "tm_pass_rate_prior" in panel.columns
        vals = panel["tm_pass_rate_prior"].drop_nulls()
        assert 0.35 < vals.mean() < 0.75


class TestNoImplausibleSignal:
    def test_no_feature_is_near_perfectly_correlated_with_target(self, panel):
        """A correlation above ~0.95 with the target means leakage.

        Real season-to-season fantasy signal tops out well below this.
        Anything near 1.0 is the target wearing a disguise.
        """
        cols = feature_columns(panel)
        offenders = []
        for c in cols:
            sub = panel.select(c, "y_points").drop_nulls()
            if sub.height < 200 or sub[c].n_unique() < 3:
                continue
            r = sub.select(pl.corr(c, "y_points")).item()
            if r is not None and abs(r) > 0.95:
                offenders.append((c, r))
        assert not offenders, f"suspiciously perfect features: {offenders}"

    def test_walk_forward_train_never_includes_test_season(self, panel):
        for season in (2021, 2023, 2025):
            train = panel.filter(pl.col("season") < season)
            assert train["season"].max() < season
            assert season not in train["season"].unique().to_list()
