"""Tests for the composite availability x production model.

The Monte Carlo sampler has several ways to be silently wrong — crossed
quantiles breaking inverse-transform sampling, games escaping [0, 17],
the copula not actually coupling anything — and every one of them
produces plausible-looking numbers rather than an error. Hence these.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gridiron.calibration import (
    MAX_GAMES,
    QUANTILE_GRID,
    CompositeProjector,
    QuantileGrid,
)
from gridiron.config import ScoringConfig
from gridiron.features import build_panel
from gridiron.offseason import attach_offseason_features

SEASONS = list(range(2014, 2026))

# These tests assert *structural* correctness — monotonic quantiles,
# respected bounds, a copula that actually couples — none of which needs a
# production-sized fit. A coarse grid with short boosting keeps the suite
# usable; scripts/eval_conformal.py is where the real numbers come from.
FAST_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
FAST_ROUNDS = 60


def fast_grid(target: str, **kw) -> QuantileGrid:
    return QuantileGrid(target, levels=FAST_LEVELS, num_boost_round=FAST_ROUNDS, **kw)


def fast_composite(**kw) -> CompositeProjector:
    return CompositeProjector(levels=FAST_LEVELS, num_boost_round=FAST_ROUNDS, **kw)

# Each test here fits a grid of 13 LightGBM quantile models per target, so
# the module dominates suite runtime. CI runs `pytest -m "not slow"` on
# every commit and the full suite nightly.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    base = build_panel(SEASONS, ScoringConfig.half_ppr()).filter(
        pl.col("games_played_lag1") >= 6
    )
    return attach_offseason_features(base, SEASONS)


@pytest.fixture(scope="module")
def split(panel):
    return (
        panel.filter(pl.col("season") < 2024),
        panel.filter(pl.col("season") == 2024),
    )


class TestQuantileGrid:
    def test_quantiles_are_monotonic(self, split):
        train, test = split
        g = fast_grid("y_games", lower=0.0, upper=MAX_GAMES).fit(train)
        q = g.quantile_matrix(test)
        assert (np.diff(q, axis=1) >= -1e-9).all(), "quantiles cross"

    def test_bounds_are_respected(self, split):
        """Games played outside [0, 17] would silently corrupt every total."""
        train, test = split
        g = fast_grid("y_games", lower=0.0, upper=MAX_GAMES).fit(train)
        q = g.quantile_matrix(test)
        assert q.min() >= 0.0
        assert q.max() <= MAX_GAMES

    def test_pit_is_roughly_uniform(self, split):
        """A calibrated model puts actuals uniformly across its quantiles.

        This is deliberately loose — it catches a badly broken PIT, not
        mild miscalibration, which is what the coverage metric is for.
        """
        train, test = split
        g = fast_grid("y_ppg", lower=0.0).fit(train)
        pit = g.pit(test)
        pit = pit[np.isfinite(pit)]
        assert len(pit) > 100
        assert 0.30 < pit.mean() < 0.70

    def test_inverse_transform_recovers_the_median(self, split):
        """Sampling at u=0.5 must return the predicted median."""
        train, test = split
        g = fast_grid("y_ppg", lower=0.0).fit(train)
        q = g.quantile_matrix(test)
        median_idx = FAST_LEVELS.index(0.50)
        u = np.full((test.height, 1), 0.5)
        assert np.allclose(g.sample(test, u)[:, 0], q[:, median_idx], atol=1e-6)


class TestCompositeProjector:
    def test_copula_finds_positive_dependence(self, split):
        """Availability and production are positively related — good
        players both stay on the field and score more. A fitted rho near
        zero means the copula estimation silently failed."""
        train, _ = split
        cp = fast_composite(dependence="copula").fit(train)
        assert 0.15 < cp.rho < 0.95, f"implausible copula rho: {cp.rho}"

    def test_independent_mode_really_is_independent(self, split):
        train, _ = split
        cp = fast_composite(dependence="independent").fit(train)
        assert cp.rho == 0.0

    def test_quantiles_are_ordered_and_non_negative(self, split):
        train, test = split
        f = fast_composite(n_samples=500).fit(train).predict_frame(test)
        p10, p25, p50, p75, p90 = (
            f["p10"].to_numpy(), f["p25"].to_numpy(), f["p50"].to_numpy(),
            f["p75"].to_numpy(), f["p90"].to_numpy(),
        )
        assert (p10 >= 0).all()
        assert (p10 <= p25).all() and (p25 <= p50).all()
        assert (p50 <= p75).all() and (p75 <= p90).all()

    def test_totals_cannot_exceed_max_games_times_max_ppg(self, split):
        """Composition must respect the bound on its factors."""
        train, test = split
        cp = fast_composite(n_samples=500).fit(train)
        totals = cp.sample_totals(test)
        ppg_max = cp._ppg.quantile_matrix(test).max()
        assert totals.max() <= MAX_GAMES * ppg_max + 1e-6

    def test_sampling_is_reproducible(self, split):
        train, test = split
        cp = fast_composite(n_samples=300, seed=7).fit(train)
        assert np.allclose(cp.sample_totals(test), cp.sample_totals(test))

    def test_dependence_widens_the_upper_tail(self, split):
        """The point of the copula.

        Independent sampling pairs high production with low availability
        as often as not, which shaves the top of the distribution. Positive
        dependence should push p90 up.
        """
        train, test = split
        indep = fast_composite(dependence="independent", n_samples=1500).fit(train)
        cop = fast_composite(dependence="copula", n_samples=1500).fit(train)
        p90_indep = indep.predict_frame(test)["p90"].mean()
        p90_cop = cop.predict_frame(test)["p90"].mean()
        assert p90_cop > p90_indep
