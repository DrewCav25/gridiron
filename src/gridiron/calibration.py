"""Composite projections: availability × per-game production.

Finding #4 in the README: predicting season-total quantiles directly gives
p10-p90 intervals that cover only 62% of outcomes instead of 80%. The
diagnosis was structural rather than a tuning problem. Season totals are

    total = games_played x points_per_game

and *most* of the variance lives in the first factor. 31% of fantasy-
relevant players appear in 8 or fewer games; availability has a standard
deviation of about 5 games. A single model asked to predict season totals
has to smear that bimodal injury risk into one conditional distribution,
and it resolves the conflict by hedging toward the middle — which is
exactly the too-narrow interval the calibration check caught.

The fix is to model the two factors separately and compose them:

  1. a quantile grid over games played
  2. a quantile grid over points per game
  3. Monte Carlo sampling from both, multiplied

Step 3 has a subtlety worth doing properly. Availability and production
are *not* independent — good players both stay on the field and score
more, and the two are correlated at rho ~ 0.57 in the raw data. Sampling
them independently would badly understate the upper tail. So the sampler
uses a Gaussian copula whose parameter is estimated from out-of-sample
PIT values, which measures the dependence remaining *after* the features
have explained what they can.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.stats import norm, spearmanr

import lightgbm as lgb

from .models import _design_matrix, feature_columns

# Fine enough to capture shape, coarse enough to stay fast. The 0.02/0.98
# endpoints matter: with a 0.05/0.95 grid, clamping would truncate exactly
# the tails this module exists to widen.
QUANTILE_GRID = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.90, 0.95, 0.98)

MAX_GAMES = 17.0


@dataclass
class QuantileGrid:
    """A full predictive distribution for one target, as a quantile function."""

    target: str
    levels: tuple[float, ...] = QUANTILE_GRID
    num_boost_round: int = 250
    lower: float | None = None
    upper: float | None = None
    params: dict = field(default_factory=lambda: {
        "objective": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 4,
        "verbosity": -1,
    })
    _cols: list[str] = field(default_factory=list, init=False)
    _boosters: dict = field(default_factory=dict, init=False)

    def fit(self, train: pl.DataFrame) -> "QuantileGrid":
        self._cols = feature_columns(train)
        X = _design_matrix(train, self._cols)
        y = train[self.target].to_numpy().astype(np.float64)
        ds = lgb.Dataset(X, label=y)
        for a in self.levels:
            self._boosters[a] = lgb.train(
                dict(self.params, alpha=a), ds, num_boost_round=self.num_boost_round
            )
        return self

    def quantile_matrix(self, test: pl.DataFrame) -> np.ndarray:
        """(n_rows, n_levels) predicted quantiles, sorted and clipped.

        Independently fit quantile models can cross — p60 below p50 — on
        small samples. Sorting each row repairs monotonicity, which is
        required for the inverse-transform sampling below to be valid.
        """
        X = _design_matrix(test, self._cols)
        q = np.column_stack([self._boosters[a].predict(X) for a in self.levels])
        q = np.sort(q, axis=1)
        if self.lower is not None:
            q = np.maximum(q, self.lower)
        if self.upper is not None:
            q = np.minimum(q, self.upper)
        return q

    def sample(self, test: pl.DataFrame, u: np.ndarray) -> np.ndarray:
        """Inverse-transform sample. ``u`` is (n_rows, n_samples) uniforms."""
        q = self.quantile_matrix(test)
        levels = np.asarray(self.levels)
        out = np.empty_like(u, dtype=np.float64)
        for i in range(q.shape[0]):
            out[i] = np.interp(u[i], levels, q[i])
        return out

    def pit(self, df: pl.DataFrame) -> np.ndarray:
        """Probability integral transform: where each actual falls in its
        own predicted distribution. Uniform if the model is well calibrated.
        """
        q = self.quantile_matrix(df)
        y = df[self.target].to_numpy().astype(np.float64)
        levels = np.asarray(self.levels)
        return np.array([
            float(np.interp(y[i], q[i], levels)) for i in range(q.shape[0])
        ])


@dataclass
class CompositeProjector:
    """Season totals as availability x production, sampled jointly.

    ``dependence`` selects how the two factors are coupled:
      - ``"copula"``   estimate rank correlation from held-out PIT (default)
      - ``"independent"``  assume none (understates the upper tail)
      - ``"comonotonic"``  perfect coupling (overstates it)

    The three are kept switchable because comparing them is the experiment
    that shows the dependence assumption matters, rather than asserting it.
    """

    n_samples: int = 4000
    dependence: str = "copula"
    seed: int = 0
    # Overridable so tests can assert structural correctness on a coarse
    # grid without paying for a production-sized fit.
    levels: tuple[float, ...] = QUANTILE_GRID
    num_boost_round: int = 250
    _games: QuantileGrid = field(default=None, init=False)
    _ppg: QuantileGrid = field(default=None, init=False)
    _rho: float = field(default=0.0, init=False)

    def fit(self, train: pl.DataFrame) -> "CompositeProjector":
        # Hold out the most recent training season to estimate dependence
        # out of sample. Using in-sample PIT would measure the models'
        # overfitting rather than the true residual dependence.
        seasons = sorted(train["season"].unique().to_list())
        if len(seasons) >= 4:
            inner_cut = seasons[-1]
            inner_train = train.filter(pl.col("season") < inner_cut)
            inner_valid = train.filter(pl.col("season") == inner_cut)
        else:
            inner_train, inner_valid = train, train

        def grid(target: str, **kw) -> QuantileGrid:
            return QuantileGrid(
                target, levels=self.levels,
                num_boost_round=self.num_boost_round, **kw
            )

        self._games = grid("y_games", lower=0.0, upper=MAX_GAMES)
        self._ppg = grid("y_ppg", lower=0.0)

        if self.dependence == "copula":
            g_inner = grid("y_games", lower=0.0, upper=MAX_GAMES).fit(inner_train)
            p_inner = grid("y_ppg", lower=0.0).fit(inner_train)
            pit_g = g_inner.pit(inner_valid)
            pit_p = p_inner.pit(inner_valid)
            ok = np.isfinite(pit_g) & np.isfinite(pit_p)
            rho_s = spearmanr(pit_g[ok], pit_p[ok]).statistic if ok.sum() > 30 else 0.0
            # Spearman rho -> Gaussian copula parameter.
            self._rho = float(np.clip(2.0 * np.sin(np.pi * rho_s / 6.0), -0.95, 0.95))
        elif self.dependence == "comonotonic":
            self._rho = 0.95
        else:
            self._rho = 0.0

        self._games.fit(train)
        self._ppg.fit(train)
        return self

    def _uniforms(self, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        z1 = rng.standard_normal((n_rows, self.n_samples))
        z2 = rng.standard_normal((n_rows, self.n_samples))
        # Gaussian copula: correlate, then map back to uniforms.
        z2 = self._rho * z1 + np.sqrt(max(1.0 - self._rho**2, 0.0)) * z2
        return norm.cdf(z1), norm.cdf(z2)

    def sample_totals(self, test: pl.DataFrame) -> np.ndarray:
        """(n_rows, n_samples) simulated season totals."""
        u_g, u_p = self._uniforms(test.height)
        games = np.clip(self._games.sample(test, u_g), 0.0, MAX_GAMES)
        ppg = np.maximum(self._ppg.sample(test, u_p), 0.0)
        return games * ppg

    def predict_frame(
        self, test: pl.DataFrame, quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    ) -> pl.DataFrame:
        totals = self.sample_totals(test)
        qs = np.quantile(totals, quantiles, axis=1)
        cols = [
            pl.Series(f"p{int(q * 100):02d}", qs[i]) for i, q in enumerate(quantiles)
        ]
        cols.append(pl.Series("mean_total", totals.mean(axis=1)))
        return test.with_columns(cols)

    @property
    def rho(self) -> float:
        """Fitted Gaussian copula parameter (0 = independent)."""
        return self._rho


@dataclass
class ConformalCalibrator:
    """Split-conformal correction for interval coverage.

    Composing availability and production narrows the coverage gap but does
    not close it, because both *marginals* are individually overconfident —
    gradient boosted quantile regression tends to be, since predictions are
    averaged within leaves. No amount of restructuring the composition
    fixes an error in the pieces.

    Conformalized quantile regression (Romano, Patterson & Candes, 2019)
    fixes it empirically instead of parametrically. Fit on one slice of
    data, measure on a held-out slice how far outside the predicted
    interval reality actually fell, and widen by that amount. The result
    has a finite-sample coverage guarantee under exchangeability and needs
    no assumption about the shape of the error.

    The exchangeability assumption deserves a caveat here: NFL seasons are
    not exchangeable (rule changes, a 17th game added in 2021), so the
    guarantee is approximate rather than exact. It still works well in
    practice, and it is reported as measured rather than assumed.

    Usage — the calibration season must not be in the model's training set:

        model = CompositeProjector().fit(train_before_calib)
        cal = ConformalCalibrator().fit(model, calib_season_df)
        out = cal.predict_frame(model, test_df)
    """

    target: str = "y_points"
    _offsets: dict[tuple[float, float], float] = field(default_factory=dict, init=False)

    def fit(self, model, calib: pl.DataFrame, pairs=((0.10, 0.90), (0.25, 0.75))):
        """Measure how far reality escaped each nominal interval."""
        y = calib[self.target].to_numpy().astype(np.float64)
        for lo_q, hi_q in pairs:
            f = model.predict_frame(calib, quantiles=(lo_q, hi_q))
            lo = f[f"p{int(lo_q * 100):02d}"].to_numpy()
            hi = f[f"p{int(hi_q * 100):02d}"].to_numpy()
            # Conformity score: signed distance outside the interval.
            scores = np.maximum(lo - y, y - hi)
            level = hi_q - lo_q
            n = len(scores)
            # Finite-sample corrected quantile of the conformity scores.
            k = min(int(np.ceil((n + 1) * level)) - 1, n - 1)
            self._offsets[(lo_q, hi_q)] = float(np.sort(scores)[max(k, 0)])
        return self

    def predict_frame(self, model, test: pl.DataFrame, pairs=((0.10, 0.90), (0.25, 0.75))):
        """Apply the learned widening to a model's intervals."""
        out = test
        for lo_q, hi_q in pairs:
            f = model.predict_frame(test, quantiles=(lo_q, hi_q, 0.50))
            d = self._offsets.get((lo_q, hi_q), 0.0)
            out = out.with_columns(
                pl.Series(
                    f"p{int(lo_q * 100):02d}",
                    np.maximum(f[f"p{int(lo_q * 100):02d}"].to_numpy() - d, 0.0),
                ),
                pl.Series(
                    f"p{int(hi_q * 100):02d}",
                    f[f"p{int(hi_q * 100):02d}"].to_numpy() + d,
                ),
            )
            if "p50" not in out.columns:
                out = out.with_columns(pl.Series("p50", f["p50"].to_numpy()))
        return out

    @property
    def offsets(self) -> dict:
        return dict(self._offsets)

    def widen_samples(self, samples: np.ndarray, pair=(0.10, 0.90)) -> np.ndarray:
        """Apply the conformal correction to a full sample matrix.

        The calibrator learns an additive widening for *intervals*, but the
        draft simulator consumes *samples*. Rescaling each player's draws
        about their own median by the factor that widens the p10-p90 band
        by 2d carries the correction over, so downstream sampling inherits
        the calibrated spread instead of the overconfident raw one.

        Approximate rather than exact — it assumes the correction scales
        the distribution rather than reshaping it — but it beats feeding
        the optimizer intervals known to be 18 points too narrow.
        """
        d = self._offsets.get(pair)
        if not d:
            return samples
        median = np.median(samples, axis=1, keepdims=True)
        lo, hi = np.percentile(samples, [pair[0] * 100, pair[1] * 100], axis=1)
        width = np.maximum(hi - lo, 1e-6)[:, None]
        scale = (width + 2.0 * d) / width
        return np.maximum(median + (samples - median) * scale, 0.0)
