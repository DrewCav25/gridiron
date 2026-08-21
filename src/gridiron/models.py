"""Projection models.

Model 1 is a straightforward gradient-boosted point estimator.
Model 2 predicts **quantiles** — and that is the point of the project.

Public consensus projections (and tools built on them, like averaging
ESPN/CBS/NFL) give you a single number per player. Averaging sources
destroys exactly the variance information a drafter needs: you draft
differently in round 3 than round 12, and "safe 200 points" is a
completely different asset from "150 or 280 depending on the season".
Quantile models give you a shape the consensus structurally cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

import lightgbm as lgb


DEFAULT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

# Columns that are targets or identifiers, never features.
_EXCLUDE = {
    "player_id", "player_display_name", "team", "season",
    "y_points", "y_ppg", "y_games", "baseline_persistence",
}


# Offseason features (see offseason.py). These are contemporaneous with the
# season being predicted but settled before week 1, so they are legitimate.
# Anything added here must be covered by tests/test_leakage.py.
OFFSEASON_FEATURES = {
    "changed_team",
    "new_head_coach",
    "depth_position",
    "draft_best_pick_at_pos",
    "draft_early_picks_at_pos",
    "draft_any_picks_at_pos",
    "tm_pass_att_pg_prior",
    "tm_rush_att_pg_prior",
    "tm_pass_rate_prior",
    "tm_pass_epa_pg_prior",
    "tm_rush_epa_pg_prior",
    "tm_off_td_pg_prior",
    "tm_targets_pg_prior",
    "tm_pass_rate_prior_delta",
    "tm_pass_att_pg_prior_delta",
    "tm_off_td_pg_prior_delta",
}

# Known before kickoff, so not leakage.
KNOWN_PRESEASON = {"age", "years_exp"}


def feature_columns(panel: pl.DataFrame) -> list[str]:
    """Numeric features eligible for training, with targets excluded.

    Three categories are allowed and nothing else:
      1. ``*_lag*`` columns — prior-season statistics, lagged by
         construction in ``features.build_panel``
      2. ``KNOWN_PRESEASON`` — age and experience, known before kickoff
      3. ``OFFSEASON_FEATURES`` — team moves, draft capital, coaching
         changes and depth chart, all settled before week 1

    The allowlist is deliberate. A denylist would silently admit any new
    column someone joins onto the panel, which is exactly how leakage gets
    into projects like this one.
    """
    cols = []
    for c in panel.columns:
        if c in _EXCLUDE:
            continue
        if not panel.schema[c].is_numeric():
            continue
        if (
            c.endswith(("_lag1", "_lag2", "_lag3"))
            or c in KNOWN_PRESEASON
            or c in OFFSEASON_FEATURES
        ):
            cols.append(c)
    return cols


def _design_matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Features plus one-hot position."""
    X = df.select(cols).to_numpy().astype(np.float64)
    pos = df["position"].to_numpy()
    onehot = np.column_stack([(pos == p).astype(np.float64)
                              for p in ("QB", "RB", "WR", "TE")])
    return np.hstack([X, onehot])


@dataclass
class GBMProjector:
    """LightGBM point-estimate projector for season fantasy points."""

    target: str = "y_points"
    params: dict = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 4,
        "lambda_l2": 1.0,
        "verbosity": -1,
    })
    num_boost_round: int = 500
    _cols: list[str] = field(default_factory=list, init=False)
    _booster: lgb.Booster | None = field(default=None, init=False)

    def fit(self, train: pl.DataFrame) -> "GBMProjector":
        self._cols = feature_columns(train)
        X = _design_matrix(train, self._cols)
        y = train[self.target].to_numpy()
        self._booster = lgb.train(
            self.params, lgb.Dataset(X, label=y),
            num_boost_round=self.num_boost_round,
        )
        return self

    def predict(self, test: pl.DataFrame) -> np.ndarray:
        assert self._booster is not None, "call fit() first"
        return self._booster.predict(_design_matrix(test, self._cols))

    def fit_predict(self, train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
        return self.fit(train).predict(test)


@dataclass
class QuantileProjector:
    """Predicts a distribution: one LightGBM model per quantile.

    Yields p10/p25/p50/p75/p90 per player, which feeds:
      - risk-aware VOR (replacement level on a distribution, not a mean)
      - ceiling/floor draft strategies
      - honest calibration reporting
    """

    target: str = "y_points"
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    num_boost_round: int = 400
    base_params: dict = field(default_factory=lambda: {
        "objective": "quantile",
        "learning_rate": 0.04,
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

    def fit(self, train: pl.DataFrame) -> "QuantileProjector":
        self._cols = feature_columns(train)
        X = _design_matrix(train, self._cols)
        y = train[self.target].to_numpy()
        ds = lgb.Dataset(X, label=y)
        for q in self.quantiles:
            params = dict(self.base_params, alpha=q)
            self._boosters[q] = lgb.train(
                params, ds, num_boost_round=self.num_boost_round
            )
        return self

    def predict(self, test: pl.DataFrame) -> dict[float, np.ndarray]:
        X = _design_matrix(test, self._cols)
        return {q: b.predict(X) for q, b in self._boosters.items()}

    def predict_frame(self, test: pl.DataFrame) -> pl.DataFrame:
        """Attach p10..p90 columns, enforcing monotonicity.

        Independently-fit quantile models can cross (p25 > p50) on small
        samples. Sorting each row is the standard cheap fix; the alternative
        is a monotonic joint model, which is v2 territory.
        """
        preds = self.predict(test)
        qs = sorted(preds)
        stacked = np.sort(np.column_stack([preds[q] for q in qs]), axis=1)
        return test.with_columns([
            pl.Series(f"p{int(q * 100):02d}", stacked[:, i])
            for i, q in enumerate(qs)
        ])
