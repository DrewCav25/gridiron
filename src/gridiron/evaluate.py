"""Walk-forward evaluation and metrics.

Two rules this module exists to enforce:

1. **Walk-forward only.** To score season N we train on seasons < N. There
   is no random split anywhere in this project. Shuffling a time series
   means training on the future, and it is the reason most public fantasy
   models report numbers they cannot reproduce live.

2. **Rank metrics lead.** You draft an *ordering*, not a point total. A model
   that is off by 20 points on everyone but gets the order right is more
   useful than one with lower MAE and worse ordering. Spearman and top-k
   hit rate are the headline numbers; MAE and RMSE are secondary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import polars as pl
from scipy.stats import spearmanr


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def topk_hit_rate(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Fraction of the actual top-k that appear in the predicted top-k.

    This is the question a drafter actually asks: "did you correctly
    identify the top 12 RBs?"
    """
    if len(y_true) < k:
        return float("nan")
    actual = set(np.argsort(-y_true)[:k])
    predicted = set(np.argsort(-y_pred)[:k])
    return len(actual & predicted) / k


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Quantile (pinball) loss — the correct scoring rule for quantile models."""
    d = y_true - y_pred
    return float(np.mean(np.maximum(quantile * d, (quantile - 1) * d)))


def interval_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Fraction of actuals falling inside the predicted interval.

    For a p10-p90 band this should be ~0.80. Reporting it is the single
    fastest way to look like someone who has done this professionally,
    because almost no portfolio project checks calibration at all.
    """
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


# --------------------------------------------------------------------------
# Walk-forward harness
# --------------------------------------------------------------------------

@dataclass
class FoldResult:
    season: int
    position: str
    n: int
    metrics: dict[str, float]


def evaluate_predictions(
    df: pl.DataFrame,
    pred_col: str,
    target_col: str = "y_points",
    by_position: bool = True,
    ks: Sequence[int] = (12, 24, 36),
) -> pl.DataFrame:
    """Score a frame that already contains predictions.

    Metrics are computed *within season and position*, because that is the
    comparison a drafter makes. Pooling positions inflates rank correlation
    dramatically — QBs outscore everyone, so any model that knows what a QB
    is looks good on a pooled metric. Don't report the pooled number.
    """
    groups = ["season", "position"] if by_position else ["season"]
    rows = []

    for key, sub in df.drop_nulls([pred_col, target_col]).group_by(groups):
        y = sub[target_col].to_numpy()
        p = sub[pred_col].to_numpy()
        if len(y) < 5:
            continue
        m = {
            "spearman": spearman(y, p),
            "mae": mae(y, p),
            "rmse": rmse(y, p),
            "n": len(y),
        }
        for k in ks:
            if len(y) >= k:
                m[f"top{k}_hit"] = topk_hit_rate(y, p, k)
        row = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
        row.update(m)
        rows.append(row)

    return pl.DataFrame(rows).sort(groups)


def summarize(results: pl.DataFrame, by: str = "position") -> pl.DataFrame:
    """Average fold metrics, weighted by fold size."""
    metric_cols = [
        c for c in results.columns
        if c not in {"season", "position", "n"} and results.schema[c].is_numeric()
    ]
    return results.group_by(by).agg(
        [pl.col("n").sum().alias("n_total"), pl.len().alias("n_folds")]
        + [
            ((pl.col(c) * pl.col("n")).sum() / pl.col("n").sum()).alias(c)
            for c in metric_cols
        ]
    ).sort(by)


def walk_forward(
    panel: pl.DataFrame,
    fit_predict: Callable[[pl.DataFrame, pl.DataFrame], np.ndarray],
    test_seasons: Sequence[int],
    pred_col: str = "y_hat",
    min_train_seasons: int = 3,
) -> pl.DataFrame:
    """Train on seasons < N, predict season N, for each N in ``test_seasons``.

    ``fit_predict`` receives (train_df, test_df) and returns predictions for
    test_df. It must not look at ``test_df``'s target columns.
    """
    out = []
    for season in sorted(test_seasons):
        train = panel.filter(pl.col("season") < season)
        test = panel.filter(pl.col("season") == season)
        if train["season"].n_unique() < min_train_seasons or test.is_empty():
            continue
        preds = fit_predict(train, test)
        out.append(test.with_columns(pl.Series(pred_col, preds)))

    if not out:
        raise ValueError("no evaluable folds — check test_seasons and panel coverage")
    return pl.concat(out, how="vertical_relaxed")


# --------------------------------------------------------------------------
# Baselines — build these before any model, and never delete them
# --------------------------------------------------------------------------

def baseline_persistence(train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    """Last season's total points. The floor. Some published models lose to it."""
    return test["fantasy_points_total_lag1"].fill_null(0.0).to_numpy()


def baseline_ppg_times_16(train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    """Last season's per-game rate projected over a full season.

    Better than raw persistence because it doesn't punish players who
    missed time — which is most of the reason raw persistence fails.
    """
    ppg = test["fantasy_points_per_game_lag1"].fill_null(0.0).to_numpy()
    return ppg * 16.0


def baseline_opportunity(train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    """Points per opportunity (league-average, by position) times last
    season's opportunity count. Isolates 'volume is what matters'."""
    rate = (
        train.group_by("position")
        .agg(
            (pl.col("y_points").sum() / pl.col("opportunities_lag1").sum())
            .alias("pts_per_opp")
        )
    )
    joined = test.join(rate, on="position", how="left")
    return (
        joined["opportunities_lag1"].fill_null(0.0)
        * joined["pts_per_opp"].fill_null(0.0)
    ).to_numpy()
