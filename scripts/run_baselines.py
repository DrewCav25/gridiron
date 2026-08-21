"""Reproduce every number in the README.

    python scripts/run_baselines.py

First run downloads ~14 seasons of nflverse data and caches it to
data/cache/ as parquet. Subsequent runs are fast.
"""

from __future__ import annotations

import argparse
import logging

import polars as pl

from gridiron.config import ScoringConfig
from gridiron.features import build_panel, build_season_table, stickiness_report
from gridiron.evaluate import (
    baseline_opportunity,
    baseline_persistence,
    baseline_ppg_times_16,
    evaluate_predictions,
    interval_coverage,
    pinball_loss,
    summarize,
    walk_forward,
)
from gridiron.models import GBMProjector, QuantileProjector, OFFSEASON_FEATURES
from gridiron.offseason import attach_offseason_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
pl.Config.set_tbl_rows(50)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2012)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--test-from", type=int, default=2018)
    ap.add_argument("--scoring", default="half_ppr",
                    choices=["standard", "half_ppr", "ppr"])
    args = ap.parse_args()

    scoring = getattr(ScoringConfig, args.scoring)()
    seasons = list(range(args.start, args.end + 1))
    test_seasons = list(range(args.test_from, args.end + 1))

    # ---- 1. The thesis: opportunity is sticky, efficiency is not ----------
    section("1. Year-over-year stickiness (min 8 games)")
    table = build_season_table(seasons, scoring)
    for pos in ("RB", "WR", "TE", "QB"):
        print(f"\n--- {pos} ---")
        print(stickiness_report(table.filter(pl.col("position") == pos)))

    # ---- 2. Baselines vs model, walk-forward -----------------------------
    section(f"2. Walk-forward {test_seasons[0]}-{test_seasons[-1]}, season total points")
    panel = build_panel(seasons, scoring).filter(pl.col("games_played_lag1") >= 6)

    runs = {
        "persistence": baseline_persistence,
        "ppg_x16": baseline_ppg_times_16,
        "opportunity": baseline_opportunity,
        "gbm": lambda tr, te: GBMProjector().fit_predict(tr, te),
    }
    frames = []
    for name, fn in runs.items():
        scored = walk_forward(panel, fn, test_seasons)
        res = evaluate_predictions(scored, "y_hat")
        frames.append(summarize(res, "position").with_columns(pl.lit(name).alias("model")))

    print(
        pl.concat(frames)
        .select("model", "position", "n_total", "spearman", "top12_hit", "mae")
        .sort(["position", "spearman"], descending=[False, True])
    )

    # ---- 3. Offseason features -------------------------------------------
    section("3. Phase 3 — does offseason information close the gap?")
    full = attach_offseason_features(panel, seasons)
    lag_only = full.drop([c for c in OFFSEASON_FEATURES if c in full.columns])
    strict = attach_offseason_features(panel, seasons, include_depth_chart=False)

    gbm = lambda tr, te: GBMProjector().fit_predict(tr, te)  # noqa: E731
    variants = [
        ("persistence", lag_only, baseline_persistence),
        ("gbm_lag_only", lag_only, gbm),
        ("gbm_offseason", full, gbm),
        ("gbm_strict_draftday", strict, gbm),
    ]
    frames = []
    for name, pnl, fn in variants:
        scored = walk_forward(pnl, fn, test_seasons)
        frames.append(
            summarize(evaluate_predictions(scored, "y_hat"), "position")
            .with_columns(pl.lit(name).alias("model"))
        )
    print(
        pl.concat(frames)
        .select("model", "position", "n_total", "spearman", "top12_hit", "mae")
        .sort(["position", "spearman"], descending=[False, True])
    )

    print("\n--- Feature importance (gain), top 15 ---")
    m = GBMProjector().fit(full.filter(pl.col("season") < test_seasons[-1]))
    names = m._cols + ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]
    gains = m._booster.feature_importance(importance_type="gain")
    imp = (
        pl.DataFrame({"feature": names, "gain": gains})
        .with_columns((pl.col("gain") / pl.col("gain").sum() * 100).alias("pct"))
        .sort("gain", descending=True)
    )
    for row in imp.head(15).iter_rows(named=True):
        tag = "  <- offseason" if row["feature"] in OFFSEASON_FEATURES else ""
        print(f"  {row['pct']:6.2f}%  {row['feature']}{tag}")

    # ---- 4. Calibration --------------------------------------------------
    section("4. Quantile calibration (the check most projects skip)")
    rows = []
    for season in test_seasons[2:]:
        tr = panel.filter(pl.col("season") < season)
        te = panel.filter(pl.col("season") == season)
        f = QuantileProjector().fit(tr).predict_frame(te)
        y = f["y_points"].to_numpy()
        rows.append({
            "season": season,
            "n": len(y),
            "cov_p10_p90": interval_coverage(y, f["p10"].to_numpy(), f["p90"].to_numpy()),
            "cov_p25_p75": interval_coverage(y, f["p25"].to_numpy(), f["p75"].to_numpy()),
            "pinball_p50": pinball_loss(y, f["p50"].to_numpy(), 0.5),
        })
    cal = pl.DataFrame(rows)
    print(cal)
    print(f"\nmean p10-p90 coverage: {cal['cov_p10_p90'].mean():.3f}  (target 0.80)")
    print(f"mean p25-p75 coverage: {cal['cov_p25_p75'].mean():.3f}  (target 0.50)")
    print("\nIntervals are too narrow. See README finding #4 — this is the")
    print("headline open problem, and the reason calibration gets reported.")


if __name__ == "__main__":
    main()
