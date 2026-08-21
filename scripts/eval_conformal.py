"""Phase 4b: does conformal calibration close the remaining coverage gap?

Fit on seasons < N-1, calibrate the intervals on season N-1, test on N.
The calibration season is never in the model's training data, which is
what makes the conformal correction valid.
"""

from __future__ import annotations

import polars as pl

from gridiron.config import ScoringConfig
from gridiron.features import build_panel
from gridiron.offseason import attach_offseason_features
from gridiron.evaluate import interval_coverage
from gridiron.calibration import CompositeProjector, ConformalCalibrator

SEASONS = list(range(2012, 2026))
TEST = list(range(2021, 2026))


def main() -> None:
    panel = build_panel(SEASONS, ScoringConfig.half_ppr()).filter(
        pl.col("games_played_lag1") >= 6
    )
    full = attach_offseason_features(panel, SEASONS)

    rows = []
    for season in TEST:
        proper = full.filter(pl.col("season") < season - 1)
        calib = full.filter(pl.col("season") == season - 1)
        test = full.filter(pl.col("season") == season)
        y = test["y_points"].to_numpy()

        model = CompositeProjector(dependence="copula", n_samples=3000).fit(proper)

        raw = model.predict_frame(test)
        rows.append({
            "season": season, "model": "composite_copula",
            "cov90": interval_coverage(y, raw["p10"].to_numpy(), raw["p90"].to_numpy()),
            "cov50": interval_coverage(y, raw["p25"].to_numpy(), raw["p75"].to_numpy()),
            "width90": float((raw["p90"] - raw["p10"]).mean()),
        })

        cal = ConformalCalibrator().fit(model, calib)
        adj = cal.predict_frame(model, test)
        rows.append({
            "season": season, "model": "+ conformal",
            "cov90": interval_coverage(y, adj["p10"].to_numpy(), adj["p90"].to_numpy()),
            "cov50": interval_coverage(y, adj["p25"].to_numpy(), adj["p75"].to_numpy()),
            "width90": float((adj["p90"] - adj["p10"]).mean()),
        })
        print(f"  {season} offsets={ {k: round(v, 1) for k, v in cal.offsets.items()} }",
              flush=True)

    res = pl.DataFrame(rows)
    res.write_parquet("data/cache/_phase4b_results.parquet")

    pl.Config.set_tbl_rows(20)
    print("\n=== Phase 4b: conformal calibration (targets 0.80 / 0.50) ===")
    print(
        res.group_by("model")
        .agg(pl.col("cov90").mean(), pl.col("cov50").mean(), pl.col("width90").mean())
        .sort("cov90")
    )
    print("\n--- per season ---")
    print(res.sort(["model", "season"]))


if __name__ == "__main__":
    main()
