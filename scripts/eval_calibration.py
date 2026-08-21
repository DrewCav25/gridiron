"""Phase 4 evaluation: does composing availability x production fix coverage?

Compares the direct season-total quantile model (Phase 2) against the
composite model under three dependence assumptions.
"""

from __future__ import annotations

import polars as pl
import numpy as np

from gridiron.config import ScoringConfig
from gridiron.features import build_panel
from gridiron.offseason import attach_offseason_features
from gridiron.evaluate import interval_coverage, pinball_loss
from gridiron.models import QuantileProjector
from gridiron.calibration import CompositeProjector

SEASONS = list(range(2012, 2026))
TEST = list(range(2021, 2026))


def main() -> None:
    panel = build_panel(SEASONS, ScoringConfig.half_ppr()).filter(
        pl.col("games_played_lag1") >= 6
    )
    full = attach_offseason_features(panel, SEASONS)

    rows = []
    for season in TEST:
        tr = full.filter(pl.col("season") < season)
        te = full.filter(pl.col("season") == season)
        y = te["y_points"].to_numpy()

        def record(name: str, f: pl.DataFrame, rho=None) -> None:
            rows.append({
                "season": season,
                "model": name,
                "cov90": interval_coverage(y, f["p10"].to_numpy(), f["p90"].to_numpy()),
                "cov50": interval_coverage(y, f["p25"].to_numpy(), f["p75"].to_numpy()),
                "pinball50": pinball_loss(y, f["p50"].to_numpy(), 0.5),
                "rho": rho,
            })

        record("direct", QuantileProjector().fit(tr).predict_frame(te))

        for dep in ("independent", "copula", "comonotonic"):
            cp = CompositeProjector(dependence=dep).fit(tr)
            record(f"composite_{dep}", cp.predict_frame(te), cp.rho)
            print(f"  {season} {dep} rho={cp.rho:.3f}", flush=True)

    res = pl.DataFrame(rows)
    res.write_parquet("data/cache/_phase4_results.parquet")

    summary = res.group_by("model").agg(
        pl.col("cov90").mean(),
        pl.col("cov50").mean(),
        pl.col("pinball50").mean(),
        pl.col("rho").mean(),
    ).sort("cov90")

    pl.Config.set_tbl_rows(20)
    print("\n=== Phase 4: interval coverage (targets: cov90=0.80, cov50=0.50) ===")
    print(summary)
    print("\n--- per season ---")
    print(res.sort(["model", "season"]))


if __name__ == "__main__":
    main()
