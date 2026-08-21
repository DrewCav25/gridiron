"""Phase 5b: does optimizing against calibrated distributions recover the edge?

Finding #5 showed the Monte Carlo agent beating greedy VOR at its own
objective (+40.7, 90.8% of drafts) while *losing* on realized points
(-16.0). The diagnosis was the optimizer's curse: the rollouts scored
candidates with **point** projections, so the agent exploited the shape of
a noisy surface, error included.

This script tests that diagnosis directly. Same agent, same rollouts, one
change: rollouts are scored against the Phase 4 calibrated distributions
instead of point estimates. If the diagnosis is right the edge should
return. If it doesn't, greedy VOR is simply correct at this signal-to-noise
ratio — which is also worth knowing.
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from gridiron.config import LeagueConfig, ScoringConfig
from gridiron.features import build_panel
from gridiron.offseason import attach_offseason_features
from gridiron.evaluate import walk_forward
from gridiron.models import GBMProjector
from gridiron.calibration import CompositeProjector, ConformalCalibrator
from gridiron.draft import (
    AdpAgent,
    DraftPool,
    GreedyVorAgent,
    MonteCarloAgent,
    optimal_lineup_points,
    simulate_draft,
)

SEASONS = list(range(2012, 2026))
LEAGUE = LeagueConfig(
    teams=12, qb=1, rb=2, wr=2, te=1, flex=1, k=0, dst=0, bench=6,
    scoring=ScoringConfig.half_ppr(),
)
ROUNDS = LEAGUE.roster_size


def build_pools(test_seasons: list[int], n_scenarios: int) -> dict[int, DraftPool]:
    """Draft boards carrying both a point projection and calibrated samples."""
    panel = build_panel(SEASONS, ScoringConfig.half_ppr()).filter(
        pl.col("games_played_lag1") >= 6
    )
    full = attach_offseason_features(panel, SEASONS)
    scored = walk_forward(
        full, lambda tr, te: GBMProjector().fit_predict(tr, te), test_seasons
    )

    pools = {}
    for season in test_seasons:
        # Conformal needs a calibration season the model never trained on.
        proper = full.filter(pl.col("season") < season - 1)
        calib = full.filter(pl.col("season") == season - 1)
        test = scored.filter(pl.col("season") == season)

        composite = CompositeProjector(dependence="copula", n_samples=n_scenarios)
        composite.fit(proper)
        conformal = ConformalCalibrator().fit(composite, calib)

        raw = composite.sample_totals(test)
        samples = conformal.widen_samples(raw)

        pool = DraftPool.from_frame(test)
        # from_frame drops rows with null projections; align sample rows to
        # the surviving players rather than assuming the frames match.
        keep = (
            test.with_row_index("_i")
            .filter(pl.col("player_id").is_in(pool.player_id.tolist()))["_i"]
            .to_numpy()
        )
        pool.samples = samples[keep]
        pools[season] = pool
        print(f"  {season}: pool={pool.n} samples={pool.samples.shape} "
              f"rho={composite.rho:.3f}", flush=True)
    return pools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+",
                    default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--rollouts", type=int, default=24)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--scenarios", type=int, default=200)
    args = ap.parse_args()

    pools = build_pools(args.seasons, args.scenarios)
    rows = []

    for season, pool in pools.items():
        for slot in range(LEAGUE.teams):
            for rep in range(args.replicates):
                seed = 1000 * season + 37 * slot + rep

                def run(agent) -> float:
                    strategies = [AdpAgent() for _ in range(LEAGUE.teams)]
                    strategies[slot] = agent
                    rosters = simulate_draft(pool, LEAGUE, strategies, ROUNDS, seed=seed)
                    return optimal_lineup_points(
                        rosters[slot], pool.actual, pool.position, LEAGUE
                    )

                mc_kw = dict(
                    n_candidates=args.candidates,
                    n_rollouts=args.rollouts,
                    seed=seed,
                )
                rows.append({
                    "season": season, "slot": slot, "rep": rep,
                    "greedy_vor": run(GreedyVorAgent()),
                    "mc_point": run(MonteCarloAgent(**mc_kw)),
                    "mc_dist": run(MonteCarloAgent(use_distribution=True, **mc_kw)),
                    "mc_dist_floor": run(MonteCarloAgent(
                        use_distribution=True, objective="floor", **mc_kw)),
                })
        print(f"  season {season} drafts done ({len(rows)})", flush=True)

    res = pl.DataFrame(rows).with_columns(
        (pl.col("mc_point") - pl.col("greedy_vor")).alias("point_minus_greedy"),
        (pl.col("mc_dist") - pl.col("greedy_vor")).alias("dist_minus_greedy"),
        (pl.col("mc_dist") - pl.col("mc_point")).alias("dist_minus_point"),
        (pl.col("mc_dist_floor") - pl.col("greedy_vor")).alias("floor_minus_greedy"),
    )
    res.write_parquet("data/cache/_phase5b_results.parquet")

    pl.Config.set_tbl_rows(30)
    print("\n=== Phase 5b: mean realized starting-lineup points ===")
    print(res.select("greedy_vor", "mc_point", "mc_dist", "mc_dist_floor").mean()
          .transpose(include_header=True, header_name="agent",
                     column_names=["mean_points"]))

    print("\n=== Paired differences on REALIZED points ===")
    for col, label in (
        ("point_minus_greedy", "MC point-estimate  - greedy VOR"),
        ("dist_minus_greedy", "MC distribution    - greedy VOR"),
        ("floor_minus_greedy", "MC floor (p25)     - greedy VOR"),
        ("dist_minus_point", "MC distribution    - MC point-estimate"),
    ):
        d = res[col].to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{label}: {d.mean():+7.2f}  "
              f"95% CI [{d.mean() - 1.96 * se:+7.2f}, {d.mean() + 1.96 * se:+7.2f}]  "
              f"win {(d > 0).mean():5.1%}")

    print("\n--- by season ---")
    print(res.group_by("season").agg(
        pl.col("greedy_vor").mean(), pl.col("mc_point").mean(),
        pl.col("mc_dist").mean(), pl.col("dist_minus_greedy").mean(),
    ).sort("season"))


if __name__ == "__main__":
    main()
