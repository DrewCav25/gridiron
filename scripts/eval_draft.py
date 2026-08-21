"""Phase 5: does sequential lookahead beat greedy VOR?

The comparison is **paired**. For a given season, draft slot and seed, the
same draft is run twice — once with the focal team drafting greedily by
VOR, once with it running Monte Carlo lookahead. Everything else is held
identical: the same projections, the same opponents, the same random
draws for opponent behaviour.

That isolates strategy. An unpaired comparison would be swamped by
variance from which players happened to fall to which slot, and would
need far more simulations to say anything.
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
from gridiron.draft import (
    AdpAgent,
    DraftPool,
    GreedyVorAgent,
    MonteCarloAgent,
    optimal_lineup_points,
    simulate_draft,
)

SEASONS = list(range(2012, 2026))

# No kickers or defenses — neither is modelled, per the projection scope.
LEAGUE = LeagueConfig(
    teams=12, qb=1, rb=2, wr=2, te=1, flex=1, k=0, dst=0, bench=6,
    scoring=ScoringConfig.half_ppr(),
)
ROUNDS = LEAGUE.roster_size


def build_pools(test_seasons: list[int]) -> dict[int, DraftPool]:
    """Walk-forward projections turned into draft boards, one per season."""
    panel = build_panel(SEASONS, ScoringConfig.half_ppr()).filter(
        pl.col("games_played_lag1") >= 6
    )
    full = attach_offseason_features(panel, SEASONS)
    scored = walk_forward(
        full, lambda tr, te: GBMProjector().fit_predict(tr, te), test_seasons
    )
    return {
        int(s): DraftPool.from_frame(scored.filter(pl.col("season") == s))
        for s in test_seasons
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--rollouts", type=int, default=24)
    ap.add_argument("--candidates", type=int, default=8)
    args = ap.parse_args()

    pools = build_pools(args.seasons)
    rows = []

    for season, pool in pools.items():
        for slot in range(LEAGUE.teams):
            for rep in range(args.replicates):
                seed = 1000 * season + 37 * slot + rep

                def run(focal_agent) -> tuple[float, float]:
                    """Returns (projected, realized) starting-lineup points.

                    Projected is the objective the agent can actually
                    optimize; realized is the outcome, which carries a
                    full season of luck on top.
                    """
                    strategies = [AdpAgent() for _ in range(LEAGUE.teams)]
                    strategies[slot] = focal_agent
                    rosters = simulate_draft(pool, LEAGUE, strategies, ROUNDS, seed=seed)
                    r = rosters[slot]
                    return (
                        optimal_lineup_points(r, pool.projection, pool.position, LEAGUE),
                        optimal_lineup_points(r, pool.actual, pool.position, LEAGUE),
                    )

                adp_p, adp_a = run(AdpAgent())
                grd_p, grd_a = run(GreedyVorAgent())
                mc_p, mc_a = run(MonteCarloAgent(
                    n_candidates=args.candidates,
                    n_rollouts=args.rollouts,
                    seed=seed,
                ))
                rows.append({
                    "season": season, "slot": slot, "rep": rep,
                    "adp": adp_a, "greedy_vor": grd_a, "monte_carlo": mc_a,
                    "adp_proj": adp_p, "greedy_vor_proj": grd_p,
                    "monte_carlo_proj": mc_p,
                })
        print(f"  season {season} done ({len(rows)} drafts)", flush=True)

    res = pl.DataFrame(rows).with_columns(
        (pl.col("monte_carlo") - pl.col("greedy_vor")).alias("mc_minus_greedy"),
        (pl.col("greedy_vor") - pl.col("adp")).alias("greedy_minus_adp"),
        (pl.col("monte_carlo_proj") - pl.col("greedy_vor_proj")).alias("mc_minus_greedy_proj"),
    )
    res.write_parquet("data/cache/_phase5_results.parquet")

    pl.Config.set_tbl_rows(30)
    print("\n=== Phase 5: mean realized starting-lineup points ===")
    print(
        res.select("adp", "greedy_vor", "monte_carlo").mean().transpose(
            include_header=True, header_name="agent", column_names=["mean_points"]
        )
    )

    def report(col: str, label: str) -> None:
        d = res[col].to_numpy()
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        print(f"\nPaired: Monte Carlo - greedy VOR, {label}")
        print(f"  n = {n} paired drafts")
        print(f"  mean difference = {d.mean():+.2f} points/season")
        print(f"  95% CI = [{d.mean() - 1.96 * se:+.2f}, {d.mean() + 1.96 * se:+.2f}]")
        print(f"  win rate = {(d > 0).mean():.1%}")
        print(f"  sd of paired difference = {d.std(ddof=1):.1f}")

    report("mc_minus_greedy_proj", "scored on PROJECTIONS (the agent objective)")
    report("mc_minus_greedy", "scored on REALIZED points")

    g = res["greedy_minus_adp"].to_numpy()
    print(f"\nPaired: greedy VOR - ADP field  (mean {g.mean():+.2f}, "
          f"win rate {(g > 0).mean():.1%})")

    print("\n--- by season ---")
    print(res.group_by("season").agg(
        pl.col("adp").mean(), pl.col("greedy_vor").mean(),
        pl.col("monte_carlo").mean(), pl.col("mc_minus_greedy").mean(),
    ).sort("season"))

    print("\n--- by draft slot (1 = first overall) ---")
    print(res.group_by("slot").agg(
        pl.col("mc_minus_greedy").mean(), pl.col("greedy_vor").mean(),
    ).sort("slot"))


if __name__ == "__main__":
    main()
