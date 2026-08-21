"""Tests for distribution-aware roster scoring (Phase 5b).

The claim this code makes is subtle: scoring a roster against sampled
outcomes is *not* the same as scoring it against point projections, and
the gap is Jensen's inequality applied to an order statistic. That is easy
to get wrong in a way that silently degrades to the point-estimate case —
which would make the Phase 5b comparison meaningless while still producing
plausible numbers. These tests pin the distinction down.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridiron.config import LeagueConfig
from gridiron.draft import (
    POS_INDEX,
    AdpAgent,
    DraftPool,
    GreedyVorAgent,
    MonteCarloAgent,
    expected_lineup_points,
    optimal_lineup_points,
    simulate_draft,
)

LEAGUE = LeagueConfig(teams=12, qb=1, rb=2, wr=2, te=1, flex=1, k=0, dst=0, bench=6)


@pytest.fixture
def tiny():
    position = np.array(["QB", "QB", "RB", "RB", "RB", "WR", "WR", "TE"])
    points = np.array([10.0, 30.0, 5.0, 20.0, 15.0, 8.0, 12.0, 7.0])
    pos_idx = np.array([POS_INDEX[p] for p in position])
    return position, points, pos_idx


@pytest.fixture
def pool() -> DraftPool:
    rng = np.random.default_rng(0)
    positions, projections = [], []
    for pos, n, top, decay in (
        ("QB", 40, 320.0, 3.0), ("RB", 90, 300.0, 2.6),
        ("WR", 110, 290.0, 1.7), ("TE", 45, 240.0, 4.5),
    ):
        for i in range(n):
            positions.append(pos)
            projections.append(max(top - decay * i, 10.0))
    projections = np.array(projections)
    position = np.array(positions)
    n = len(position)
    p = DraftPool(
        player_id=np.arange(n).astype(str),
        name=np.array([f"P{i}" for i in range(n)]),
        position=position,
        projection=projections,
        actual=projections + rng.normal(0, 60, n),
        adp_rank=np.argsort(np.argsort(-projections)).astype(float),
    )
    # Heteroscedastic samples: spread grows with projection, so high-value
    # players carry more uncertainty — as they do in reality.
    spread = 0.35 * projections[:, None]
    p.samples = np.maximum(
        projections[:, None] + rng.normal(0, 1, (n, 200)) * spread, 0.0
    )
    return p


class TestExpectedLineupPoints:
    def test_degenerate_samples_match_the_scalar_scorer(self, tiny):
        """With zero variance the two scorers must agree exactly.

        If they don't, the vectorised implementation has a slot-filling
        bug that every other test would be measuring around.
        """
        position, points, pos_idx = tiny
        samples = np.repeat(points[:, None], 40, axis=1)
        roster = list(range(len(points)))
        assert expected_lineup_points(roster, samples, pos_idx, LEAGUE) == pytest.approx(
            optimal_lineup_points(roster, points, position, LEAGUE)
        )

    def test_jensen_gap_is_positive(self, tiny):
        """E[lineup(X)] >= lineup(E[X]) — the reason this module exists.

        A lineup is an order statistic and therefore convex, so scoring
        with point projections systematically understates roster value.
        """
        position, points, pos_idx = tiny
        rng = np.random.default_rng(3)
        samples = points[:, None] + rng.normal(0, 6.0, (len(points), 4000))
        roster = list(range(len(points)))
        stochastic = expected_lineup_points(roster, samples, pos_idx, LEAGUE)
        deterministic = optimal_lineup_points(roster, points, position, LEAGUE)
        assert stochastic > deterministic

    def test_depth_has_option_value_under_uncertainty(self, tiny):
        """Two mediocre volatile RBs can beat one steady RB of equal mean.

        A point estimate cannot express this at all; it is exactly the
        blind spot the distribution scoring exists to fix.
        """
        pos_idx = np.array([POS_INDEX[p] for p in ["QB", "RB", "RB", "WR", "WR", "TE"]])
        rng = np.random.default_rng(11)
        base = np.array([200.0, 150.0, 150.0, 150.0, 150.0, 100.0])
        steady = base[:, None] + rng.normal(0, 1.0, (6, 3000))
        volatile = base[:, None] + rng.normal(0, 60.0, (6, 3000))
        roster = list(range(6))
        assert (
            expected_lineup_points(roster, volatile, pos_idx, LEAGUE)
            > expected_lineup_points(roster, steady, pos_idx, LEAGUE)
        )

    def test_floor_is_below_mean_is_below_ceiling(self, tiny):
        position, points, pos_idx = tiny
        rng = np.random.default_rng(5)
        samples = np.maximum(points[:, None] + rng.normal(0, 5.0, (len(points), 3000)), 0)
        roster = list(range(len(points)))
        floor = expected_lineup_points(roster, samples, pos_idx, LEAGUE, "floor")
        mean = expected_lineup_points(roster, samples, pos_idx, LEAGUE, "mean")
        ceiling = expected_lineup_points(roster, samples, pos_idx, LEAGUE, "ceiling")
        assert floor < mean < ceiling

    def test_empty_roster_scores_zero(self, tiny):
        _, points, pos_idx = tiny
        samples = np.repeat(points[:, None], 10, axis=1)
        assert expected_lineup_points([], samples, pos_idx, LEAGUE) == 0.0


class TestDistributionAgent:
    def test_distribution_agent_drafts_a_legal_roster(self, pool):
        strategies = [AdpAgent() for _ in range(12)]
        strategies[3] = MonteCarloAgent(use_distribution=True, n_rollouts=6, seed=1)
        rosters = simulate_draft(pool, LEAGUE, strategies, LEAGUE.roster_size, seed=7)
        picks = [p for r in rosters for p in r]
        assert len(picks) == len(set(picks))
        assert all(len(r) == LEAGUE.roster_size for r in rosters)

    def test_distribution_and_point_agents_differ(self, pool):
        """If they draft identically the Phase 5b comparison is vacuous."""
        def draft(use_distribution: bool):
            strategies = [AdpAgent() for _ in range(12)]
            strategies[0] = MonteCarloAgent(
                use_distribution=use_distribution, n_rollouts=12, seed=2
            )
            return simulate_draft(pool, LEAGUE, strategies, LEAGUE.roster_size, seed=9)[0]

        assert draft(True) != draft(False)

    def test_falls_back_to_point_estimates_without_samples(self, pool):
        """Missing samples must degrade gracefully, not crash."""
        pool.samples = None
        strategies = [AdpAgent() for _ in range(12)]
        strategies[0] = MonteCarloAgent(use_distribution=True, n_rollouts=6, seed=3)
        rosters = simulate_draft(pool, LEAGUE, strategies, LEAGUE.roster_size, seed=4)
        assert len(rosters[0]) == LEAGUE.roster_size

    def test_distribution_agent_is_reproducible(self, pool):
        def draft():
            strategies = [AdpAgent() for _ in range(12)]
            strategies[5] = MonteCarloAgent(use_distribution=True, n_rollouts=8, seed=21)
            return simulate_draft(pool, LEAGUE, strategies, LEAGUE.roster_size, seed=6)
        assert draft() == draft()

    def test_agents_still_cannot_see_realized_points(self, pool):
        """Re-asserted for the distribution path specifically.

        `samples` and `actual` are both season-outcome shaped, so it would
        be easy to wire the wrong one in and manufacture a huge fake edge.
        """
        strategies = [AdpAgent() for _ in range(12)]
        strategies[2] = MonteCarloAgent(use_distribution=True, n_rollouts=6, seed=8)
        strategies[7] = GreedyVorAgent()
        before = simulate_draft(pool, LEAGUE, strategies, LEAGUE.roster_size, seed=13)

        pool.actual = np.random.default_rng(77).permutation(pool.actual)
        strategies[2] = MonteCarloAgent(use_distribution=True, n_rollouts=6, seed=8)
        after = simulate_draft(pool, LEAGUE, strategies, LEAGUE.roster_size, seed=13)
        assert before == after


class TestConformalSampleWidening:
    def test_widening_increases_spread_and_keeps_the_median(self):
        from gridiron.calibration import ConformalCalibrator

        rng = np.random.default_rng(0)
        samples = rng.normal(100.0, 20.0, (50, 2000))
        cal = ConformalCalibrator()
        cal._offsets[(0.10, 0.90)] = 15.0

        widened = cal.widen_samples(samples)
        before = np.percentile(samples, 90, axis=1) - np.percentile(samples, 10, axis=1)
        after = np.percentile(widened, 90, axis=1) - np.percentile(widened, 10, axis=1)
        assert (after > before).all()
        assert np.allclose(
            np.median(widened, axis=1), np.median(samples, axis=1), atol=1e-6
        )

    def test_no_offset_is_a_no_op(self):
        from gridiron.calibration import ConformalCalibrator

        samples = np.random.default_rng(1).normal(50, 10, (20, 100))
        assert np.array_equal(ConformalCalibrator().widen_samples(samples), samples)

    def test_widened_samples_stay_non_negative(self):
        from gridiron.calibration import ConformalCalibrator

        samples = np.random.default_rng(2).normal(5.0, 4.0, (30, 500))
        cal = ConformalCalibrator()
        cal._offsets[(0.10, 0.90)] = 20.0
        assert cal.widen_samples(samples).min() >= 0.0
