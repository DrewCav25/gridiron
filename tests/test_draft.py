"""Draft simulator and agent tests.

The most important test in this file is
``test_agents_cannot_see_realized_points``. The whole Phase 5 result rests
on agents drafting from projections alone; if realized points leaked into
a pick decision the simulation would report a spectacular edge that does
not exist. Shuffling the actuals and asserting the draft is unchanged
proves that directly, rather than by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridiron.config import LeagueConfig
from gridiron.draft import (
    POSITION_CAPS,
    AdpAgent,
    DraftPool,
    GreedyVorAgent,
    MonteCarloAgent,
    optimal_lineup_points,
    replacement_levels,
    score_rosters,
    simulate_draft,
    snake_order,
    value_over_replacement,
    _future_gaps,
)

LEAGUE = LeagueConfig(teams=12, qb=1, rb=2, wr=2, te=1, flex=1, k=0, dst=0, bench=6)


@pytest.fixture
def pool() -> DraftPool:
    """Synthetic board with a known structure.

    Deliberately gives each position a different scarcity curve, so the
    optimizer has something real to reason about.
    """
    rng = np.random.default_rng(0)
    positions, projections = [], []
    for pos, n, top, decay in (
        ("QB", 40, 320.0, 3.0),
        ("RB", 90, 300.0, 2.6),
        ("WR", 110, 290.0, 1.7),
        ("TE", 45, 240.0, 4.5),
    ):
        for i in range(n):
            positions.append(pos)
            projections.append(max(top - decay * i, 10.0))
    projections = np.array(projections)
    position = np.array(positions)
    n = len(position)
    return DraftPool(
        player_id=np.arange(n).astype(str),
        name=np.array([f"P{i}" for i in range(n)]),
        position=position,
        projection=projections,
        actual=projections + rng.normal(0, 60, n),
        adp_rank=np.argsort(np.argsort(-projections)).astype(float),
    )


class TestDraftOrder:
    def test_snake_reverses_each_round(self):
        assert snake_order(3, 2) == [0, 1, 2, 2, 1, 0]

    def test_every_team_picks_once_per_round(self):
        order = snake_order(12, 13)
        assert len(order) == 156
        for r in range(13):
            assert sorted(order[r * 12:(r + 1) * 12]) == list(range(12))

    def test_future_gaps_match_the_order(self):
        order = snake_order(4, 3)
        # Team 0 picks at slots 0, 7, 8.
        assert _future_gaps(order, 0, 0) == [6, 0]

    def test_turn_ends_are_back_to_back(self):
        """The team on the wheel picks twice in a row across the turn."""
        order = snake_order(6, 4)
        assert _future_gaps(order, 5, 5)[0] == 0


class TestValueOverReplacement:
    def test_replacement_level_deepens_with_league_size(self, pool):
        small = replacement_levels(pool.projection, pool.position, LeagueConfig(teams=10))
        big = replacement_levels(pool.projection, pool.position, LeagueConfig(teams=14))
        assert big["RB"] < small["RB"], "more teams must mean a lower replacement bar"

    def test_vor_is_projection_minus_baseline(self, pool):
        v = value_over_replacement(pool.projection, pool.position, LEAGUE)
        levels = replacement_levels(pool.projection, pool.position, LEAGUE)
        i = 0
        assert v[i] == pytest.approx(pool.projection[i] - levels[pool.position[i]])

    def test_scarce_position_gets_relatively_more_credit(self, pool):
        """Tight end decays fastest in the fixture, so the top TE should
        rank higher by VOR than by raw projection."""
        v = value_over_replacement(pool.projection, pool.position, LEAGUE)
        te = pool.position == "TE"
        best_te = int(np.argmax(np.where(te, pool.projection, -np.inf)))
        rank_by_proj = int((pool.projection > pool.projection[best_te]).sum())
        rank_by_vor = int((v > v[best_te]).sum())
        assert rank_by_vor < rank_by_proj


class TestLineupScoring:
    def test_greedy_lineup_picks_the_best_at_each_slot(self):
        position = np.array(["QB", "QB", "RB", "RB", "RB", "WR", "WR", "TE"])
        points = np.array([10.0, 30.0, 5.0, 20.0, 15.0, 8.0, 12.0, 7.0])
        roster = list(range(8))
        # QB 30 + RB 20 + 15 + WR 12 + 8 + TE 7 + flex (best leftover RB 5) = 97
        assert optimal_lineup_points(roster, points, position, LEAGUE) == pytest.approx(97.0)

    def test_flex_takes_the_best_remaining_eligible(self):
        position = np.array(["QB", "RB", "RB", "RB", "WR", "WR", "TE"])
        points = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
        total = optimal_lineup_points(list(range(7)), points, position, LEAGUE)
        assert total == pytest.approx(10 + 9 + 8 + 6 + 5 + 4 + 7)

    def test_incomplete_roster_does_not_crash(self):
        position = np.array(["QB"])
        points = np.array([10.0])
        assert optimal_lineup_points([0], points, position, LEAGUE) == 10.0


class TestSimulation:
    def test_no_player_is_drafted_twice(self, pool):
        rosters = simulate_draft(
            pool, LEAGUE, [AdpAgent() for _ in range(12)], LEAGUE.roster_size, seed=1
        )
        picks = [p for r in rosters for p in r]
        assert len(picks) == len(set(picks))

    def test_every_team_fills_its_roster(self, pool):
        rosters = simulate_draft(
            pool, LEAGUE, [GreedyVorAgent() for _ in range(12)], LEAGUE.roster_size, seed=1
        )
        assert all(len(r) == LEAGUE.roster_size for r in rosters)

    def test_position_caps_are_respected(self, pool):
        strategies = [GreedyVorAgent() for _ in range(12)]
        rosters = simulate_draft(pool, LEAGUE, strategies, LEAGUE.roster_size, seed=3)
        for r in rosters:
            counts: dict[str, int] = {}
            for i in r:
                counts[pool.position[i]] = counts.get(pool.position[i], 0) + 1
            for pos, cap in POSITION_CAPS.items():
                assert counts.get(pos, 0) <= cap

    def test_agents_cannot_see_realized_points(self, pool):
        """Permuting realized points must not change a single pick.

        This is the test the Phase 5 headline depends on. If `actual`
        influenced any agent, the simulation would report an edge that
        exists only because the agent knew the future.
        """
        strategies = [AdpAgent() for _ in range(12)]
        strategies[4] = GreedyVorAgent()
        strategies[9] = MonteCarloAgent(n_rollouts=6, seed=5)
        before = simulate_draft(pool, LEAGUE, strategies, LEAGUE.roster_size, seed=11)

        shuffled = DraftPool(
            player_id=pool.player_id, name=pool.name, position=pool.position,
            projection=pool.projection,
            actual=np.random.default_rng(99).permutation(pool.actual),
            adp_rank=pool.adp_rank,
        )
        strategies[9] = MonteCarloAgent(n_rollouts=6, seed=5)
        after = simulate_draft(shuffled, LEAGUE, strategies, LEAGUE.roster_size, seed=11)
        assert before == after

    def test_scoring_reads_actuals_not_projections(self, pool):
        rosters = simulate_draft(
            pool, LEAGUE, [AdpAgent() for _ in range(12)], LEAGUE.roster_size, seed=2
        )
        realized = score_rosters(rosters, pool, LEAGUE)
        projected = np.array([
            optimal_lineup_points(r, pool.projection, pool.position, LEAGUE)
            for r in rosters
        ])
        assert not np.allclose(realized, projected)


class TestAgents:
    def test_greedy_takes_the_highest_vor_available(self, pool):
        vor = value_over_replacement(pool.projection, pool.position, LEAGUE)
        available = np.arange(pool.n)
        choice = GreedyVorAgent().pick(
            available=available, counts={}, pool=pool, league=LEAGUE,
            rng=np.random.default_rng(0), vor=vor, my_future_picks=[],
        )
        assert choice == int(np.argmax(vor))

    def test_monte_carlo_is_reproducible(self, pool):
        def draft():
            strategies = [AdpAgent() for _ in range(12)]
            strategies[0] = MonteCarloAgent(n_rollouts=8, seed=42)
            return simulate_draft(pool, LEAGUE, strategies, LEAGUE.roster_size, seed=4)

        assert draft() == draft()

    def test_candidates_span_positions(self, pool):
        """The optimizer must consider the best player at each position.

        Top-N by VOR alone frequently returns N players at one position,
        which would hide the scarcity tradeoff entirely.
        """
        vor = value_over_replacement(pool.projection, pool.position, LEAGUE)
        agent = MonteCarloAgent(n_candidates=8)
        cands = agent._candidates(np.arange(pool.n), pool, vor)
        assert len(set(pool.position[c] for c in cands)) == 4

    def test_monte_carlo_beats_greedy_on_its_own_objective(self, pool):
        """The optimizer should improve the thing it optimizes.

        Scored on projections rather than realized points — realized
        outcomes carry a season of luck and need hundreds of paired
        simulations to resolve (see scripts/eval_draft.py).
        """
        wins = 0
        for slot in range(0, 12, 3):
            results = {}
            for name, agent in (
                ("greedy", GreedyVorAgent()),
                ("mc", MonteCarloAgent(n_rollouts=16, seed=slot)),
            ):
                strategies = [AdpAgent() for _ in range(12)]
                strategies[slot] = agent
                rosters = simulate_draft(
                    pool, LEAGUE, strategies, LEAGUE.roster_size, seed=100 + slot
                )
                results[name] = optimal_lineup_points(
                    rosters[slot], pool.projection, pool.position, LEAGUE
                )
            wins += results["mc"] >= results["greedy"]
        assert wins >= 3, "lookahead should match or beat greedy in most slots"
