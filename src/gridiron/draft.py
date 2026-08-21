"""Draft simulation and the sequential draft optimizer.

Value Over Replacement ranks players by how much better they are than the
best player you could get for free at their position. It is the right
first idea, and it is what most public draft tools stop at. But drafting
greedily by VOR is **myopic**: at pick 15 the question is not "who has the
highest VOR right now", it is "which pick maximizes the value of my final
starting lineup, given who will still be on the board at picks 34, 39 and
58".

Those differ whenever positional scarcity curves differ. Taking the elite
tight end now can be worth less than taking a running back and accepting a
slightly worse tight end two rounds later, because the drop-off from TE1
to TE4 may be shallower than the drop-off from RB12 to RB30.

:class:`MonteCarloAgent` optimizes the actual objective by rolling the
draft forward. Opponents are modelled as drafting near ADP with noise, and
each candidate first pick is scored by the expected final starting-lineup
value it leads to.

Note on scope: kickers and team defenses are excluded throughout, matching
the projection models. Neither is meaningfully predictable at the season
level and including them would add noise without adding signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .config import LeagueConfig

POSITIONS = ("QB", "RB", "WR", "TE")

# Soft caps: real drafters do not take six quarterbacks, and without caps
# a greedy agent will happily hoard one position once VOR says so.
POSITION_CAPS = {"QB": 2, "RB": 6, "WR": 6, "TE": 2}


@dataclass
class DraftPool:
    """The board: everyone available, with projections and realized results.

    ``projection`` is what agents may see. ``actual`` is realized season
    points and is used **only** for scoring completed rosters — no agent
    ever reads it.
    """

    player_id: np.ndarray
    name: np.ndarray
    position: np.ndarray
    projection: np.ndarray
    actual: np.ndarray
    adp_rank: np.ndarray
    pos_idx: np.ndarray = None
    # (n_players, n_scenarios) draws from the calibrated composite model.
    # Column s is one joint draw of the season. Optional: agents fall back
    # to point projections when this is absent.
    samples: np.ndarray = None

    def __post_init__(self):
        if self.pos_idx is None:
            self.pos_idx = np.array([POS_INDEX[p] for p in self.position])

    @property
    def n(self) -> int:
        return len(self.player_id)

    @classmethod
    def from_frame(
        cls,
        df: pl.DataFrame,
        projection_col: str = "y_hat",
        actual_col: str = "y_points",
    ) -> "DraftPool":
        df = df.filter(pl.col("position").is_in(list(POSITIONS))).drop_nulls(
            [projection_col, actual_col]
        )
        proj = df[projection_col].to_numpy().astype(np.float64)
        return cls(
            player_id=df["player_id"].to_numpy(),
            name=df["player_display_name"].to_numpy(),
            position=df["position"].to_numpy(),
            projection=proj,
            actual=df[actual_col].to_numpy().astype(np.float64),
            adp_rank=_proxy_adp(proj, df["position"].to_numpy(), df),
        )


def _proxy_adp(projection: np.ndarray, position: np.ndarray, df: pl.DataFrame) -> np.ndarray:
    """Stand-in for public Average Draft Position.

    Real ADP comes from FantasyPros via DynastyProcess (`load_ff_rankings`),
    which is unreachable from some networks. When it is unavailable this
    falls back to ranking by *last season's* points — which is roughly what
    a casual drafter uses, and correlates strongly with published ADP.

    This is a documented approximation, not a claim of equivalence. It
    matters for one comparison only (agents vs. the public field); the
    headline greedy-VOR vs. Monte-Carlo comparison holds projections and
    opponent behaviour fixed, so it is unaffected.
    """
    if "fantasy_points_total_lag1" in df.columns:
        basis = df["fantasy_points_total_lag1"].fill_null(0.0).to_numpy()
    else:
        basis = projection
    # Rank by VOR on the basis so positions interleave the way a real
    # board does, rather than listing every QB first.
    league = LeagueConfig()
    v = value_over_replacement(basis, position, league)
    return np.argsort(np.argsort(-v)).astype(np.float64)


def replacement_levels(
    projection: np.ndarray, position: np.ndarray, league: LeagueConfig
) -> dict[str, float]:
    """Projected points of the first *undrafted-as-a-starter* player.

    Derived from league settings rather than assumed. A 10-team league has
    a much shallower replacement level than a 14-team league, which is why
    VOR computed with hardcoded settings is wrong for most people.

    Flex slots are distributed across eligible positions in proportion to
    how often each actually fills a flex — a simplification, but a far
    better one than ignoring flex entirely.
    """
    flex_total = league.flex * league.teams + league.superflex * league.teams
    flex_weights = {"RB": 0.45, "WR": 0.45, "TE": 0.10}

    levels = {}
    for pos in POSITIONS:
        mask = position == pos
        if not mask.any():
            levels[pos] = 0.0
            continue
        starters = league.base_starters(pos)
        if pos in league.flex_positions and flex_total:
            starters += int(round(flex_total * flex_weights.get(pos, 0.0)))
        ranked = np.sort(projection[mask])[::-1]
        idx = min(starters, len(ranked) - 1)
        levels[pos] = float(ranked[idx])
    return levels


def value_over_replacement(
    projection: np.ndarray, position: np.ndarray, league: LeagueConfig
) -> np.ndarray:
    levels = replacement_levels(projection, position, league)
    baseline = np.array([levels.get(p, 0.0) for p in position])
    return projection - baseline


def expected_lineup_points(
    roster: list[int],
    samples: np.ndarray,
    pos_idx: np.ndarray,
    league: LeagueConfig,
    objective: str = "mean",
) -> float:
    """Roster value under the *distribution* of season outcomes.

    This is not the same as ``optimal_lineup_points`` fed a point
    projection, and the difference is the whole point of Phase 5b.

    A starting lineup is an order statistic — you play your best QB, your
    best two RBs, and so on. Order statistics are **convex**, so by
    Jensen's inequality

        E[lineup(X)] >= lineup(E[X])

    Scoring a roster with point projections therefore systematically
    *understates* its value, and understates it unevenly: it penalises
    exactly the high-variance players whose upside a lineup can capture
    while their downside gets benched. Depth at a position has option
    value that a point estimate cannot express at all.

    ``samples`` is (n_players, n_scenarios); column *s* is one joint draw
    of the season. Vectorised across scenarios because this runs inside
    every Monte Carlo rollout.
    """
    n_scen = samples.shape[1]
    total = np.zeros(n_scen)
    leftovers: list[np.ndarray] = []

    slots_by_pos = {"QB": league.qb, "RB": league.rb, "WR": league.wr, "TE": league.te}
    for pos, slots in slots_by_pos.items():
        code = POS_INDEX[pos]
        rows = [i for i in roster if pos_idx[i] == code]
        if not rows:
            continue
        block = -np.sort(-samples[rows], axis=0)  # descending within position
        total += block[:slots].sum(axis=0)
        if pos in league.flex_positions and block.shape[0] > slots:
            leftovers.append(block[slots:])

    if league.flex and leftovers:
        pooled = -np.sort(-np.vstack(leftovers), axis=0)
        total += pooled[: league.flex].sum(axis=0)

    if objective == "floor":
        return float(np.percentile(total, 25))
    if objective == "ceiling":
        return float(np.percentile(total, 75))
    return float(total.mean())


def optimal_lineup_points(
    roster: list[int], points: np.ndarray, position: np.ndarray, league: LeagueConfig
) -> float:
    """Best legal starting lineup from a roster.

    Greedy is exact for this structure: fill the dedicated slots with the
    best player at each position, then give the flex to the best remaining
    flex-eligible player. With a single flex tier there is no case where
    taking a worse player at a dedicated slot frees up a better total.
    """
    by_pos: dict[str, list[float]] = {p: [] for p in POSITIONS}
    for i in roster:
        by_pos[position[i]].append(points[i])
    for p in by_pos:
        by_pos[p].sort(reverse=True)

    total = 0.0
    leftovers: list[float] = []
    for pos, slots in (("QB", league.qb), ("RB", league.rb),
                       ("WR", league.wr), ("TE", league.te)):
        taken = by_pos[pos][:slots]
        total += sum(taken)
        leftovers.extend(by_pos[pos][slots:] if pos in league.flex_positions else [])

    leftovers.sort(reverse=True)
    total += sum(leftovers[: league.flex])
    return float(total)


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------

POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}
CAP_ARRAY = np.array([POSITION_CAPS[p] for p in POSITIONS])


def _eligible(
    available: np.ndarray, counts: dict[str, int], pos_idx: np.ndarray
) -> np.ndarray:
    """Mask out positions the roster is already full at.

    Vectorised over integer position codes — this runs inside every Monte
    Carlo rollout, so the obvious Python loop dominated the profile.
    """
    filled = np.array([counts.get(p, 0) for p in POSITIONS])
    full = filled >= CAP_ARRAY
    if not full.any():
        return np.ones(len(available), dtype=bool)
    return ~full[pos_idx[available]]


@dataclass
class AdpAgent:
    """Drafts near ADP with Gumbel noise. Models the rest of the league."""

    sigma: float = 6.0

    def pick(self, available, counts, pool, league, rng, **_) -> int:
        ok = _eligible(available, counts, pool.pos_idx)
        cand = available[ok] if ok.any() else available
        noise = rng.gumbel(0.0, self.sigma, size=len(cand))
        return int(cand[np.argmin(pool.adp_rank[cand] + noise)])


@dataclass
class GreedyVorAgent:
    """Always takes the highest Value Over Replacement available.

    The standard approach, and the baseline the optimizer has to beat.
    """

    def pick(self, available, counts, pool, league, rng, vor=None, **_) -> int:
        ok = _eligible(available, counts, pool.pos_idx)
        cand = available[ok] if ok.any() else available
        return int(cand[np.argmax(vor[cand])])


@dataclass
class MonteCarloAgent:
    """Maximizes expected final starting-lineup value via draft rollouts.

    For each of the top ``n_candidates`` picks by VOR, simulate the rest of
    the draft ``n_rollouts`` times and keep the candidate with the best mean
    outcome.

    Opponents are not simulated pick by pick. Instead each rollout draws a
    noisy ADP ordering once and treats the players ahead of you in that
    ordering as gone by your next turn. This is both far cheaper and a
    reasonable model of the thing that actually matters — *who survives to
    your next pick* — rather than which specific rival took whom.
    """

    n_candidates: int = 8
    n_rollouts: int = 24
    sigma: float = 6.0
    objective: str = "mean"  # "mean" | "floor" | "ceiling"
    use_distribution: bool = False
    n_scenarios: int = 120
    seed: int = 12345
    _rng: np.random.Generator = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def pick(self, available, counts, pool, league, rng,
             vor=None, my_future_picks=None, my_roster=None, **_) -> int:
        ok = _eligible(available, counts, pool.pos_idx)
        cand_all = available[ok] if ok.any() else available
        if len(cand_all) == 0:
            return int(available[0])

        candidates = self._candidates(cand_all, pool, vor)
        if my_future_picks is None or len(my_future_picks) == 0:
            return int(candidates[0])

        scores = np.empty(len(candidates))
        for c, first in enumerate(candidates):
            totals = np.empty(self.n_rollouts)
            for r in range(self.n_rollouts):
                totals[r] = self._rollout(
                    first, available, counts, pool, league, self._rng, vor,
                    my_future_picks, my_roster or [],
                )
            if self.use_distribution:
                # _rollout already applied the objective to the scenario
                # distribution; averaging across rollouts marginalises over
                # opponent behaviour, which is a separate source of noise.
                scores[c] = totals.mean()
            elif self.objective == "floor":
                scores[c] = np.percentile(totals, 25)
            elif self.objective == "ceiling":
                scores[c] = np.percentile(totals, 75)
            else:
                scores[c] = totals.mean()
        return int(candidates[int(np.argmax(scores))])

    def _candidates(self, cand_all, pool, vor) -> np.ndarray:
        """Best available at each position, then fill by overall VOR.

        Taking simply the top-N by VOR frequently returns N players at the
        same position, which hides exactly the tradeoff this agent exists
        to evaluate: spend the pick on a scarce position now, or take the
        deeper position and come back for the scarce one later.
        """
        picked: list[int] = []
        for pos_code in range(len(POSITIONS)):
            at_pos = cand_all[pool.pos_idx[cand_all] == pos_code]
            if len(at_pos):
                picked.append(int(at_pos[np.argmax(vor[at_pos])]))
        remaining = np.array([c for c in cand_all if c not in set(picked)])
        if len(remaining) and len(picked) < self.n_candidates:
            extra = remaining[np.argsort(-vor[remaining])[: self.n_candidates - len(picked)]]
            picked.extend(int(x) for x in extra)
        return np.array(picked, dtype=int)

    def _rollout(self, first, available, counts, pool, league, rng, vor,
                 my_future_picks, my_roster) -> float:
        # Start from the roster already drafted. Scoring only the picks
        # from here forward misvalues every candidate: an agent holding
        # two elite running backs would count a third as a starter.
        roster = list(my_roster) + [first]
        local_counts = dict(counts)
        local_counts[pool.position[first]] = local_counts.get(pool.position[first], 0) + 1

        remaining = available[available != first]
        # One noisy ordering per rollout: lower score = taken sooner.
        board = pool.adp_rank[remaining] + rng.gumbel(0.0, self.sigma, size=len(remaining))
        order = np.argsort(board)
        remaining = remaining[order]

        taken = 0
        for gap in my_future_picks:
            # `gap` opponents pick before my next turn.
            taken += gap
            survivors = remaining[taken:]
            if len(survivors) == 0:
                break
            ok = _eligible(survivors, local_counts, pool.pos_idx)
            pickable = survivors[ok] if ok.any() else survivors
            choice = int(pickable[np.argmax(vor[pickable])])
            roster.append(choice)
            local_counts[pool.position[choice]] = (
                local_counts.get(pool.position[choice], 0) + 1
            )
            remaining = remaining[remaining != choice]

        if self.use_distribution and pool.samples is not None:
            return expected_lineup_points(
                roster, pool.samples[:, : self.n_scenarios], pool.pos_idx,
                league, objective=self.objective,
            )
        return optimal_lineup_points(roster, pool.projection, pool.position, league)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def snake_order(teams: int, rounds: int) -> list[int]:
    order = []
    for r in range(rounds):
        seq = range(teams) if r % 2 == 0 else range(teams - 1, -1, -1)
        order.extend(seq)
    return order


def _future_gaps(order: list[int], team: int, position_in_order: int) -> list[int]:
    """How many opponents pick between each of this team's remaining turns."""
    mine = [i for i, t in enumerate(order) if t == team and i > position_in_order]
    gaps = []
    prev = position_in_order
    for m in mine:
        gaps.append(m - prev - 1)
        prev = m
    return gaps


def simulate_draft(
    pool: DraftPool,
    league: LeagueConfig,
    strategies: list,
    rounds: int,
    seed: int = 0,
) -> list[list[int]]:
    """Run one snake draft. Returns each team's roster as player indices."""
    rng = np.random.default_rng(seed)
    order = snake_order(league.teams, rounds)

    available = np.arange(pool.n)
    rosters: list[list[int]] = [[] for _ in range(league.teams)]
    counts: list[dict[str, int]] = [dict() for _ in range(league.teams)]
    vor = value_over_replacement(pool.projection, pool.position, league)

    for slot, team in enumerate(order):
        if len(available) == 0:
            break
        choice = strategies[team].pick(
            available=available,
            counts=counts[team],
            pool=pool,
            league=league,
            rng=rng,
            vor=vor,
            my_future_picks=_future_gaps(order, team, slot),
            my_roster=rosters[team],
        )
        rosters[team].append(choice)
        counts[team][pool.position[choice]] = (
            counts[team].get(pool.position[choice], 0) + 1
        )
        available = available[available != choice]

    return rosters


def score_rosters(
    rosters: list[list[int]], pool: DraftPool, league: LeagueConfig
) -> np.ndarray:
    """Realized starting-lineup points for each team.

    This is the only place ``pool.actual`` is read. Agents never see it.
    """
    return np.array([
        optimal_lineup_points(r, pool.actual, pool.position, league) for r in rosters
    ])
