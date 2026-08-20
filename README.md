# gridiron

Fantasy football projections trained on 14 seasons of nflverse data, forecast as **distributions** rather than point estimates, evaluated with strict walk-forward backtests.

Most public fantasy tooling consumes third-party projections and averages them. This builds the model underneath, and — more importantly — reports honestly on where it works and where it doesn't.

**Status:** Phases 0–4 complete — data pipeline, scoring engine, baselines, offseason features, and calibrated distributional projections. Draft optimizer is next.

---

## Findings so far

### 1. Opportunity is sticky. Efficiency is noise.

Year-over-year correlation of each metric with itself, 2012–2025, minimum 8 games played:

| Metric | RB | WR |
|---|---|---|
| Target share | **0.65** | **0.74** |
| Air yards share | 0.48 | **0.75** |
| WOPR | 0.65 | 0.75 |
| Carries | 0.64 | — |
| Snap share | 0.63 | 0.63 |
| Fantasy PPG | 0.64 | 0.69 |
| — | | |
| Yards per carry | 0.25 | — |
| Catch rate | 0.15 | 0.40 |
| Yards per target | 0.04 | 0.18 |
| **TD rate per touch** | **0.16** | **0.10** |

A wide receiver's touchdown rate has a year-over-year correlation of **0.10** — statistically almost nothing. His target share is **0.74**. A player who scored 14 touchdowns on 190 touches is not a 14-touchdown player next season; he is a 190-touch player who got lucky.

This is the entire feature strategy in one table: **predict volume, regress efficiency.**

*Caveat, stated because the table invites it:* "carries" for WRs shows a high correlation largely because most receivers have zero carries in consecutive seasons. Correlation of zeros is not signal. Pooled all-position numbers are similarly inflated — position alone explains much of the variance — which is why every number here is reported within position.

### 2. Persistence is a much harder baseline than expected

Walk-forward, 2018–2025. Train on all prior seasons, predict the next one. Restricted to players with at least 6 games in the prior season, to keep deep-bench noise out. Metrics computed within season and position, since that is the comparison a drafter actually makes.

| Model | Position | Spearman | Top-12 hit | MAE |
|---|---|---|---|---|
| **GBM** | QB | **0.587** | 0.550 | **74.1** |
| Persistence | QB | 0.574 | 0.621 | 80.9 |
| **Persistence** | RB | **0.639** | 0.478 | 53.6 |
| GBM | RB | 0.594 | 0.501 | 51.6 |
| **Persistence** | WR | **0.679** | 0.426 | 43.1 |
| GBM | WR | 0.673 | 0.415 | **41.5** |
| **Opportunity** | TE | **0.659** | 0.511 | **29.0** |
| Persistence | TE | 0.655 | 0.479 | 30.5 |
| GBM | TE | 0.609 | 0.508 | 30.5 |

**The gradient boosted model wins on MAE at three of four positions but loses on rank correlation at three of four.** It is better calibrated in magnitude and no better — sometimes worse — at ordering. Simply carrying forward last season's point total beats it at RB, WR and TE.

That is a negative result, and it is reported rather than buried because it is the most informative thing the project has produced so far.

This is a real result, not a bug, and it points directly at the actual problem: **the information that moves projections year over year is offseason information** — free agency, the draft, coaching changes, depth chart competition — and none of it is present in lagged box score statistics. A model fed only last year's stats has no idea a team just drafted a running back in the first round.

That is why human consensus projections beat naive persistence: humans encode the offseason. Closing that gap is the actual work, and the roadmap below is built around it.

### 3. Adding offseason information closes the gap

Finding #2 was a hypothesis about *why* the model lost, so Phase 3 tested it directly: add the things that happen between seasons and nothing else.

- **team change** entering the season (week-1 roster)
- **incoming draft capital at the player's position** — the earliest pick his new team spent on his position in that April's draft
- **new team's prior-season offensive profile**, and the delta from his old team's (pass rate, attempts per game, offensive TDs)
- **head coaching change**
- **week-1 depth chart position**

Same model, same hyperparameters, same walk-forward split. Only the feature set changed:

| Position | Persistence | GBM, lags only | **GBM + offseason** |
|---|---|---|---|
| QB | 0.574 | 0.584 | **0.671** |
| RB | 0.639 | 0.602 | **0.662** |
| TE | 0.655 | 0.602 | **0.671** |
| WR | 0.679 | 0.674 | **0.728** |

*(Spearman rank correlation, walk-forward 2018–2025, within season and position.)*

The model now beats persistence at **all four positions**, and beats its own lag-only version by 0.05–0.09 — a far larger jump than any amount of hyperparameter tuning produced. MAE improves everywhere too. The diagnosis in finding #2 was correct: the missing information was never in the box scores.

Offseason features account for **16% of total model gain** despite being 16 of 90 features. The largest single one is depth chart position, at 6.5% gain — fourth most important feature overall, behind only the three prior-season scoring columns.

**The honest caveat.** Week-1 depth charts publish just before the season, which can be *after* an August fantasy draft. Stripping them out to simulate strict draft-day information:

| Position | Persistence | GBM, strict draft-day |
|---|---|---|
| QB | 0.574 | **0.625** |
| RB | 0.639 | 0.626 |
| TE | 0.655 | 0.630 |
| WR | 0.679 | **0.697** |

Without the depth chart the result is mixed — clear wins at QB and WR, marginal losses at RB and TE. So the headline "beats persistence everywhere" depends on information you may not have on draft day. Both variants ship (`include_depth_chart=False`) and both are reported, because quoting only the better one would misrepresent what the model can do when you actually need it.

### 4. The quantile model was overconfident — three fixes, only the third worked

Predicting p10/p25/p50/p75/p90 season totals directly and checking whether reality lands inside the bands gave **0.605 coverage on p10–p90 against a target of 0.80**, and 0.332 on p25–p75 against 0.50. Badly overconfident.

The cause is structural rather than a tuning problem. Season totals decompose as

```
total = games_played × points_per_game
```

and most of the variance is in the first factor — 31% of fantasy-relevant players appear in 8 or fewer games, and availability has a standard deviation of about 5 games. One model asked to predict season totals has to smear bimodal injury risk into a single conditional distribution, and it hedges toward the middle.

**Fix 1 — model the factors separately and compose them.** Two quantile grids, one over games played and one over points per game, Monte Carlo sampled and multiplied.

**Fix 2 — couple them properly.** Availability and production are not independent: good players both stay on the field and score more. Sampling uses a Gaussian copula whose parameter is estimated from *out-of-sample* PIT values, so it measures the dependence remaining after the features have explained what they can. Fitted ρ ≈ 0.41, stable across seasons.

| Model | p10–p90 coverage | p25–p75 coverage |
|---|---|---|
| Direct season-total quantiles | 0.605 | 0.332 |
| Composite, **independent** | 0.582 | 0.347 |
| Composite, **copula** (ρ≈0.41) | 0.661 | 0.390 |
| Composite, comonotonic (ρ=0.95) | 0.736 | 0.434 |

Two things worth reading off this table. Independent sampling is *worse than doing nothing* — confirming the dependence is real and that ignoring it actively harms the upper tail. And the physically implausible comonotonic assumption scores best of the three, which is itself diagnostic: if cranking dependence past what the data supports keeps improving coverage, the dependence structure was never the whole problem. **The marginals themselves were too narrow.** Gradient boosted quantile regression tends to be overconfident, since predictions get averaged within leaves.

**Fix 3 — conformal calibration.** [Conformalized quantile regression](https://arxiv.org/abs/1905.03222) corrects this empirically rather than parametrically: fit on seasons < N−1, measure on season N−1 how far reality actually fell outside the predicted interval, and widen by that amount. No assumption about error shape required.

| Model | p10–p90 | p25–p75 | Mean p10–p90 width |
|---|---|---|---|
| Composite + copula | 0.651 | 0.384 | 101.6 |
| **+ conformal** | **0.784** | **0.491** | 118.7 |
| *target* | *0.80* | *0.50* | |

Essentially calibrated, at a cost of 17% wider intervals — which is the honest price of intervals that mean what they claim. That tradeoff belongs to the user, not the model, so both are available.

*(The copula row differs slightly between the two tables because the conformal experiment holds out season N−1 for calibration and so trains on one less season.)*

**Caveat on the guarantee.** Split conformal gives finite-sample coverage *under exchangeability*, and NFL seasons are not exchangeable — rule changes, and a 17th game added in 2021. So the guarantee here is approximate. The 0.784 above is measured on held-out seasons rather than assumed from theory, which is the only reason to trust it.

---

## What's here

```
src/gridiron/
  config.py      LeagueConfig + ScoringConfig — nothing hardcodes PPR
  data.py        nflreadpy loaders with parquet caching
  scoring.py     fantasy point engine, validated against nflverse
  features.py    lagged player-season panel + stickiness analysis
  offseason.py   team moves, draft capital, coaching + depth charts
  models.py      GBM point projector + quantile projector
  calibration.py availability x production composition + conformal
  evaluate.py    walk-forward harness, rank metrics, calibration
tests/           33 tests — scoring, leakage, and sampler correctness
scripts/         run_baselines.py, eval_calibration.py, eval_conformal.py
```

### The scoring engine is validated, not assumed

`tests/test_scoring.py` checks the standard and PPR presets against nflverse's own `fantasy_points` and `fantasy_points_ppr` columns across **37,626 player-weeks**. Maximum deviation: **0.0000**. Every number in this repo sits on top of that, so it is tested first.

Scoring and league structure are fully configurable — PPR / half / standard, TE premium, yardage bonuses, team count, superflex. Replacement level for VOR is derived from league settings rather than assumed, because a 10-team league and a 14-team league have very different replacement levels.

---

## Methodology

**Walk-forward only.** To score season N, train on seasons < N. No random splits anywhere. Shuffling a time series trains on the future, and it is the most common flaw in public fantasy models.

**Features are allowlisted, not denylisted.** `models.feature_columns` admits exactly three categories — lagged prior-season stats, age/experience, and explicitly vetted offseason features. Anything else joined onto the panel is invisible to the model, and `tests/test_leakage.py::test_no_unvetted_features` fails if someone adds a feature without vetting it. A denylist would silently admit new columns, which is exactly how leakage gets into projects like this.

**Leakage is tested, not asserted.** `tests/test_leakage.py` checks that lag-1 features equal the prior season's value (catching off-by-one join errors, the most damaging and least visible bug available here), that draft capital never comes from a future draft, that team assignment comes from the week-1 roster rather than a later week (which would encode surviving cuts or a midseason trade), and that no feature correlates above 0.95 with the target.

**Rank metrics lead.** Spearman and top-k hit rate are the headline; MAE and RMSE are secondary. You draft an ordering, not a point total.

**Metrics are computed within season and position.** Pooling positions inflates rank correlation dramatically — quarterbacks outscore everyone, so any model that knows what a quarterback is looks good on a pooled metric.

---

## Setup

```bash
pip install -e ".[dev]"
python scripts/run_baselines.py      # downloads + caches ~14 seasons, then reproduces the tables
python scripts/eval_conformal.py     # Phase 4 calibration results

pytest -m "not slow"                 # fast suite, ~30s
pytest                               # full suite incl. calibration, ~7min
```

Data comes from **`nflreadpy`**. Note that the widely-tutorialized `nfl_data_py` package is [officially deprecated](https://github.com/nflverse/nfl_data_py) in favour of it. `nflreadpy` returns Polars frames rather than pandas.

One caveat: `load_ff_rankings` pulls FantasyPros ECR/ADP from DynastyProcess via a direct GitHub URL, which returns 403 on some sandboxed or proxied networks. Everything else comes from nflverse-data releases and is unaffected — a failure there costs you the consensus baseline, not the pipeline.

---

## Roadmap

**~~Phase 3 — close the offseason information gap.~~ Done** — see finding #3.

**Phase 3b — beat the real baseline.** Wire up the ESPN/CBS/NFL consensus via `load_ff_rankings` and compare against it directly. Persistence is the floor; consensus is the target.

**~~Phase 4 — fix calibration.~~ Done** — see finding #4. Coverage went 0.605 → 0.784 against a 0.80 target.

**Phase 5 — the draft optimizer.** Now unblocked, and it needs the calibrated distributions from Phase 4: a risk-aware drafter optimizing for p10 (safe roster) or p90 (upside roster) is only meaningful if those quantiles are honest.  Greedy VOR is myopic: at pick 15 you want the pick that maximizes expected final *starting lineup* value given who will survive to picks 34 and 39, not the highest VOR available now. Monte Carlo rollouts with ADP-based opponent models, evaluated over 10,000 simulated drafts against greedy-VOR and ADP-following agents.

**Phase 6 — ship it.** CLI, live draft mode, Sleeper API sync, deployed demo.

### Known limitations

- **Rookies are excluded.** No prior NFL season means no lagged features. They need a separate model built on draft capital and college production — a genuinely harder problem, deliberately out of scope for v1 rather than papered over.
- **Kickers and team defenses are not modeled.** Both are close to unpredictable at the season level. Pretending otherwise would be dishonest.
- Snap counts begin in 2012, which sets the practical start of the panel.
