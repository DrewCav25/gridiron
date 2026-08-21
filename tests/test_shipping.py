"""Tests for the shipped tooling: Sleeper sync, forward projections, report, CLI.

The forward-projection path is the one that needs watching. Every other
part of this project predicts a season that already happened, where a
leak shows up as an implausibly good backtest. Here there is no outcome to
leak *from* — but there is a subtler failure: silently projecting from
stale or absent data and producing confident nonsense. These tests check
the shape and provenance of what comes out.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from gridiron.cli import build_parser
from gridiron.config import ScoringConfig
from gridiron.data import LAST_COMPLETED_SEASON
from gridiron.report import write_report
from gridiron.sleeper import SCORING_MAP, describe, parse_league, parse_scoring

UPCOMING = LAST_COMPLETED_SEASON + 1


# A recorded Sleeper payload. Tests run against this rather than the live
# API so the suite stays offline and deterministic.
SLEEPER_PAYLOAD = {
    "total_rosters": 12,
    "roster_positions": [
        "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF",
        "BN", "BN", "BN", "BN", "BN", "BN",
    ],
    "scoring_settings": {
        "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0,
        "rush_yd": 0.1, "rush_td": 6.0,
        "rec_yd": 0.1, "rec_td": 6.0, "rec": 0.5,
        "fum_lost": -2.0,
    },
}


class TestSleeperParsing:
    def test_reads_scoring_from_the_payload(self):
        s = parse_scoring(SLEEPER_PAYLOAD)
        assert s.reception == 0.5
        assert s.name == "half_ppr"
        assert s.pass_td == 4.0
        assert s.rec_td == 6.0

    def test_reads_roster_structure(self):
        lg = parse_league(SLEEPER_PAYLOAD)
        assert lg.teams == 12
        assert (lg.qb, lg.rb, lg.wr, lg.te, lg.flex) == (1, 2, 2, 1, 1)
        assert lg.k == 1 and lg.dst == 1
        assert lg.bench == 6
        assert lg.roster_size == 15

    def test_full_ppr_and_standard_are_named_correctly(self):
        for value, name in ((1.0, "ppr"), (0.0, "standard")):
            payload = dict(SLEEPER_PAYLOAD)
            payload["scoring_settings"] = dict(
                SLEEPER_PAYLOAD["scoring_settings"], rec=value
            )
            assert parse_scoring(payload).name == name

    def test_te_premium_is_picked_up(self):
        payload = dict(SLEEPER_PAYLOAD)
        payload["scoring_settings"] = dict(
            SLEEPER_PAYLOAD["scoring_settings"], rec=1.0, bonus_rec_te=0.5
        )
        assert parse_scoring(payload).te_reception_bonus == 0.5

    def test_superflex_is_distinguished_from_flex(self):
        payload = dict(SLEEPER_PAYLOAD)
        payload["roster_positions"] = ["QB", "SUPER_FLEX", "RB", "WR", "BN"]
        lg = parse_league(payload)
        assert lg.superflex == 1 and lg.flex == 0
        assert lg.flex_slots("QB") == lg.teams

    def test_unknown_keys_do_not_crash(self):
        """Sleeper exposes far more keys than we map, including IDP."""
        payload = dict(SLEEPER_PAYLOAD)
        payload["scoring_settings"] = dict(
            SLEEPER_PAYLOAD["scoring_settings"], idp_tkl=1.0, def_st_ff=1.0
        )
        assert parse_scoring(payload).reception == 0.5

    def test_empty_payload_falls_back_to_defaults(self):
        lg = parse_league({})
        assert lg.teams == 12
        assert lg.roster_size > 0

    def test_scoring_map_targets_real_config_fields(self):
        fields = set(ScoringConfig.__dataclass_fields__)
        assert set(SCORING_MAP.values()) <= fields

    def test_describe_is_readable(self):
        text = describe(parse_league(SLEEPER_PAYLOAD))
        assert "12-team" in text and "half_ppr" in text


@pytest.mark.slow
class TestForwardProjections:
    @pytest.fixture(scope="class")
    def projections(cls) -> pl.DataFrame:
        from gridiron.predict import project_season
        return project_season(UPCOMING)

    def test_projects_a_plausible_number_of_players(self, projections):
        assert 200 < projections.height < 1200

    def test_every_position_is_represented(self, projections):
        assert set(projections["position"].unique()) == {"QB", "RB", "WR", "TE"}

    def test_projections_are_positive_and_bounded(self, projections):
        """Caught a real bug: unbounded regression extrapolated below zero
        for deep-bench players, putting nonsense at the bottom of the board."""
        p = projections["projected_points"]
        assert p.min() >= 0, "negative season projection is meaningless"
        assert p.max() < 600, "no one scores 600 fantasy points in a season"

    def test_no_outcome_columns_leak_into_the_output(self, projections):
        """There is no outcome for an unplayed season — if a y_ column
        appears here, the panel was built from the wrong source."""
        assert not [c for c in projections.columns if c.startswith("y_")]

    def test_every_player_is_on_a_current_roster(self, projections):
        assert projections["team"].null_count() == 0
        assert projections["team"].n_unique() >= 30

    def test_projections_regress_toward_the_mean(self, projections):
        """Last season's top scorers should be projected lower.

        Regression to the mean is the core modelling claim (finding #1).
        If projections simply mirrored last season, the model would be
        adding nothing over the persistence baseline.
        """
        top = projections.sort("last_season_points", descending=True).head(25)
        assert (top["projected_points"] < top["last_season_points"]).sum() >= 15

    def test_season_column_is_the_upcoming_season(self, projections):
        assert projections["season"].unique().to_list() == [UPCOMING]


@pytest.mark.slow
class TestReport:
    def test_report_is_self_contained(self, tmp_path):
        from gridiron.predict import project_season

        proj = project_season(UPCOMING)
        path = write_report(proj, UPCOMING, ScoringConfig.half_ppr(),
                            tmp_path / "index.html")
        html = path.read_text(encoding="utf-8")

        assert html.startswith("<!doctype html>")
        # No external requests: GitHub Pages must render it with no network.
        for forbidden in ("http://", "cdn.", "<script src=", "<link rel=\"stylesheet\""):
            assert forbidden not in html, f"external dependency: {forbidden}"

    def test_report_embeds_every_projection(self, tmp_path):
        import re
        from gridiron.predict import project_season

        proj = project_season(UPCOMING)
        path = write_report(proj, UPCOMING, ScoringConfig.half_ppr(),
                            tmp_path / "index.html")
        blob = re.search(r"const DATA = (\[.*?\]);",
                         path.read_text(encoding="utf-8"), re.S)
        assert blob, "projection data not embedded"
        assert len(json.loads(blob.group(1))) == proj.height

    def test_report_supports_both_colour_schemes(self, tmp_path):
        from gridiron.predict import project_season

        proj = project_season(UPCOMING)
        path = write_report(proj, UPCOMING, ScoringConfig.half_ppr(),
                            tmp_path / "index.html")
        html = path.read_text(encoding="utf-8")
        assert "prefers-color-scheme: dark" in html


class TestCli:
    def test_every_subcommand_parses(self):
        parser = build_parser()
        for argv in (
            ["project", "--season", "2026"],
            ["draft", "--season", "2026", "--teams", "10", "--pick", "4"],
            ["sleeper", "--league-id", "123"],
            ["export", "--season", "2026", "--out", "docs/index.html"],
        ):
            args = parser.parse_args(argv)
            assert callable(args.func)

    def test_scoring_choices_are_validated(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["project", "--scoring", "nonsense"])

    def test_depth_chart_defaults_off_for_draft_day_realism(self):
        args = build_parser().parse_args(["project"])
        assert args.depth_chart is False

    def test_sleeper_requires_a_league_id(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["sleeper"])
