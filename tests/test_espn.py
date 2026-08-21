"""Tests for external-baseline ingestion and comparison.

The parser has to survive whatever a copied projections table turns into —
tabs, runs of spaces, position cells like "RB, FLEX", header junk. And the
join has to survive name formatting differences between sources, because a
silently dropped join is indistinguishable from "the model has no opinion
on this player", which is the worst possible failure mode for a comparison.
"""

from __future__ import annotations

import polars as pl
import pytest

from gridiron.espn import (
    agreement,
    biggest_disagreements,
    compare,
    load_external,
    parse_cheat_sheet,
    parse_pasted,
)

# A cheat sheet packs several position columns onto every text line, so one
# line interleaves entries from different groups. Each entry is
# self-describing: pos-rank. (overall) Name, TEAM $value bye
CHEAT_SHEET = """
2026 ESPN Fantasy Football Draft Kit
PPR League Cheat Sheet
Quarterbacks Running Backs Wide Receivers
1. (36) Josh Allen, BUF $22 7 1. (1) Jahmyr Gibbs, DET $57 6 1. (3) Ja'Marr Chase, CIN $56 6
2. (55) Jayden Daniels, WSH $10 7 15. (30) Josh Jacobs, GB $27 11 2. (4) Puka Nacua, LAR $55 11
"""

PASTED = """
Ja'Marr Chase Cin WR    17    285.4
Bijan Robinson Atl RB, FLEX   17   271.9
Jahmyr Gibbs\tDet\tRB\t17\t262.3
Josh Jacobs GB RB   16   210.1
Amon-Ra St. Brown Det WR 17 244.8
Marvin Harrison Jr. Ari WR   17   198.2
PLAYER  TEAM  POS  GP  PTS
Totals 123
"""


@pytest.fixture
def external() -> pl.DataFrame:
    return parse_pasted(PASTED)


@pytest.fixture
def ours() -> pl.DataFrame:
    return pl.DataFrame({
        "player_display_name": [
            "Ja'Marr Chase", "Bijan Robinson", "Jahmyr Gibbs",
            "Josh Jacobs", "Amon-Ra St. Brown", "Marvin Harrison Jr.",
        ],
        "position": ["WR", "RB", "RB", "RB", "WR", "WR"],
        "team": ["CIN", "ATL", "DET", "GB", "DET", "ARI"],
        "projected_points": [190.2, 181.6, 206.1, 229.5, 222.7, 93.0],
    })


class TestParsing:
    def test_parses_every_player_row(self, external):
        assert external.height == 6

    def test_skips_header_and_total_rows(self, external):
        names = external["player_display_name"].to_list()
        assert "PLAYER" not in names and "Totals" not in names

    def test_handles_tabs_and_multiple_spaces(self, external):
        gibbs = external.filter(pl.col("player_display_name") == "Jahmyr Gibbs")
        assert gibbs.height == 1
        assert gibbs["external_points"][0] == pytest.approx(262.3)

    def test_handles_multi_position_eligibility(self, external):
        """ESPN writes flex-eligible players as 'RB, FLEX'."""
        bijan = external.filter(pl.col("player_display_name") == "Bijan Robinson")
        assert bijan["position"][0] == "RB"

    def test_keeps_apostrophes_and_suffixes_in_names(self, external):
        names = set(external["player_display_name"])
        assert "Ja'Marr Chase" in names
        assert "Marvin Harrison Jr." in names

    def test_takes_the_last_number_as_points(self, external):
        """Rows carry games played before the projection."""
        chase = external.filter(pl.col("player_display_name") == "Ja'Marr Chase")
        assert chase["external_points"][0] == pytest.approx(285.4)

    def test_team_abbreviations_are_normalised(self):
        df = parse_pasted("Jayden Daniels Wsh QB 17 320.5")
        assert df["team"][0] == "WAS"

    def test_empty_input_raises_rather_than_returning_nothing(self):
        with pytest.raises(ValueError):
            parse_pasted("no player rows here\njust prose")

    def test_csv_column_aliases_are_recognised(self, tmp_path):
        path = tmp_path / "espn.csv"
        path.write_text("Player,Pos,Team,FPTS\nJoe Burrow,QB,CIN,310.4\n")
        df = load_external(path)
        assert df["external_points"][0] == pytest.approx(310.4)
        assert df["source"][0] == "espn"

    def test_csv_missing_points_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("Player,Pos\nJoe Burrow,QB\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_external(path)


class TestComparison:
    def test_all_players_match_across_sources(self, ours, external):
        """Name normalisation must survive apostrophes, periods and suffixes."""
        c = compare(ours, external)
        assert c.height == 6, "name join dropped players"

    def test_ranks_are_computed_within_position(self, ours, external):
        c = compare(ours, external)
        rb = c.filter(pl.col("position") == "RB")
        assert sorted(rb["our_rank"].to_list()) == [1.0, 2.0, 3.0]

    def test_rank_delta_sign_is_readable(self, ours, external):
        """Positive delta = we rank the player higher than they do."""
        c = compare(ours, external)
        jacobs = c.filter(pl.col("player_display_name") == "Josh Jacobs")
        assert jacobs["our_rank"][0] == 1.0
        assert jacobs["their_rank"][0] == 3.0
        assert jacobs["rank_delta"][0] == 2.0

    def test_agreement_reports_per_position(self, ours, external):
        a = agreement(compare(ours, external), min_players=3)
        assert set(a["position"]) == {"RB", "WR"}
        assert a["spearman"].is_finite().all()

    def test_agreement_returns_typed_empty_frame_when_too_few_players(
        self, ours, external
    ):
        """An unhelpful schema error at the end of a long pipeline is worse
        than an obviously empty result."""
        a = agreement(compare(ours, external), min_players=50)
        assert a.height == 0
        assert "spearman" in a.columns

    def test_biggest_disagreements_surfaces_both_directions(self, ours, external):
        d = biggest_disagreements(compare(ours, external), n=2)
        deltas = d["rank_delta"].to_list()
        assert max(deltas) > 0 and min(deltas) < 0

    def test_unmatched_external_players_are_dropped_not_zero_filled(self, ours):
        """A player we do not project must not appear as a zero.

        Zero-filling would make the model look absurdly wrong on anyone it
        legitimately has no row for — rookies, in particular.
        """
        extra = parse_pasted(PASTED + "\nSome Rookie Chi RB 17 180.0")
        c = compare(ours, extra)
        assert "Some Rookie" not in c["player_display_name"].to_list()
        assert c.height == 6


class TestCheatSheet:
    def test_parses_entries_from_interleaved_columns(self):
        """One text line carries a QB, an RB and a WR — all must be found."""
        d = parse_cheat_sheet(CHEAT_SHEET)
        assert d.height == 6

    def test_uses_overall_rank_not_positional_rank(self):
        """'15. (30) Josh Jacobs' is RB15 but the 30th player overall.

        Taking the positional rank would make every position look
        identically ranked and silently destroy the comparison.
        """
        d = parse_cheat_sheet(CHEAT_SHEET)
        jacobs = d.filter(pl.col("player_display_name") == "Josh Jacobs")
        assert jacobs["external_rank"][0] == 30

    def test_captures_auction_value_and_bye(self):
        d = parse_cheat_sheet(CHEAT_SHEET)
        gibbs = d.filter(pl.col("player_display_name") == "Jahmyr Gibbs")
        assert gibbs["auction_value"][0] == 57.0
        assert gibbs["bye"][0] == 6

    def test_ignores_title_and_column_headers(self):
        names = parse_cheat_sheet(CHEAT_SHEET)["player_display_name"].to_list()
        assert not any("Cheat" in n or "Quarterbacks" in n for n in names)

    def test_team_abbreviations_are_normalised(self):
        d = parse_cheat_sheet(CHEAT_SHEET)
        teams = set(d["team"])
        assert "WAS" in teams and "WSH" not in teams
        assert "LA" in teams and "LAR" not in teams

    def test_output_is_sorted_by_overall_rank(self):
        ranks = parse_cheat_sheet(CHEAT_SHEET)["external_rank"].to_list()
        assert ranks == sorted(ranks)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="no cheat-sheet entries"):
            parse_cheat_sheet("just some prose with no entries")

    def test_compare_uses_supplied_rank_when_points_absent(self):
        """A cheat sheet has no point totals, so `compare` must rank from
        the supplied overall rank rather than erroring on a missing column."""
        external = parse_cheat_sheet(CHEAT_SHEET)
        ours = pl.DataFrame({
            "player_display_name": ["Jahmyr Gibbs", "Josh Jacobs", "Ja'Marr Chase"],
            "position": ["RB", "RB", "WR"],
            "team": ["DET", "GB", "CIN"],
            "projected_points": [246.9, 220.3, 240.0],
        })
        c = compare(ours, external)
        assert c.height == 3
        assert "points_delta" not in c.columns
        rb = c.filter(pl.col("position") == "RB").sort("their_rank")
        assert rb["player_display_name"].to_list() == ["Jahmyr Gibbs", "Josh Jacobs"]
