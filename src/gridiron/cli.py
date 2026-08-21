"""Command line interface.

    gridiron project --season 2026
    gridiron draft   --season 2026 --pick 7
    gridiron sleeper --league-id 123456789012345678
    gridiron export  --season 2026 --out docs/index.html

``draft`` is the one that matters on draft day: it holds the board, takes
players off it as they go, and recommends your pick with the reasoning
shown rather than just a number.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import polars as pl

from .config import LeagueConfig, ScoringConfig
from .draft import (
    POSITIONS,
    POSITION_CAPS,
    optimal_lineup_points,
    replacement_levels,
    value_over_replacement,
)
from .predict import project_season

SCORING_PRESETS = {
    "standard": ScoringConfig.standard,
    "half_ppr": ScoringConfig.half_ppr,
    "ppr": ScoringConfig.ppr,
}


# --------------------------------------------------------------------------
# project
# --------------------------------------------------------------------------

def cmd_project(args: argparse.Namespace) -> int:
    scoring = SCORING_PRESETS[args.scoring]()
    proj = project_season(
        args.season, scoring=scoring, include_depth_chart=args.depth_chart
    )

    if args.position:
        proj = proj.filter(pl.col("position").is_in([p.upper() for p in args.position]))

    pl.Config.set_tbl_rows(args.top)
    pl.Config.set_fmt_str_lengths(24)
    print(
        proj.select(
            "player_display_name", "position", "team",
            "last_season_points", "projected_points",
        ).head(args.top)
    )

    if args.out:
        proj.write_csv(args.out)
        print(f"\nwrote {proj.height} projections to {args.out}")
    return 0


# --------------------------------------------------------------------------
# draft
# --------------------------------------------------------------------------

def _board(season: int, scoring: ScoringConfig, league: LeagueConfig, depth_chart: bool):
    proj = project_season(season, scoring=scoring, include_depth_chart=depth_chart)
    proj = proj.filter(pl.col("position").is_in(list(POSITIONS)))
    vor = value_over_replacement(
        proj["projected_points"].to_numpy(), proj["position"].to_numpy(), league
    )
    return proj.with_columns(pl.Series("vor", vor).round(1)).sort("vor", descending=True)


def _explain(row: dict, league: LeagueConfig, levels: dict[str, float]) -> str:
    """Why this player, in words rather than a bare number."""
    bits = [
        f"{row['projected_points']:.0f} projected",
        f"{row['vor']:+.0f} over replacement "
        f"({levels[row['position']]:.0f} at {row['position']})",
    ]
    if row.get("changed_team"):
        bits.append("changed teams")
    if row.get("new_head_coach"):
        bits.append("new head coach")
    pick = row.get("draft_best_pick_at_pos")
    if pick is not None and pick < 100:
        bits.append(f"team drafted {row['position']} at pick {int(pick)}")
    return "; ".join(bits)


def cmd_draft(args: argparse.Namespace) -> int:
    scoring = SCORING_PRESETS[args.scoring]()
    league = LeagueConfig(
        teams=args.teams, qb=args.qb, rb=args.rb, wr=args.wr, te=args.te,
        flex=args.flex, k=0, dst=0, bench=args.bench, scoring=scoring,
    )

    print(f"Loading {args.season} projections ...", file=sys.stderr)
    board = _board(args.season, scoring, league, args.depth_chart)
    levels = replacement_levels(
        board["projected_points"].to_numpy(), board["position"].to_numpy(), league
    )

    taken: set[str] = set()
    my_roster: list[dict] = []

    print(f"\n{league.teams}-team {scoring.name}, pick {args.pick}. "
          f"Roster: {league.roster_size}\n")
    print("Commands:  <name>  take a player off the board")
    print("           +<name> add to YOUR roster")
    print("           board   show top available")
    print("           roster  show your team")
    print("           undo    undo the last action")
    print("           quit\n")

    history: list[tuple[str, str]] = []

    def available() -> pl.DataFrame:
        return board.filter(~pl.col("player_display_name").is_in(list(taken)))

    def show_board(n: int = 12) -> None:
        counts = {p: sum(1 for r in my_roster if r["position"] == p) for p in POSITIONS}
        avail = available()
        eligible = avail.filter(
            ~pl.col("position").is_in(
                [p for p in POSITIONS if counts.get(p, 0) >= POSITION_CAPS[p]]
            )
        )
        print()
        for i, row in enumerate(eligible.head(n).iter_rows(named=True), 1):
            flag = "  <-- recommended" if i == 1 else ""
            print(f"{i:2d}. {row['player_display_name']:<24s} "
                  f"{row['position']:<3s} {row['team'] or '':<4s} "
                  f"{_explain(row, league, levels)}{flag}")
        print()

    show_board()

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw:
            continue

        cmd = raw.lower()
        if cmd in {"quit", "q", "exit"}:
            return 0
        if cmd == "board":
            show_board()
            continue
        if cmd == "roster":
            if not my_roster:
                print("  (empty)")
            for r in my_roster:
                print(f"  {r['position']:<3s} {r['player_display_name']:<24s} "
                      f"{r['projected_points']:.0f}")
            if my_roster:
                pts = np.array([r["projected_points"] for r in my_roster])
                pos = np.array([r["position"] for r in my_roster])
                print(f"  projected starting lineup: "
                      f"{optimal_lineup_points(list(range(len(pts))), pts, pos, league):.0f}")
            continue
        if cmd == "undo":
            if not history:
                print("  nothing to undo")
                continue
            action, name = history.pop()
            taken.discard(name)
            if action == "mine":
                my_roster[:] = [r for r in my_roster if r["player_display_name"] != name]
            print(f"  undid {name}")
            continue

        mine = raw.startswith("+")
        query = raw.lstrip("+").strip().lower()
        matches = available().filter(
            pl.col("player_display_name").str.to_lowercase().str.contains(query, literal=True)
        )
        if matches.is_empty():
            print(f"  no available player matching {query!r}")
            continue
        if matches.height > 1 and not any(
            r["player_display_name"].lower() == query
            for r in matches.iter_rows(named=True)
        ):
            names = [r["player_display_name"] for r in matches.head(6).iter_rows(named=True)]
            print("  ambiguous: " + ", ".join(names))
            continue

        row = matches.row(0, named=True)
        taken.add(row["player_display_name"])
        history.append(("mine" if mine else "gone", row["player_display_name"]))
        if mine:
            my_roster.append(row)
            print(f"  added {row['player_display_name']} to your roster")
        else:
            print(f"  {row['player_display_name']} off the board")
        show_board()


# --------------------------------------------------------------------------
# sleeper
# --------------------------------------------------------------------------

def cmd_sleeper(args: argparse.Namespace) -> int:
    from .sleeper import describe, load_league

    try:
        league = load_league(args.league_id)
    except Exception as exc:  # network, bad id, unexpected payload
        print(f"could not load Sleeper league {args.league_id!r}: {exc}", file=sys.stderr)
        return 1
    print(describe(league))
    return 0


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> int:
    from .espn import agreement, biggest_disagreements, compare, load_external, parse_cheat_sheet

    scoring = SCORING_PRESETS[args.scoring]()
    source = str(args.source)
    external = (
        parse_cheat_sheet(source) if source.lower().endswith(".pdf")
        else load_external(source)
    )
    ours = project_season(args.season, scoring=scoring,
                          include_depth_chart=args.depth_chart)
    c = compare(ours, external)

    print(f"matched {c.height} of {external.height} external entries "
          f"({c.height / external.height:.0%})\n")
    print("=== Rank agreement, within position ===")
    pl.Config.set_tbl_rows(10)
    print(agreement(c))
    print("\n=== Biggest disagreements ===")
    pl.Config.set_tbl_rows(2 * args.top)
    pl.Config.set_fmt_str_lengths(22)
    cols = ["player_display_name", "position", "team",
            "our_rank", "their_rank", "rank_delta", "projected_points"]
    print(biggest_disagreements(c, n=args.top).select(
        [x for x in cols if x in c.columns]
    ))
    if args.out:
        c.write_csv(args.out)
        print(f"\nwrote comparison to {args.out}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .report import write_report

    scoring = SCORING_PRESETS[args.scoring]()
    proj = project_season(
        args.season, scoring=scoring, include_depth_chart=args.depth_chart
    )
    path = write_report(proj, args.season, scoring, args.out)
    print(f"wrote {path}")
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="gridiron", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--season", type=int, default=2026)
        p.add_argument("--scoring", choices=list(SCORING_PRESETS), default="half_ppr")
        p.add_argument("--depth-chart", action="store_true",
                       help="use week-1 depth charts (more accurate, "
                            "but may not be published before your draft)")
        return p

    p = common(sub.add_parser("project", help="season projections"))
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--position", nargs="+")
    p.add_argument("--out", help="write full projections to CSV")
    p.set_defaults(func=cmd_project)

    p = common(sub.add_parser("draft", help="interactive draft assistant"))
    p.add_argument("--teams", type=int, default=12)
    p.add_argument("--pick", type=int, default=1)
    p.add_argument("--qb", type=int, default=1)
    p.add_argument("--rb", type=int, default=2)
    p.add_argument("--wr", type=int, default=2)
    p.add_argument("--te", type=int, default=1)
    p.add_argument("--flex", type=int, default=1)
    p.add_argument("--bench", type=int, default=6)
    p.set_defaults(func=cmd_draft)

    p = sub.add_parser("sleeper", help="read scoring + roster from a Sleeper league")
    p.add_argument("--league-id", required=True)
    p.set_defaults(func=cmd_sleeper)

    p = common(sub.add_parser("compare", help="compare against an external source"))
    p.add_argument("--source", required=True,
                   help="ESPN cheat-sheet PDF, or a CSV/text projections export")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--out", help="write the full comparison to CSV")
    p.set_defaults(func=cmd_compare)

    p = common(sub.add_parser("export", help="write a static HTML report"))
    p.add_argument("--out", default="docs/index.html")
    p.set_defaults(func=cmd_export)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
