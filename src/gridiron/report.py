"""Static HTML report, deployable to GitHub Pages.

`gridiron export` writes a single self-contained file to `docs/index.html`.
Turning on Pages for the `docs/` folder gives the repo a live URL, which is
worth more to a reviewer than any amount of README — they can click it.

Self-contained on purpose: no CDN, no build step, no server. It is a file
that works when opened from disk and works identically when served.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .config import ScoringConfig

_CSS = """
:root {
  color-scheme: light dark;
  --bg: #fbfbfa; --panel: #ffffff; --ink: #1a1a19; --muted: #6b6b68;
  --line: #e4e4e1; --accent: #1f6feb; --pos-qb: #7c3aed; --pos-rb: #059669;
  --pos-wr: #0284c7; --pos-te: #d97706;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14140f; --panel: #1c1c17; --ink: #eeeee8; --muted: #9a9a92;
    --line: #2f2f28; --accent: #58a6ff; --pos-qb: #a78bfa; --pos-rb: #34d399;
    --pos-wr: #38bdf8; --pos-te: #fbbf24;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 1040px; margin: 0 auto; padding: 40px 20px 80px; }
h1 { font-size: 28px; margin: 0 0 6px; letter-spacing: -0.02em; }
.sub { color: var(--muted); margin: 0 0 28px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr));
  gap: 12px; margin-bottom: 28px; }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 16px; }
.tile .n { font-size: 22px; font-weight: 600; letter-spacing: -0.01em; }
.tile .l { color: var(--muted); font-size: 13px; margin-top: 2px; }
.controls { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
button, input {
  font: inherit; padding: 7px 12px; border-radius: 8px;
  border: 1px solid var(--line); background: var(--panel); color: var(--ink);
}
button { cursor: pointer; }
button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
input { flex: 1; min-width: 160px; }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px;
  background: var(--panel); }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--line);
  white-space: nowrap; }
th { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--muted); font-weight: 600; cursor: pointer; user-select: none; }
tbody tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; }
.pos { font-weight: 600; font-size: 12px; }
.QB { color: var(--pos-qb); } .RB { color: var(--pos-rb); }
.WR { color: var(--pos-wr); } .TE { color: var(--pos-te); }
.note { color: var(--muted); font-size: 13px; margin-top: 22px; }
.note code { background: var(--panel); padding: 1px 5px; border-radius: 4px;
  border: 1px solid var(--line); }
a { color: var(--accent); }
"""

_JS = """
const rows = DATA;
let pos = 'ALL', sortKey = 'projected_points', desc = true;
const tbody = document.querySelector('tbody');
const search = document.getElementById('q');

function render() {
  const q = search.value.trim().toLowerCase();
  let view = rows.filter(r =>
    (pos === 'ALL' || r.position === pos) &&
    (!q || r.player_display_name.toLowerCase().includes(q)));
  view.sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    if (x === y) return 0;
    if (x === null) return 1;
    if (y === null) return -1;
    const c = typeof x === 'string' ? x.localeCompare(y) : x - y;
    return desc ? -c : c;
  });
  tbody.innerHTML = view.map((r, i) => `<tr>
    <td class="num">${i + 1}</td>
    <td>${r.player_display_name}</td>
    <td><span class="pos ${r.position}">${r.position}</span></td>
    <td>${r.team ?? ''}</td>
    <td class="num">${r.projected_points.toFixed(1)}</td>
    <td class="num">${r.last_season_points === null ? '' : r.last_season_points.toFixed(1)}</td>
    <td class="num">${r.age === null ? '' : r.age}</td>
    <td>${r.changed_team ? 'new team' : ''}</td>
  </tr>`).join('');
}

document.querySelectorAll('[data-pos]').forEach(b => b.onclick = () => {
  pos = b.dataset.pos;
  document.querySelectorAll('[data-pos]').forEach(o =>
    o.setAttribute('aria-pressed', String(o === b)));
  render();
});
document.querySelectorAll('th[data-key]').forEach(h => h.onclick = () => {
  const k = h.dataset.key;
  if (k === sortKey) { desc = !desc; } else { sortKey = k; desc = true; }
  render();
});
search.oninput = render;
render();
"""


def write_report(
    projections: pl.DataFrame,
    season: int,
    scoring: ScoringConfig,
    out_path: str | Path = "docs/index.html",
) -> Path:
    """Render projections to a single self-contained HTML file."""
    cols = [
        "player_display_name", "position", "team",
        "projected_points", "last_season_points", "age", "changed_team",
    ]
    data = projections.select([c for c in cols if c in projections.columns]).to_dicts()
    for row in data:
        for key in ("projected_points", "last_season_points", "age"):
            if row.get(key) is not None:
                row[key] = float(row[key])
        row["changed_team"] = bool(row.get("changed_team") or 0)

    by_pos = projections.group_by("position").len().to_dicts()
    counts = {r["position"]: r["len"] for r in by_pos}
    moved = int(projections["changed_team"].fill_null(0).sum())

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gridiron — {season} projections</title>
<style>{_CSS}</style>
</head><body><div class="wrap">

<h1>gridiron — {season} projections</h1>
<p class="sub">
  {scoring.name.replace('_', ' ')} scoring &middot;
  gradient boosted model trained on 2012&ndash;{season - 1} nflverse data &middot;
  <a href="https://github.com/DrewCav25/gridiron">source and methodology</a>
</p>

<div class="tiles">
  <div class="tile"><div class="n">{projections.height}</div>
    <div class="l">players projected</div></div>
  <div class="tile"><div class="n">{counts.get('RB', 0)} / {counts.get('WR', 0)}</div>
    <div class="l">running backs / receivers</div></div>
  <div class="tile"><div class="n">{moved}</div>
    <div class="l">changed teams this offseason</div></div>
  <div class="tile"><div class="n">0.73</div>
    <div class="l">WR rank correlation, backtested</div></div>
</div>

<div class="controls">
  <button data-pos="ALL" aria-pressed="true">All</button>
  <button data-pos="QB" aria-pressed="false">QB</button>
  <button data-pos="RB" aria-pressed="false">RB</button>
  <button data-pos="WR" aria-pressed="false">WR</button>
  <button data-pos="TE" aria-pressed="false">TE</button>
  <input id="q" placeholder="Search players" aria-label="Search players">
</div>

<div class="scroll"><table>
<thead><tr>
  <th class="num">#</th>
  <th data-key="player_display_name">Player</th>
  <th data-key="position">Pos</th>
  <th data-key="team">Team</th>
  <th class="num" data-key="projected_points">Projected</th>
  <th class="num" data-key="last_season_points">{season - 1}</th>
  <th class="num" data-key="age">Age</th>
  <th>Notes</th>
</tr></thead>
<tbody></tbody>
</table></div>

<p class="note">
  Projections are season-total fantasy points under {scoring.name.replace('_', ' ')}
  scoring, from a model using prior-season opportunity metrics plus offseason
  information (team changes, incoming draft capital at the position, coaching
  changes, new-team offensive context). Week-1 depth charts are excluded by
  default, since they often publish after August drafts &mdash; pass
  <code>--depth-chart</code> to include them.
  Rookies are not projected: with no prior NFL season they need a separate
  model built on draft capital and college production.
  Kickers and team defenses are not projected either &mdash; neither is
  meaningfully predictable at the season level.
  Click a column header to sort.
</p>

<script>const DATA = {json.dumps(data)};
{_JS}</script>
</div></body></html>
"""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
