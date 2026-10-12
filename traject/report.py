"""Single-file HTML report for one run, or for a diff between two runs."""

from __future__ import annotations

import html
from pathlib import Path

from .diff import DiffResult
from .models import ScoredTrajectory

_CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e5e5e5;--pass:#1a7f37;--fail:#c42b1c;--card:#fafafa;--accent:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e6e6e6;--muted:#9a9a9a;--line:#2a2d35;--pass:#3fb950;--fail:#f85149;--card:#171a21;--accent:#60a5fa}}
*{box-sizing:border-box}body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;max-width:1100px;margin-inline:auto}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:8px 0}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;border:1px solid var(--line);margin:1px 2px 1px 0}
.pass{color:var(--pass);border-color:var(--pass)}.fail{color:var(--fail);border-color:var(--fail)}
details{border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:8px 0;background:var(--card)}summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;font-size:12.5px}
.turn{margin:6px 0;padding:6px 10px;border-left:3px solid var(--line)}.turn.user{border-color:var(--accent)}.turn.agent{border-color:var(--pass)}
.tool{font-family:ui-monospace,monospace;font-size:12px;color:var(--muted);margin-left:12px}
.num{font-variant-numeric:tabular-nums}.delta-up{color:var(--pass)}.delta-down{color:var(--fail)}
"""


def _e(s: object) -> str:
    return html.escape(str(s))


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def _trajectory_html(r: ScoredTrajectory) -> str:
    t = r.trajectory
    parts = [f"<details><summary>{_e(t.scenario_id)} &nbsp; <span class='muted'>{_pct(r.score())} · {len(t.turns)} turns · ended by {_e(t.ended_by)}</span></summary>"]
    if r.verdict:
        parts.append("<p>" + "".join(
            f"<span class='pill {'pass' if c.passed else 'fail'}' title='{_e(c.evidence)}'>{_e(c.id)}</span>"
            for c in r.verdict.criteria
        ) + f" &nbsp; <span class='muted'>task {'completed' if r.verdict.task_completed else 'not completed'}</span></p>")
        parts.append(f"<p class='muted'>{_e(r.verdict.summary)}</p>")
        parts.append("<table><tr><th>criterion</th><th>result</th><th>conf</th><th>evidence</th></tr>")
        for c in r.verdict.criteria:
            parts.append(f"<tr><td>{_e(c.id)}</td><td class='{'pass' if c.passed else 'fail'}'>{'pass' if c.passed else 'fail'}</td><td class='num'>{c.confidence:.2f}</td><td>{_e(c.evidence)}</td></tr>")
        parts.append("</table>")
    if t.error:
        parts.append(f"<pre class='fail'>{_e(t.error)}</pre>")
    for i, turn in enumerate(t.turns):
        parts.append(f"<div class='turn {turn.role}'><b>[{i}] {turn.role}</b> {_e(turn.text)}")
        for ev in turn.tool_events:
            parts.append(f"<div class='tool'>⚙ {_e(ev.name)}({_e(ev.input)}) → {_e(ev.output[:400])}{' ERROR' if ev.error else ''}</div>")
        parts.append("</div>")
    parts.append("</details>")
    return "\n".join(parts)


def run_report_html(results: list[ScoredTrajectory], title: str = "traject run") -> str:
    scored = [r for r in results if r.verdict]
    mean = sum(r.score() for r in scored) / len(scored) if scored else None
    tcr = sum(1 for r in scored if r.verdict.task_completed) / len(scored) if scored else None
    target = results[0].trajectory.target if results else "?"
    # per-criterion pass rate across scenarios
    crit: dict[str, list[bool]] = {}
    for r in scored:
        for c in r.verdict.criteria:
            crit.setdefault(c.id, []).append(c.passed)
    crit_rows = "".join(
        f"<tr><td>{_e(k)}</td><td class='num'>{_pct(sum(v) / len(v))}</td><td class='num'>{len(v)}</td></tr>"
        for k, v in sorted(crit.items())
    )
    body = f"""
<h1>{_e(title)}</h1>
<p class='muted'>target <b>{_e(target)}</b> · {len(results)} trajectories · mean score <b>{_pct(mean)}</b> · task completion <b>{_pct(tcr)}</b></p>
<h2>Per-criterion pass rate</h2>
<table><tr><th>criterion</th><th>pass rate</th><th>n</th></tr>{crit_rows}</table>
<h2>Trajectories</h2>
{''.join(_trajectory_html(r) for r in results)}
"""
    return _page(title, body)


def diff_report_html(d: DiffResult, a: list[ScoredTrajectory], b: list[ScoredTrajectory]) -> str:
    rows = []
    for s in d.scenarios:
        cls = "delta-up" if (s.delta or 0) > 0 else "delta-down" if (s.delta or 0) < 0 else ""
        rows.append(f"<tr><td>{_e(s.scenario_id)}</td><td class='num'>{_pct(s.before)}</td><td class='num'>{_pct(s.after)}</td><td class='num {cls}'>{'—' if s.delta is None else f'{s.delta * 100:+.0f}%'}</td></tr>")
    reg = "".join(f"<li>{_e(f.scenario_id)} · <span class='pill fail'>{_e(f.criterion_id)}</span></li>" for f in d.regressions) or "<li class='muted'>none</li>"
    imp = "".join(f"<li>{_e(f.scenario_id)} · <span class='pill pass'>{_e(f.criterion_id)}</span></li>" for f in d.improvements) or "<li class='muted'>none</li>"
    body = f"""
<h1>traject diff</h1>
<p class='muted'><b>{_e(d.target_a)}</b> → <b>{_e(d.target_b)}</b> · mean {_pct(d.mean_a)} → {_pct(d.mean_b)} · wins {d.wins} / losses {d.losses} / ties {d.ties}</p>
<h2>Regressions (pass → fail)</h2><ul>{reg}</ul>
<h2>Improvements (fail → pass)</h2><ul>{imp}</ul>
<h2>Per-scenario</h2>
<table><tr><th>scenario</th><th>before</th><th>after</th><th>delta</th></tr>{''.join(rows)}</table>
<h2>Trajectories: {_e(d.target_b)}</h2>{''.join(_trajectory_html(r) for r in b)}
<h2>Trajectories: {_e(d.target_a)}</h2>{''.join(_trajectory_html(r) for r in a)}
"""
    return _page("traject diff", body)


def _page(title: str, body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{_e(title)}</title><style>{_CSS}</style></head><body>{body}</body></html>"


def write(path: str | Path, content: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p
