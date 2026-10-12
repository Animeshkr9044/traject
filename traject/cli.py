from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .calibrate import calibrate as _calibrate
from .calibrate import load_labels
from .diff import diff_runs
from .judge import Judge
from .models import ScoredTrajectory
from .providers import get_provider
from .report import diff_report_html, run_report_html, write
from .runner import load_run, load_scenarios, run_all, write_summary
from .simulator import Simulator
from .targets import load_target

app = typer.Typer(help="Trajectory evals for multi-turn LLM agents.", no_args_is_help=True)
con = Console()


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


@app.command()
def run(
    scenarios: Path = typer.Argument(..., help="scenario YAML file or directory"),
    target: Path = typer.Option(..., "--target", "-t", help="target YAML"),
    out: Path = typer.Option(..., "--out", "-o", help="output directory for this run"),
    sim_provider: str = typer.Option("anthropic", help="anthropic | openai | mock"),
    sim_model: str | None = typer.Option(None),
    target_provider: str | None = typer.Option(None, help="override the target YAML's provider"),
    target_model: str | None = typer.Option(None, help="override the target YAML's model"),
    judge_provider: str | None = typer.Option(None, help="defaults to sim provider; 'none' to skip judging"),
    judge_model: str | None = typer.Option(None),
    concurrency: int = typer.Option(4),
    repeats: int = typer.Option(1, help="run each scenario N times (variance check)"),
    html: Path | None = typer.Option(None, help="also write an HTML report here"),
):
    """Run scenarios against a target, judge them, write JSON (+ optional HTML)."""
    scs = load_scenarios(scenarios)
    tgt = load_target(target, target_provider, target_model)
    sim = Simulator(get_provider(sim_provider, sim_model))
    jp = judge_provider or sim_provider
    judge = None if jp == "none" else Judge(get_provider(jp, judge_model))
    con.print(f"[bold]{len(scs)}[/] scenarios × {repeats} → target [bold]{tgt.name}[/] (sim={sim_provider}, judge={jp})")

    def progress(r: ScoredTrajectory):
        t = r.trajectory
        mark = "[red]ERR[/]" if t.ended_by == "error" else _pct(r.score())
        con.print(f"  {t.scenario_id:<40} {mark:>6}  {len(t.turns)} turns  {t.ended_by}")

    results = asyncio.run(run_all(scs, tgt, sim, judge, out, concurrency, repeats, progress))
    _print_summary(results)
    if html:
        write(html, run_report_html(results, title=f"traject · {tgt.name}"))
        con.print(f"html → {html}")
    con.print(f"json → {out}/")


@app.command()
def judge(
    run_dir: Path = typer.Argument(..., help="existing run directory"),
    scenarios: Path = typer.Option(..., "--scenarios", "-s"),
    provider: str = typer.Option("anthropic"),
    model: str | None = typer.Option(None),
    out: Path | None = typer.Option(None, help="write to a new dir instead of in place"),
):
    """(Re-)judge an existing run, e.g. with a different judge model."""
    scs = {s.id: s for s in load_scenarios(scenarios)}
    results = load_run(run_dir)
    j = Judge(get_provider(provider, model))

    async def go():
        async def one(r: ScoredTrajectory):
            sc = scs[r.trajectory.scenario_id]
            return await j.judge(sc, r.trajectory)

        return await asyncio.gather(*(one(r) for r in results))

    scored = asyncio.run(go())
    dest = out or run_dir
    dest.mkdir(parents=True, exist_ok=True)
    for r in scored:
        (dest / f"{r.trajectory.scenario_id}.json").write_text(r.model_dump_json(indent=2))
    write_summary(dest, scored[0].trajectory.target, list(scored))
    _print_summary(list(scored))


@app.command()
def diff(
    run_a: Path = typer.Argument(...),
    run_b: Path = typer.Argument(...),
    html: Path | None = typer.Option(None),
    fail_on_regression: bool = typer.Option(False, help="exit 1 if any criterion went pass→fail (CI gate)"),
):
    """Compare two runs over the same scenarios."""
    a, b = load_run(run_a), load_run(run_b)
    d = diff_runs(a, b)
    con.print(f"[bold]{d.target_a}[/] → [bold]{d.target_b}[/]   mean {_pct(d.mean_a)} → {_pct(d.mean_b)}   "
              f"wins {d.wins} / losses {d.losses} / ties {d.ties}")
    t = Table("scenario", "before", "after", "delta")
    for s in d.scenarios:
        dl = "—" if s.delta is None else f"{s.delta * 100:+.0f}%"
        style = "green" if (s.delta or 0) > 0 else "red" if (s.delta or 0) < 0 else ""
        t.add_row(s.scenario_id, _pct(s.before), _pct(s.after), f"[{style}]{dl}[/]" if style else dl)
    con.print(t)
    if d.regressions:
        con.print("[red bold]regressions (pass → fail):[/]")
        for f in d.regressions:
            con.print(f"  {f.scenario_id} · {f.criterion_id}")
    if d.improvements:
        con.print("[green bold]improvements (fail → pass):[/]")
        for f in d.improvements:
            con.print(f"  {f.scenario_id} · {f.criterion_id}")
    if d.only_in_a or d.only_in_b:
        con.print(f"[yellow]unmatched scenarios[/]: only in A {d.only_in_a}, only in B {d.only_in_b}")
    if html:
        write(html, diff_report_html(d, a, b))
        con.print(f"html → {html}")
    if fail_on_regression and d.regressions:
        raise typer.Exit(code=1)


@app.command()
def report(run_dir: Path = typer.Argument(...), html: Path = typer.Option(..., "--html")):
    """Render an HTML report for a run."""
    results = load_run(run_dir)
    write(html, run_report_html(results, title=f"traject · {results[0].trajectory.target}"))
    con.print(f"html → {html}")


@app.command()
def calibrate(
    run_dir: Path = typer.Argument(...),
    labels: Path = typer.Option(..., "--labels", "-l", help="JSONL of human labels"),
    show_disagreements: bool = typer.Option(True),
):
    """Measure judge agreement with human labels (accuracy + Cohen's kappa)."""
    rep = _calibrate(load_run(run_dir), load_labels(labels))
    o = rep.overall
    con.print(f"[bold]overall[/]  n={o.n}  agreement={_pct(o.agreement)}  kappa={'—' if o.kappa is None else f'{o.kappa:.2f}'}  "
              f"too-lenient(fp)={o.fp}  too-strict(fn)={o.fn}")
    t = Table("criterion", "n", "agreement", "kappa", "fp", "fn")
    for cid, s in sorted(rep.per_criterion.items()):
        t.add_row(cid, str(s.n), _pct(s.agreement), "—" if s.kappa is None else f"{s.kappa:.2f}", str(s.fp), str(s.fn))
    con.print(t)
    if rep.unlabeled or rep.unmatched:
        con.print(f"[yellow]{rep.unlabeled} judged criteria had no human label; {rep.unmatched} labels matched nothing[/]")
    if show_disagreements and rep.disagreements():
        con.print("[bold]disagreements[/]")
        for d in rep.disagreements():
            con.print(f"  {d['scenario_id']} · {d['criterion_id']}  human={d['human']} judge={d['judge']} "
                      f"(conf {d['confidence']:.2f}) — {d['evidence'][:140]}")


@app.command("export-labels")
def export_labels(run_dir: Path = typer.Argument(...), out: Path = typer.Option(..., "--out", "-o")):
    """Dump a JSONL template of (scenario, criterion, judge verdict) for a human to correct."""
    lines = []
    for r in load_run(run_dir):
        if not r.verdict:
            continue
        for c in r.verdict.criteria:
            lines.append(json.dumps({
                "scenario_id": r.trajectory.scenario_id, "criterion_id": c.id,
                "passed": c.passed, "_judge_evidence": c.evidence,
            }))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    con.print(f"{len(lines)} rows → {out}  (edit `passed`, keep `_judge_evidence` for reference)")


def _print_summary(results: list[ScoredTrajectory]):
    scored = [r for r in results if r.verdict]
    if not scored:
        con.print("no judged trajectories")
        return
    mean = sum(r.score() for r in scored) / len(scored)
    tcr = sum(1 for r in scored if r.verdict.task_completed) / len(scored)
    tok = sum((r.trajectory.agent_usage.input_tokens + r.trajectory.agent_usage.output_tokens) for r in results)
    con.print(f"[bold]mean score {_pct(mean)}[/]  task completion {_pct(tcr)}  agent tokens {tok:,}")


if __name__ == "__main__":
    app()
