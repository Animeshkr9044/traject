"""Compare two runs of the same scenarios: per-scenario deltas, per-criterion flips, regressions."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import ScoredTrajectory


class CriterionFlip(BaseModel):
    scenario_id: str
    criterion_id: str
    before: bool
    after: bool


class ScenarioDelta(BaseModel):
    scenario_id: str
    before: float | None
    after: float | None
    delta: float | None
    task_before: bool | None
    task_after: bool | None


class DiffResult(BaseModel):
    target_a: str
    target_b: str
    mean_a: float | None
    mean_b: float | None
    wins: int = 0  # b strictly better
    losses: int = 0  # b strictly worse
    ties: int = 0
    scenarios: list[ScenarioDelta] = Field(default_factory=list)
    regressions: list[CriterionFlip] = Field(default_factory=list)  # pass -> fail
    improvements: list[CriterionFlip] = Field(default_factory=list)  # fail -> pass
    only_in_a: list[str] = Field(default_factory=list)
    only_in_b: list[str] = Field(default_factory=list)

    @property
    def win_rate(self) -> float | None:
        n = self.wins + self.losses + self.ties
        return self.wins / n if n else None


def _index(run: list[ScoredTrajectory]) -> dict[str, ScoredTrajectory]:
    """If a run has repeats (id__rN), average isn't meaningful per-criterion; keep the first.
    Repeat-aware aggregation lives in report.py."""
    out: dict[str, ScoredTrajectory] = {}
    for r in run:
        out.setdefault(r.trajectory.scenario_id, r)
    return out


def diff_runs(a: list[ScoredTrajectory], b: list[ScoredTrajectory], eps: float = 1e-9) -> DiffResult:
    ia, ib = _index(a), _index(b)
    res = DiffResult(
        target_a=a[0].trajectory.target if a else "?",
        target_b=b[0].trajectory.target if b else "?",
        mean_a=_mean([r.score() for r in ia.values()]),
        mean_b=_mean([r.score() for r in ib.values()]),
        only_in_a=sorted(set(ia) - set(ib)),
        only_in_b=sorted(set(ib) - set(ia)),
    )
    for sid in sorted(set(ia) & set(ib)):
        ra, rb = ia[sid], ib[sid]
        sa, sb = ra.score(), rb.score()
        delta = (sb - sa) if (sa is not None and sb is not None) else None
        res.scenarios.append(
            ScenarioDelta(
                scenario_id=sid, before=sa, after=sb, delta=delta,
                task_before=ra.verdict.task_completed if ra.verdict else None,
                task_after=rb.verdict.task_completed if rb.verdict else None,
            )
        )
        if delta is None:
            continue
        if delta > eps:
            res.wins += 1
        elif delta < -eps:
            res.losses += 1
        else:
            res.ties += 1
        pa, pb = ra.passed_ids(), rb.passed_ids()
        all_ids = {c.id for c in ra.rubric}
        for cid in sorted(all_ids):
            before, after = cid in pa, cid in pb
            if before and not after:
                res.regressions.append(CriterionFlip(scenario_id=sid, criterion_id=cid, before=True, after=False))
            elif after and not before:
                res.improvements.append(CriterionFlip(scenario_id=sid, criterion_id=cid, before=False, after=True))
    return res


def _mean(xs: list[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return sum(v) / len(v) if v else None
