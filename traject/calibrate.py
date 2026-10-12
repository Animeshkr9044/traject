"""Judge calibration: compare judge verdicts against human labels.

Human labels file: JSONL, one line per (scenario, criterion):
  {"scenario_id": "...", "criterion_id": "...", "passed": true}

Reports per-criterion and overall agreement + Cohen's kappa. Without this, an LLM judge is a vibe.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from .models import ScoredTrajectory


class PairStats(BaseModel):
    n: int = 0
    agree: int = 0
    tp: int = 0  # human pass, judge pass
    tn: int = 0
    fp: int = 0  # human fail, judge pass  (judge too lenient)
    fn: int = 0  # human pass, judge fail  (judge too strict)

    @property
    def agreement(self) -> float | None:
        return self.agree / self.n if self.n else None

    @property
    def kappa(self) -> float | None:
        """Cohen's kappa for two binary raters."""
        n = self.n
        if n == 0:
            return None
        po = (self.tp + self.tn) / n
        p_judge_yes = (self.tp + self.fp) / n
        p_human_yes = (self.tp + self.fn) / n
        pe = p_judge_yes * p_human_yes + (1 - p_judge_yes) * (1 - p_human_yes)
        if pe == 1.0:
            return 1.0 if po == 1.0 else 0.0
        return (po - pe) / (1 - pe)


class CalibrationReport(BaseModel):
    overall: PairStats
    per_criterion: dict[str, PairStats] = Field(default_factory=dict)
    unlabeled: int = 0  # judge results with no human label
    unmatched: int = 0  # human labels with no judge result

    def disagreements(self) -> list[dict]:
        return self._dis

    _dis: list[dict] = []


def load_labels(path: str | Path) -> dict[tuple[str, str], bool]:
    out: dict[tuple[str, str], bool] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        d = json.loads(line)
        out[(d["scenario_id"], d["criterion_id"])] = bool(d["passed"])
    return out


def calibrate(run: list[ScoredTrajectory], labels: dict[tuple[str, str], bool]) -> CalibrationReport:
    rep = CalibrationReport(overall=PairStats())
    seen: set[tuple[str, str]] = set()
    dis: list[dict] = []
    for r in run:
        if r.verdict is None:
            continue
        sid = r.trajectory.scenario_id
        for c in r.verdict.criteria:
            key = (sid, c.id)
            if key not in labels:
                rep.unlabeled += 1
                continue
            seen.add(key)
            human = labels[key]
            for stats in (rep.overall, rep.per_criterion.setdefault(c.id, PairStats())):
                stats.n += 1
                if human == c.passed:
                    stats.agree += 1
                    if human:
                        stats.tp += 1
                    else:
                        stats.tn += 1
                elif c.passed:
                    stats.fp += 1
                else:
                    stats.fn += 1
            if human != c.passed:
                dis.append({
                    "scenario_id": sid, "criterion_id": c.id, "human": human, "judge": c.passed,
                    "confidence": c.confidence, "evidence": c.evidence,
                })
    rep.unmatched = len(set(labels) - seen)
    rep._dis = dis
    return rep
