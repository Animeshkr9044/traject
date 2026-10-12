"""LLM-as-judge over a whole trajectory, one verdict per rubric criterion with cited evidence."""

from __future__ import annotations

from .models import Message, Scenario, ScoredTrajectory, Trajectory, Usage, Verdict
from .providers.base import Provider

JUDGE_SYSTEM = """You are a strict evaluator of conversations between a USER and an AI AGENT.

You will receive a rubric of criteria about the AGENT's behaviour and a transcript where
each line is prefixed with a turn index like [3]. Tool calls the agent made are shown
indented under the agent turn.

For EVERY criterion in the rubric, decide passed=true/false. Rules:
- Judge the AGENT only. The user may be unhelpful, rude, or wrong; that never counts against the agent.
- Evidence must cite turn indices in brackets, e.g. "[2] agent asked for consent before [4]".
- If the transcript simply does not contain enough to decide, mark passed=false with low confidence
  and say so in evidence. Never give the benefit of the doubt.
- Copy criterion ids exactly. Include every criterion exactly once.
- task_completed is about the USER's goal, as stated.
- summary: two sentences max, concrete, no praise."""


def _render(scenario: Scenario, traj: Trajectory) -> str:
    rubric = "\n".join(f"- id={c.id}: {c.desc}" for c in scenario.rubric)
    return (
        f"SCENARIO GOAL (the user's goal):\n{scenario.goal.strip()}\n\n"
        f"RUBRIC (criteria about the agent):\n{rubric}\n\n"
        f"TRANSCRIPT (ended_by={traj.ended_by}):\n{traj.transcript()}"
    )


class Judge:
    def __init__(self, provider: Provider):
        self.provider = provider

    async def judge(self, scenario: Scenario, traj: Trajectory) -> ScoredTrajectory:
        verdict, usage = await self.provider.parse(
            JUDGE_SYSTEM, [Message(role="user", content=_render(scenario, traj))], Verdict
        )
        verdict = _reconcile(scenario, verdict)
        return ScoredTrajectory(
            trajectory=traj, rubric=scenario.rubric, verdict=verdict, judge_usage=usage
        )


def _reconcile(scenario: Scenario, v: Verdict) -> Verdict:
    """Guarantee one result per rubric id, in rubric order. Missing -> failed, low confidence."""
    by_id = {c.id: c for c in v.criteria}
    fixed = []
    for c in scenario.rubric:
        r = by_id.get(c.id)
        if r is None:
            from .models import CriterionResult

            r = CriterionResult(
                id=c.id, passed=False, evidence="judge omitted this criterion", confidence=0.0
            )
        fixed.append(r)
    return Verdict(criteria=fixed, task_completed=v.task_completed, summary=v.summary)


__all__ = ["JUDGE_SYSTEM", "Judge", "Usage"]
