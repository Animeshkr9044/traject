"""Drive scenarios against a target, optionally judge, persist results."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .judge import Judge
from .models import Scenario, ScoredTrajectory, Trajectory, Turn, Usage
from .simulator import Simulator
from .targets.base import Target


def load_scenarios(path: str | Path) -> list[Scenario]:
    p = Path(path)
    files = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml")) if p.is_dir() else [p]
    out: list[Scenario] = []
    for f in files:
        docs = list(yaml.safe_load_all(f.read_text()))
        for d in docs:
            if d:
                out.append(Scenario.model_validate(d))
    if not out:
        raise FileNotFoundError(f"no scenarios found at {path}")
    return out


async def run_scenario(scenario: Scenario, target: Target, sim: Simulator) -> Trajectory:
    session = target.new_session()
    turns: list[Turn] = []
    sim_usage = Usage()
    ended: str = "max_turns"
    error: str | None = None
    try:
        for i in range(scenario.max_turns):
            # --- user turn
            if i == 0 and scenario.opening:
                user_text = scenario.opening
            else:
                st, u = await sim.next_turn(scenario, turns)
                sim_usage = sim_usage + u
                if st.done:
                    ended = "simulator_done"
                    if st.message.strip():
                        turns.append(Turn(role="user", text=st.message.strip()))
                    break
                user_text = st.message.strip() or "(silence)"
            turns.append(Turn(role="user", text=user_text))

            # --- agent turn
            t0 = time.perf_counter()
            reply = await session.respond(user_text)
            turns.append(
                Turn(
                    role="agent",
                    text=reply.text,
                    tool_events=reply.tool_events,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    usage=reply.usage,
                )
            )
            if any(ev.name in scenario.terminal_tools for ev in reply.tool_events):
                ended = "terminal_tool"
                break
    except Exception as e:  # keep the partial trajectory; the run must not lose other scenarios
        ended = "error"
        error = f"{type(e).__name__}: {e}"
    return Trajectory(
        scenario_id=scenario.id, target=target.name, turns=turns, ended_by=ended,
        error=error, sim_usage=sim_usage,
    )


async def run_all(
    scenarios: list[Scenario],
    target: Target,
    sim: Simulator,
    judge: Judge | None,
    out_dir: Path,
    concurrency: int = 4,
    repeats: int = 1,
    progress=None,
) -> list[ScoredTrajectory]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    results: list[ScoredTrajectory] = []

    async def one(sc: Scenario, rep: int) -> ScoredTrajectory:
        async with sem:
            traj = await run_scenario(sc, target, sim)
            if judge is not None and traj.ended_by != "error":
                scored = await judge.judge(sc, traj)
            else:
                scored = ScoredTrajectory(trajectory=traj, rubric=sc.rubric)
            name = sc.id if repeats == 1 else f"{sc.id}__r{rep}"
            (out_dir / f"{name}.json").write_text(scored.model_dump_json(indent=2))
            if progress:
                progress(scored)
            return scored

    results = await asyncio.gather(*(one(sc, r) for sc in scenarios for r in range(repeats)))
    write_summary(out_dir, target.name, results)
    return list(results)


def write_summary(out_dir: Path, target_name: str, results: list[ScoredTrajectory]) -> dict:
    scored = [r for r in results if r.verdict is not None]
    summary = {
        "target": target_name,
        "created_at": datetime.now(UTC).isoformat(),
        "n": len(results),
        "n_judged": len(scored),
        "n_errors": sum(1 for r in results if r.trajectory.ended_by == "error"),
        "mean_score": (sum(r.score() for r in scored) / len(scored)) if scored else None,
        "task_completion_rate": (
            sum(1 for r in scored if r.verdict.task_completed) / len(scored) if scored else None
        ),
        "scenarios": {
            r.trajectory.scenario_id: {
                "score": r.score(),
                "task_completed": r.verdict.task_completed if r.verdict else None,
                "ended_by": r.trajectory.ended_by,
                "turns": len(r.trajectory.turns),
                "agent_tokens": r.trajectory.agent_usage.model_dump(),
            }
            for r in results
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_run(run_dir: str | Path) -> list[ScoredTrajectory]:
    p = Path(run_dir)
    files = sorted(f for f in p.glob("*.json") if f.name != "summary.json")
    if not files:
        raise FileNotFoundError(f"no trajectories in {run_dir}")
    return [ScoredTrajectory.model_validate_json(f.read_text()) for f in files]
