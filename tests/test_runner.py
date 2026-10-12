import asyncio
from pathlib import Path

from traject.judge import Judge
from traject.models import (
    Criterion,
    CriterionResult,
    Reply,
    Scenario,
    SimTurn,
    ToolCall,
    Verdict,
)
from traject.providers.mock import MockProvider
from traject.runner import load_scenarios, run_all, run_scenario
from traject.simulator import Simulator
from traject.targets.base import TargetConfig
from traject.targets.llm_agent import LLMAgentTarget

ROOT = Path(__file__).resolve().parents[1]

SCENARIO = Scenario(
    id="s1",
    persona="p",
    goal="g",
    max_turns=6,
    terminal_tools=["submit_report"],
    rubric=[Criterion(id="a", desc="A", weight=2), Criterion(id="b", desc="B")],
)


def _target(provider: MockProvider) -> LLMAgentTarget:
    cfg = TargetConfig(name="t", kind="llm", provider="mock", tools="examples.interviewer.tools:TOOLS",
                       system="sys")
    t = LLMAgentTarget(cfg)
    t._provider_factory = lambda: provider
    return t


def test_agent_tool_loop_executes_tools_and_records_events():
    agent = MockProvider(replies=[
        Reply(text="", tool_calls=[ToolCall(id="1", name="get_question", input={"track": "backend", "index": 0})]),
        Reply(text="Q1: ...?"),
    ])
    sim = Simulator(MockProvider(parsed=[SimTurn(message="hello", done=False, reason="")]))
    target = _target(agent)
    traj = asyncio.run(run_scenario(SCENARIO.model_copy(update={"max_turns": 1}), target, sim))
    assert traj.ended_by == "max_turns"
    agent_turn = traj.turns[1]
    assert agent_turn.role == "agent" and agent_turn.text == "Q1: ...?"
    assert agent_turn.tool_events[0].name == "get_question"
    assert '"q1"' in agent_turn.tool_events[0].output
    # tool result went back into the agent's history before the second provider call
    assert agent.chat_calls[1][-1].role == "tool"


def test_terminal_tool_ends_run():
    agent = MockProvider(replies=[
        Reply(text="bye", tool_calls=[ToolCall(id="1", name="submit_report",
              input={"track": "backend", "scores": {"q1": 3}, "recommendation": "yes"})]),
        Reply(text="(unused)"),
    ])
    sim = Simulator(MockProvider(parsed=[SimTurn(message="hi", done=False, reason="")]))
    traj = asyncio.run(run_scenario(SCENARIO, _target(agent), sim))
    assert traj.ended_by == "terminal_tool"
    assert len(traj.turns) == 2


def test_simulator_done_ends_run_and_keeps_closing_line():
    agent = MockProvider(replies=[Reply(text="ok")])
    sim = Simulator(MockProvider(parsed=[
        SimTurn(message="hi", done=False, reason=""),
        SimTurn(message="thanks, bye", done=True, reason="goal reached"),
    ]))
    traj = asyncio.run(run_scenario(SCENARIO, _target(agent), sim))
    assert traj.ended_by == "simulator_done"
    assert traj.turns[-1].role == "user" and traj.turns[-1].text == "thanks, bye"


def test_unknown_tool_is_recorded_as_error_not_crash():
    agent = MockProvider(replies=[
        Reply(text="", tool_calls=[ToolCall(id="1", name="nope", input={})]),
        Reply(text="fine"),
    ])
    sim = Simulator(MockProvider(parsed=[SimTurn(message="hi", done=False, reason="")]))
    traj = asyncio.run(run_scenario(SCENARIO.model_copy(update={"max_turns": 1}), _target(agent), sim))
    ev = traj.turns[1].tool_events[0]
    assert ev.error and "unknown tool" in ev.output


def test_run_all_judges_scores_and_persists(tmp_path):
    agent = MockProvider(replies=[Reply(text="ok")])
    sim = Simulator(MockProvider(parsed=[SimTurn(message="x", done=True, reason="")]))
    judge = Judge(MockProvider(parsed=[Verdict(
        criteria=[CriterionResult(id="a", passed=True, evidence="[0]", confidence=0.9)],  # 'b' omitted on purpose
        task_completed=True, summary="s",
    )]))
    results = asyncio.run(run_all([SCENARIO], _target(agent), sim, judge, tmp_path, concurrency=2))
    r = results[0]
    assert r.verdict is not None
    assert [c.id for c in r.verdict.criteria] == ["a", "b"]  # reconciled to rubric order
    assert r.verdict.criteria[1].passed is False and r.verdict.criteria[1].confidence == 0.0
    assert r.score() == 2 / 3  # weight 2 of 3 passed
    assert (tmp_path / "s1.json").exists() and (tmp_path / "summary.json").exists()


def test_bundled_scenarios_load_and_have_unique_ids():
    scs = load_scenarios(ROOT / "scenarios" / "interviewer")
    ids = [s.id for s in scs]
    assert len(ids) == len(set(ids)) and len(ids) >= 4
    assert all(s.rubric for s in scs)
