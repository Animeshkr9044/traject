"""Core data models. Provider-neutral so Anthropic / OpenAI / mock all share them."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------- provider-neutral chat primitives ----------


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    id: str
    name: str
    input: dict[str, Any]


class Message(BaseModel):
    """One message in a conversation.

    role=user:      content is the user's text
    role=assistant: content is text (may be ""), tool_calls may be set
    role=tool:      a tool result; tool_call_id + name + content (string output)
    """

    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class Reply(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    stop_reason: str | None = None


# ---------- scenario definition (YAML) ----------


class Criterion(BaseModel):
    id: str
    desc: str
    weight: float = 1.0


class Scenario(BaseModel):
    id: str
    persona: str
    goal: str
    opening: str | None = None  # if None, simulator writes the first message
    max_turns: int = 8
    rubric: list[Criterion]
    terminal_tools: list[str] = Field(default_factory=list)  # agent calling one of these ends the run
    tags: list[str] = Field(default_factory=list)


# ---------- trajectory (what actually happened) ----------


class ToolEvent(BaseModel):
    name: str
    input: dict[str, Any]
    output: str
    error: bool = False


class Turn(BaseModel):
    role: Literal["user", "agent"]
    text: str
    tool_events: list[ToolEvent] = Field(default_factory=list)
    latency_ms: int = 0
    usage: Usage = Field(default_factory=Usage)


class Trajectory(BaseModel):
    scenario_id: str
    target: str
    turns: list[Turn]
    ended_by: Literal["simulator_done", "terminal_tool", "max_turns", "error"]
    error: str | None = None
    sim_usage: Usage = Field(default_factory=Usage)

    @property
    def agent_usage(self) -> Usage:
        u = Usage()
        for t in self.turns:
            if t.role == "agent":
                u = u + t.usage
        return u

    def transcript(self) -> str:
        lines: list[str] = []
        for i, t in enumerate(self.turns):
            who = "USER" if t.role == "user" else "AGENT"
            lines.append(f"[{i}] {who}: {t.text}".rstrip())
            for ev in t.tool_events:
                flag = " (ERROR)" if ev.error else ""
                lines.append(f"[{i}]   tool {ev.name}({ev.input}) -> {ev.output[:300]}{flag}")
        return "\n".join(lines)


# ---------- judge output ----------


class CriterionResult(BaseModel):
    id: str = Field(description="criterion id from the rubric, copied exactly")
    passed: bool
    evidence: str = Field(description="quote or cite turn indices like [3] that justify the decision")
    confidence: float = Field(ge=0, le=1)


class Verdict(BaseModel):
    criteria: list[CriterionResult]
    task_completed: bool = Field(description="did the user achieve the scenario goal")
    summary: str = Field(description="two sentences max")


class ScoredTrajectory(BaseModel):
    trajectory: Trajectory
    rubric: list[Criterion]
    verdict: Verdict | None = None
    judge_usage: Usage = Field(default_factory=Usage)

    def score(self) -> float | None:
        """Weighted pass rate in [0, 1]; None if unjudged."""
        if self.verdict is None:
            return None
        weights = {c.id: c.weight for c in self.rubric}
        total = sum(weights.values())
        if total == 0:
            return 0.0
        got = sum(weights.get(c.id, 0.0) for c in self.verdict.criteria if c.passed)
        return got / total

    def passed_ids(self) -> set[str]:
        return {c.id for c in (self.verdict.criteria if self.verdict else []) if c.passed}


# ---------- simulator output ----------


class SimTurn(BaseModel):
    message: str = Field(description="what the simulated user says next; empty if done")
    done: bool = Field(description="true if the goal is reached or the user would give up")
    reason: str = Field(description="one line: why continue or why stop")
