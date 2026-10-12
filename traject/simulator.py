"""Simulated user. An LLM plays a persona with a goal and decides when the conversation is over."""

from __future__ import annotations

from .models import Message, Scenario, SimTurn, Trajectory, Turn, Usage
from .providers.base import Provider

SIM_SYSTEM = """You are role-playing a USER talking to an AI agent, for the purpose of testing that agent.

Stay in character. Never mention that you are simulated or that this is a test.
Write only what this user would actually type: short, natural, sometimes imperfect.
Do not help the agent. Do not answer questions the persona would not know.
If the agent asks something, respond as the persona would — which may be evasive or off-topic
if the persona is like that.

Set done=true when the goal is clearly achieved, or when this user would realistically give up
or leave (e.g. the agent is stuck in a loop, is rude, or the conversation has naturally ended).
When done=true, message may be a brief closing line or empty."""


def _render_persona(scenario: Scenario) -> str:
    return (
        f"PERSONA:\n{scenario.persona.strip()}\n\n"
        f"GOAL (what this user is trying to get out of the conversation):\n{scenario.goal.strip()}"
    )


def _history_as_sim_view(turns: list[Turn]) -> list[Message]:
    """From the simulator's seat, the AGENT is the 'user' and the simulated user is the 'assistant'."""
    msgs: list[Message] = []
    for t in turns:
        if t.role == "agent":
            msgs.append(Message(role="user", content=t.text or "(agent said nothing)"))
        else:
            msgs.append(Message(role="assistant", content=t.text))
    return msgs


class Simulator:
    def __init__(self, provider: Provider):
        self.provider = provider

    async def next_turn(self, scenario: Scenario, turns: list[Turn]) -> tuple[SimTurn, Usage]:
        msgs = _history_as_sim_view(turns)
        if not msgs:
            msgs.append(Message(role="user", content="(The conversation is starting. Say your opening line.)"))
        elif msgs[-1].role == "assistant":
            msgs.append(Message(role="user", content="(agent has not replied yet; continue or wait)"))
        system = SIM_SYSTEM + "\n\n" + _render_persona(scenario)
        return await self.provider.parse(system, msgs, SimTurn)


__all__ = ["SIM_SYSTEM", "Simulator", "Trajectory"]
