"""Target whose agent loop traject runs itself: system prompt + Python tools + provider."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..models import Message, ToolEvent, ToolSpec, Usage
from ..providers import get_provider
from .base import AgentReply, TargetConfig, import_attr


@dataclass
class Tool:
    """A tool the agent under test can call. fn(**input) -> str | dict | Any (JSON-dumped)."""

    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, input_schema=self.input_schema)


class LLMAgentTarget:
    def __init__(self, cfg: TargetConfig):
        self.cfg = cfg
        self.name = cfg.name
        self.tools: list[Tool] = list(import_attr(cfg.tools)) if cfg.tools else []
        self._provider_factory = lambda: get_provider(cfg.provider, cfg.model)

    def new_session(self) -> LLMAgentSession:
        return LLMAgentSession(self, self._provider_factory())


class LLMAgentSession:
    def __init__(self, target: LLMAgentTarget, provider: Any):
        self.target = target
        self.provider = provider
        self.history: list[Message] = []
        self._tool_map = {t.name: t for t in target.tools}
        self._specs = [t.spec() for t in target.tools]

    async def respond(self, user_message: str) -> AgentReply:
        self.history.append(Message(role="user", content=user_message))
        events: list[ToolEvent] = []
        usage = Usage()
        text = ""
        for _ in range(self.target.cfg.max_tool_iterations):
            reply = await self.provider.chat(self.target.cfg.system, self.history, self._specs or None)
            usage = usage + reply.usage
            self.history.append(
                Message(role="assistant", content=reply.text, tool_calls=reply.tool_calls)
            )
            text = reply.text or text
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                out, err = await self._run_tool(call.name, call.input)
                events.append(ToolEvent(name=call.name, input=call.input, output=out, error=err))
                self.history.append(
                    Message(role="tool", tool_call_id=call.id, name=call.name, content=out)
                )
        return AgentReply(text=text, tool_events=events, usage=usage)

    async def _run_tool(self, name: str, inp: dict[str, Any]) -> tuple[str, bool]:
        tool = self._tool_map.get(name)
        if tool is None:
            return f"unknown tool {name!r}", True
        try:
            result = tool.fn(**inp)
            if inspect.isawaitable(result):
                result = await result
        except Exception as e:  # tool bugs are part of what we're evaluating; record, don't crash
            return f"{type(e).__name__}: {e}", True
        if isinstance(result, str):
            return result, False
        return json.dumps(result, default=str), False
