"""Scripted provider for tests and offline demos. No network."""

from __future__ import annotations

from collections import deque
from typing import Any, TypeVar

from pydantic import BaseModel

from ..models import Message, Reply, ToolSpec, Usage

T = TypeVar("T", bound=BaseModel)


class MockProvider:
    model = "mock"

    def __init__(
        self,
        replies: list[Reply] | None = None,
        parsed: list[BaseModel] | None = None,
        parse_fn: Any | None = None,
    ):
        """replies: consumed in order by chat(); last one repeats.
        parsed: consumed in order by parse(); last one repeats.
        parse_fn(schema, messages) -> BaseModel: overrides `parsed` when set."""
        self._replies = deque(replies or [Reply(text="ok")])
        self._parsed = deque(parsed or [])
        self._parse_fn = parse_fn
        self.chat_calls: list[list[Message]] = []
        self.parse_calls: list[tuple[type, list[Message]]] = []

    async def chat(
        self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Reply:
        self.chat_calls.append(list(messages))
        r = self._replies[0] if len(self._replies) == 1 else self._replies.popleft()
        return r.model_copy(update={"usage": Usage(input_tokens=10, output_tokens=5)})

    async def parse(self, system: str, messages: list[Message], schema: type[T]) -> tuple[T, Usage]:
        self.parse_calls.append((schema, list(messages)))
        if self._parse_fn is not None:
            return self._parse_fn(schema, messages), Usage(input_tokens=10, output_tokens=5)
        if not self._parsed:
            return _default_for(schema, messages), Usage(input_tokens=10, output_tokens=5)
        p = self._parsed[0] if len(self._parsed) == 1 else self._parsed.popleft()
        if not isinstance(p, schema):
            raise TypeError(f"scripted {type(p).__name__} but caller asked for {schema.__name__}")
        return p, Usage(input_tokens=10, output_tokens=5)


def _default_for(schema: type[T], messages: list[Message]) -> T:
    """Sensible instance when nothing is scripted, so `--sim-provider mock` gives a dry run."""
    from ..models import CriterionResult, SimTurn, Verdict

    if schema is SimTurn:
        n = sum(1 for m in messages if m.role == "assistant")
        return SimTurn(message=f"(mock user turn {n + 1})", done=n >= 2, reason="mock")  # type: ignore[return-value]
    if schema is Verdict:
        import re

        text = messages[-1].content if messages else ""
        ids = re.findall(r"^- id=([^:]+):", text, flags=re.M)
        return Verdict(  # type: ignore[return-value]
            criteria=[CriterionResult(id=i, passed=False, evidence="mock judge", confidence=0.0) for i in ids],
            task_completed=False,
            summary="mock verdict: nothing was actually judged",
        )
    raise RuntimeError(f"MockProvider.parse: nothing scripted for {schema.__name__}")
