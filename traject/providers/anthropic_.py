"""Anthropic provider. Uses the official SDK: messages.create for chat, messages.parse for schemas."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from ..models import Message, Reply, ToolCall, ToolSpec, Usage

T = TypeVar("T", bound=BaseModel)


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert neutral messages to Anthropic content blocks.

    Consecutive tool results are grouped into ONE user message, which the API requires
    when the assistant made parallel tool calls.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                    and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


class AnthropicProvider:
    DEFAULT_MODEL = "claude-opus-5-5"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens
        self.client = AsyncAnthropic()

    async def chat(
        self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Reply:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=to_anthropic_messages(messages),
            **kwargs,
        )
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                raw = block.input
                if isinstance(raw, str):  # defensive: some SDK paths hand back serialized JSON
                    raw = json.loads(raw)
                calls.append(ToolCall(id=block.id, name=block.name, input=raw))
        return Reply(
            text="".join(text_parts),
            tool_calls=calls,
            usage=Usage(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens),
            stop_reason=resp.stop_reason,
        )

    async def parse(self, system: str, messages: list[Message], schema: type[T]) -> tuple[T, Usage]:
        resp = await self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=to_anthropic_messages(messages),
            output_format=schema,
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model refused structured request: {resp.stop_details}")
        return resp.parsed_output, Usage(
            input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens
        )
