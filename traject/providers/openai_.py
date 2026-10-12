"""OpenAI provider. chat.completions.create for chat, chat.completions.parse for schemas."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from ..models import Message, Reply, ToolCall, ToolSpec, Usage

T = TypeVar("T", bound=BaseModel)


def to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.input)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


class OpenAIProvider:
    DEFAULT_MODEL = "gpt-4.1"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = AsyncOpenAI()

    async def chat(
        self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Reply:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=to_openai_messages(system, messages),
            **kwargs,
        )
        choice = resp.choices[0]
        calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                input=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (choice.message.tool_calls or [])
            if getattr(tc, "function", None) is not None
        ]
        usage = Usage(
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
        return Reply(
            text=choice.message.content or "",
            tool_calls=calls,
            usage=usage,
            stop_reason=choice.finish_reason,
        )

    async def parse(self, system: str, messages: list[Message], schema: type[T]) -> tuple[T, Usage]:
        resp = await self.client.chat.completions.parse(
            model=self.model,
            messages=to_openai_messages(system, messages),
            response_format=schema,
        )
        msg = resp.choices[0].message
        if msg.parsed is None:
            raise RuntimeError(f"structured parse failed: refusal={getattr(msg, 'refusal', None)}")
        usage = Usage(
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
        return msg.parsed, usage
