from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from ..models import Message, Reply, ToolSpec, Usage

T = TypeVar("T", bound=BaseModel)


class Provider(Protocol):
    """Minimal LLM surface traject needs: free-form chat with tools, and schema-constrained parse."""

    model: str

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> Reply: ...

    async def parse(
        self,
        system: str,
        messages: list[Message],
        schema: type[T],
    ) -> tuple[T, Usage]: ...


