"""A Target is the agent under test. traject talks to it one user message at a time."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, Field

from ..models import Message, ToolEvent, Usage


class AgentReply(BaseModel):
    text: str
    tool_events: list[ToolEvent] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class Target(Protocol):
    name: str

    def new_session(self) -> TargetSession: ...


class TargetSession(Protocol):
    """One conversation. The target owns its own history so HTTP/stateful agents work too."""

    async def respond(self, user_message: str) -> AgentReply: ...


class TargetConfig(BaseModel):
    """YAML schema for a target.

    kind: llm      -> traject runs the agent loop itself (system prompt + tools + provider)
    kind: http     -> POST {session_id, message} to url, expects {"text": ..., "tool_events": [...]}
    kind: python   -> `entrypoint: pkg.mod:factory`, factory() returns an object with async respond(str)
    """

    name: str
    kind: str = "llm"
    provider: str = "anthropic"
    model: str | None = None
    system: str = ""
    tools: str | None = None  # "pkg.mod:TOOLS" -> list[Tool]
    max_tool_iterations: int = 8
    url: str | None = None
    entrypoint: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def load_target(
    path: str | Path, provider: str | None = None, model: str | None = None
) -> Target:
    """provider/model override the YAML (e.g. `--target-provider mock` for a dry run)."""
    cfg = TargetConfig.model_validate(yaml.safe_load(Path(path).read_text()))
    if provider:
        cfg.provider = provider
    if model:
        cfg.model = model
    if cfg.kind == "llm":
        from .llm_agent import LLMAgentTarget

        return LLMAgentTarget(cfg)
    if cfg.kind == "http":
        from .http import HTTPTarget

        return HTTPTarget(cfg)
    if cfg.kind == "python":
        from .python_ import PythonTarget

        return PythonTarget(cfg)
    raise ValueError(f"unknown target kind {cfg.kind!r}")


def import_attr(spec: str) -> Any:
    """'pkg.mod:attr' -> attr"""
    mod_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:attr', got {spec!r}")
    return getattr(importlib.import_module(mod_name), attr)


__all__ = ["AgentReply", "Message", "Target", "TargetConfig", "TargetSession", "import_attr", "load_target"]
