"""Target implemented as any Python object with `async respond(user_message: str) -> AgentReply | str`."""

from __future__ import annotations

from .base import AgentReply, TargetConfig, import_attr


class PythonTarget:
    def __init__(self, cfg: TargetConfig):
        if not cfg.entrypoint:
            raise ValueError("python target needs `entrypoint: module:factory`")
        self.cfg = cfg
        self.name = cfg.name
        self._factory = import_attr(cfg.entrypoint)

    def new_session(self) -> PythonSession:
        return PythonSession(self._factory(**self.cfg.extra))


class PythonSession:
    def __init__(self, inner):
        self.inner = inner

    async def respond(self, user_message: str) -> AgentReply:
        out = await self.inner.respond(user_message)
        if isinstance(out, AgentReply):
            return out
        return AgentReply(text=str(out))
