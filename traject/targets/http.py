"""Target behind an HTTP endpoint. Contract:

POST {url}  body: {"session_id": str, "message": str}
200 body:   {"text": str, "tool_events": [{"name", "input", "output", "error"?}]}
"""

from __future__ import annotations

import uuid

import httpx

from ..models import ToolEvent
from .base import AgentReply, TargetConfig


class HTTPTarget:
    def __init__(self, cfg: TargetConfig):
        if not cfg.url:
            raise ValueError("http target needs `url`")
        self.cfg = cfg
        self.name = cfg.name

    def new_session(self) -> HTTPSession:
        return HTTPSession(self.cfg.url, headers=self.cfg.extra.get("headers", {}))


class HTTPSession:
    def __init__(self, url: str, headers: dict[str, str]):
        self.url = url
        self.session_id = uuid.uuid4().hex
        self.client = httpx.AsyncClient(headers=headers, timeout=120)

    async def respond(self, user_message: str) -> AgentReply:
        r = await self.client.post(self.url, json={"session_id": self.session_id, "message": user_message})
        r.raise_for_status()
        body = r.json()
        return AgentReply(
            text=body.get("text", ""),
            tool_events=[ToolEvent.model_validate(e) for e in body.get("tool_events", [])],
        )
