from __future__ import annotations

import os
from typing import AsyncGenerator

import httpx

from .models import MessagesRequest

ANTHROPIC_API_URL = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


async def stream_messages(request: MessagesRequest) -> AsyncGenerator[bytes, None]:
    headers = {
        "Authorization": f"Bearer {ANTHROPIC_API_KEY}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", ANTHROPIC_API_URL, headers=headers, json=request.dict()) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                yield chunk
