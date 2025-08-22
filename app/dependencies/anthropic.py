from __future__ import annotations

from functools import lru_cache

import httpx

from app.services.anthropic.streaming import AnthropicStreamingService


@lru_cache(maxsize=1)
def get_anthropic_service() -> AnthropicStreamingService:
    client = httpx.AsyncClient(timeout=60.0)
    return AnthropicStreamingService(client=client)
