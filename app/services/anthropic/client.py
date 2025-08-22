import os
from typing import AsyncGenerator

import httpx

from .models import MessagesRequest

ANTHROPIC_API_URL = os.environ.get('ANTHROPIC_API_URL', 'https://api.anthropic.com/v1/messages')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')


class AnthropicStreamingService:
    def __init__(self, client: httpx.AsyncClient, api_url: str = ANTHROPIC_API_URL, api_key: str = ANTHROPIC_API_KEY):
        self._client = client
        self._api_url = api_url
        self._api_key = api_key

    async def stream_response(self, request: MessagesRequest) -> AsyncGenerator[bytes, None]:
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Accept': 'text/event-stream',
            'Content-Type': 'application/json',
        }
        async with self._client.stream('POST', self._api_url, headers=headers, json=request.dict()) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                yield chunk
