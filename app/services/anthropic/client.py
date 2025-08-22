from typing import AsyncGenerator, Mapping

import httpx

from ...config.models import ConfigModel
from .models import MessagesRequest


class AnthropicStreamingService:
    def __init__(self, client: httpx.AsyncClient, config: ConfigModel):
        self._client = client
        self._api_url = config.anthropic_api_url
        self._api_key = config.anthropic_api_key

    async def stream_response(self, request: MessagesRequest, headers_orig: Mapping[str, str]) -> AsyncGenerator[bytes, None]:
        headers = {k:v for k,v in headers_orig.items()} if headers_orig else {}
        if self._api_key:
            headers['Authorization'] = f'Bearer {self._api_key}'
        async with self._client.stream('POST', self._api_url, headers=headers, json=request.model_dump()) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                yield chunk
