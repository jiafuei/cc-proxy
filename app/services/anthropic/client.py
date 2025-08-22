from typing import AsyncGenerator, Mapping

from fastapi import Request
from urllib.parse import urlparse
import httpx

from ...config.models import ConfigModel
from .models import MessagesRequest


class AnthropicStreamingService:
    def __init__(self, client: httpx.AsyncClient, config: ConfigModel):
        self._client = client
        self._api_url = config.anthropic_api_url
        self._api_key = config.anthropic_api_key
        self.hostname = urlparse(self._api_url).hostname

    async def stream_response(self, payload: MessagesRequest, request: Request) -> AsyncGenerator[bytes, None]:
        headers = {k:v for k,v in request.headers.items()} if request.headers else {}
        del headers['content-length']
        del headers['accept']
        del headers['connection']
        headers['host'] = self.hostname
        if self._api_key:
            headers['authorization'] = f'Bearer {self._api_key}'
        # async with self._client.stream('POST', self._api_url, headers=headers, json=request.model_dump()) as resp:
        async with self._client.stream('POST', self._api_url, headers=headers, json=payload, params=request.query_params) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                yield chunk
