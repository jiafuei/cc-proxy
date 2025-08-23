from typing import AsyncGenerator

import httpx

from app.services.pipeline.models import ProxyRequest


class HttpClientService:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def stream_request(self, prepared_request: ProxyRequest) -> AsyncGenerator[bytes, None]:
        """Execute streaming HTTP request"""
        async with self._client.stream(
            'POST',
            prepared_request.url,
            headers=prepared_request.headers,
            json=prepared_request.claude_request.model_dump(mode='json', exclude_none=True, by_alias=True),
            params=prepared_request.params,
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk

    async def post_request(self, prepared_request: ProxyRequest) -> httpx.Response:
        """Execute non-streaming HTTP request"""
        return await self._client.post(
            prepared_request.url,
            headers=prepared_request.headers,
            json=prepared_request.claude_request.model_dump(mode='json', exclude_none=True, by_alias=True),
            params=prepared_request.params,
        )
