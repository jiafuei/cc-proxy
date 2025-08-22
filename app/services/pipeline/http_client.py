from typing import Any, AsyncGenerator, Dict

import httpx


class HttpClientService:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def stream_request(self, prepared_request: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
        """Execute streaming HTTP request"""
        async with self._client.stream(
            'POST', prepared_request['url'], headers=prepared_request['headers'], json=prepared_request['json'], params=prepared_request.get('params')
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk

    async def post_request(self, prepared_request: Dict[str, Any]) -> httpx.Response:
        """Execute non-streaming HTTP request"""
        return await self._client.post(prepared_request['url'], headers=prepared_request['headers'], json=prepared_request['json'], params=prepared_request.get('params'))
