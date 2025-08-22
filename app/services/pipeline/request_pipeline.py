from typing import List

from .interfaces import RequestTransformer
from .models import ProxyRequest


class RequestPipeline:
    def __init__(self, transformers: List[RequestTransformer]):
        self._transformers = transformers

    async def execute(self, proxy_request: ProxyRequest) -> ProxyRequest:
        """Execute transformer chain on proxy request"""
        current_request = proxy_request

        for transformer in self._transformers:
            current_request = await transformer.transform(current_request)

        return current_request
