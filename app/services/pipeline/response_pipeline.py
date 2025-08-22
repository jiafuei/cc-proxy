from typing import List

from .interfaces import ResponseTransformer, StreamTransformer
from .models import ProxyRequest, ProxyResponse, StreamChunk


class ResponsePipeline:
    def __init__(self, transformers: List[ResponseTransformer], stream_transformers: List[StreamTransformer] = None):
        self._transformers = transformers
        self._stream_transformers = stream_transformers or []

    async def execute(self, proxy_response: ProxyResponse, proxy_request: ProxyRequest) -> ProxyResponse:
        """Execute transformer chain on response data"""
        current_response = proxy_response

        for transformer in self._transformers:
            current_response = await transformer.transform(current_response, proxy_request)

        return current_response

    async def execute_stream_chunk(self, chunk: StreamChunk, proxy_request: ProxyRequest) -> StreamChunk:
        """Execute stream transformer chain on individual chunk"""
        current_chunk = chunk

        for transformer in self._stream_transformers:
            current_chunk = await transformer.transform_chunk(current_chunk, proxy_request)

        return current_chunk
