from abc import ABC, abstractmethod
from typing import AsyncIterator

from .models import ProxyRequest, ProxyResponse, StreamChunk


class RequestTransformer(ABC):
    @abstractmethod
    async def transform(self, proxy_request: ProxyRequest) -> ProxyRequest:
        """Transform outgoing request data"""
        pass


class ResponseTransformer(ABC):
    @abstractmethod
    async def transform(self, proxy_response: ProxyResponse, proxy_request: ProxyRequest) -> ProxyResponse:
        """Transform incoming response data"""
        pass


class StreamTransformer(ABC):
    @abstractmethod
    async def transform_chunk(self, chunk: StreamChunk, proxy_request: ProxyRequest) -> StreamChunk:
        """Transform individual stream chunks"""
        pass

    async def transform_stream(self, stream: AsyncIterator[StreamChunk], proxy_request: ProxyRequest) -> AsyncIterator[StreamChunk]:
        """Transform entire stream - default implementation applies transform_chunk to each chunk"""
        async for chunk in stream:
            transformed_chunk = await self.transform_chunk(chunk, proxy_request)
            yield transformed_chunk
