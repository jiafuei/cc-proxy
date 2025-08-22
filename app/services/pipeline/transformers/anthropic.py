from urllib.parse import urlparse

from ....config.models import ConfigModel
from ..interfaces import RequestTransformer, ResponseTransformer, StreamTransformer
from ..models import ProxyRequest, ProxyResponse, StreamChunk


class AnthropicRequestTransformer(RequestTransformer):
    def __init__(self, config: ConfigModel):
        self._api_url = config.anthropic_api_url
        self._api_key = config.anthropic_api_key
        self.hostname = urlparse(self._api_url).hostname if self._api_url else None

    async def transform(self, proxy_request: ProxyRequest) -> ProxyRequest:
        """Prepare request for Anthropic API"""
        headers = proxy_request.headers.copy()

        # Remove headers that should not be forwarded
        for header in ['content-length', 'accept', 'connection']:
            headers.pop(header, None)

        # Set Anthropic-specific headers
        if self.hostname:
            headers['host'] = self.hostname
        if self._api_key:
            headers['authorization'] = f'Bearer {self._api_key}'

        # Update the proxy request with modified headers and URL
        proxy_request.headers = headers
        proxy_request.url = self._api_url

        return proxy_request


class AnthropicResponseTransformer(ResponseTransformer):
    async def transform(self, proxy_response: ProxyResponse, proxy_request: ProxyRequest) -> ProxyResponse:
        """Transform Anthropic response (pass-through for now)"""
        return proxy_response


class AnthropicStreamTransformer(StreamTransformer):
    async def transform_chunk(self, chunk: StreamChunk, proxy_request: ProxyRequest) -> StreamChunk:
        """Transform Anthropic stream chunks (pass-through for now)"""
        return chunk
