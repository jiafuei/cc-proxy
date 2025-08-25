"""Enhanced provider system for the simplified architecture."""

from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import orjson
from fastapi import Request

from app.common.models import ClaudeRequest
from app.config.log import get_logger
from app.config.user_models import ProviderConfig
from app.services.transformer_loader import TransformerLoader
from app.services.transformers import RequestTransformer, ResponseTransformer

logger = get_logger(__name__)


class Provider:
    """Handles requests to a specific API provider with transformer support."""

    def __init__(self, config: ProviderConfig, transformer_loader: TransformerLoader):
        self.config = config
        self.name = config.name
        self.transformer_loader = transformer_loader
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(self.config.timeout), http2=True)

        # Load transformers
        self.request_transformers: List[RequestTransformer] = []
        self.response_transformers: List[ResponseTransformer] = []
        self._load_transformers()

    def _load_transformers(self):
        """Load request and response transformers from configuration."""
        try:
            request_transformers = self.config.transformers.get('request', [])
            response_transformers = self.config.transformers.get('response', [])

            self.request_transformers = self.transformer_loader.load_transformers(request_transformers)
            self.response_transformers = self.transformer_loader.load_transformers(response_transformers)

            logger.info(f"Provider '{self.config.name}' loaded {len(self.request_transformers)} request transformers and {len(self.response_transformers)} response transformers")
        except Exception as e:
            logger.error(f"Failed to load transformers for provider '{self.config.name}': {e}")
            # Continue with empty transformer lists

    async def process_request(self, request: ClaudeRequest, original_request: Request) -> AsyncIterator[bytes]:
        """Process a request through the provider with transformer support.

        Args:
            request: Claude API request

        Yields:
            SSE-formatted response chunks
        """
        try:
            # 1. Convert ClaudeRequest to Dict and apply request transformers sequentially
            current_request = request.to_dict()  # Use to_dict() method
            current_headers = dict(original_request.headers)  # Copy headers
            
            for transformer in self.request_transformers:
                current_request, current_headers = await transformer.transform(current_request, current_headers, self.config, original_request)

            logger.debug(f'Request transformed, stream={current_request.get("stream", False)}')

            # 2. Check final stream flag after transformations
            should_stream = current_request.get('stream', False)

            # 3. Route based on stream flag
            if should_stream:
                # Stream from provider
                async for chunk in self._stream_request(current_request, current_headers):
                    # Apply response transformers to each chunk
                    transformed_chunk = chunk
                    for transformer in self.response_transformers:
                        transformed_chunk = await transformer.transform_chunk(transformed_chunk)
                    yield transformed_chunk
            else:
                # Non-streaming request
                response = await self._send_request(current_request, current_headers)

                # Apply response transformers to full response
                transformed_response = response
                for transformer in self.response_transformers:
                    transformed_response = await transformer.transform_response(transformed_response)

                # Convert to SSE format for consistent output
                async for chunk in self._convert_response_to_sse(transformed_response):
                    yield chunk

        except Exception as e:
            logger.error(f"Error processing request in provider '{self.config.name}': {e}", exc_info=True)
            # Yield error in SSE format
            error_chunk = self._format_error_as_sse(str(e))
            yield error_chunk

    async def _stream_request(self, request_data: Dict[str, Any], headers: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Make a streaming HTTP request to the provider."""
        # Use headers from transformers (which may include auth)
        final_headers = headers

        logger.debug(f'Streaming request to {self.config.url}')

        async with self.http_client.stream('POST', self.config.url, json=request_data, headers=final_headers) as response:
            response.raise_for_status()

            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk

    async def _send_request(self, request_data: Dict[str, Any], headers: Dict[str, Any]) -> Dict[str, Any]:
        """Make a non-streaming HTTP request to the provider."""
        # Use headers from transformers (which may include auth)
        final_headers = headers

        logger.debug(f'Non-streaming request to {self.config.url}')

        response = await self.http_client.post(self.config.url, json=request_data, headers=final_headers)
        response.raise_for_status()

        return response.json()

    async def _convert_response_to_sse(self, response: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Convert a non-streaming response to SSE format."""
        # Create message_start event
        message_start = {
            'type': 'message_start',
            'message': {
                'id': response.get('id', 'msg_01'),
                'type': 'message',
                'role': 'assistant',
                'content': [],
                'model': response.get('model', 'unknown'),
                'stop_reason': None,
                'stop_sequence': None,
                'usage': response.get('usage', {'input_tokens': 0, 'output_tokens': 0}),
            },
        }
        yield f'event: message_start\ndata: {orjson.dumps(message_start).decode()}\n\n'.encode()

        # Create content_block_start event
        content_block_start = {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}}
        yield f'event: content_block_start\ndata: {orjson.dumps(content_block_start).decode()}\n\n'.encode()

        # Send content as deltas
        content = response.get('content', [])
        if content and isinstance(content, list) and len(content) > 0:
            text_content = content[0].get('text', '') if content[0].get('type') == 'text' else ''

            # Split text into chunks for streaming effect
            chunk_size = 50
            for i in range(0, len(text_content), chunk_size):
                chunk_text = text_content[i : i + chunk_size]
                delta = {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': chunk_text}}
                yield f'event: content_block_delta\ndata: {orjson.dumps(delta).decode()}\n\n'.encode()

        # Create content_block_stop event
        content_block_stop = {'type': 'content_block_stop', 'index': 0}
        yield f'event: content_block_stop\ndata: {orjson.dumps(content_block_stop).decode()}\n\n'.encode()

        # Create message_stop event
        message_stop = {'type': 'message_stop'}
        yield f'event: message_stop\ndata: {orjson.dumps(message_stop).decode()}\n\n'.encode()

    def _format_error_as_sse(self, error_message: str) -> bytes:
        """Format an error as SSE event."""
        error_data = {'type': 'error', 'error': {'type': 'api_error', 'message': error_message}}
        return f'event: error\ndata: {orjson.dumps(error_data).decode()}\n\n'.encode()

    def supports_model(self, model_id: str) -> bool:
        """Check if this provider supports the given model."""
        return model_id in self.config.models

    async def close(self):
        """Clean up resources."""
        await self.http_client.aclose()

    def __str__(self) -> str:
        return f'Provider(name={self.config.name}, models={len(self.config.models)})'


class ProviderManager:
    """Manages multiple providers."""

    def __init__(self, providers_config: List[ProviderConfig], transformer_loader: TransformerLoader):
        self.providers: Dict[str, Provider] = {}
        self.model_to_provider: Dict[str, str] = {}
        self._load_providers(providers_config, transformer_loader)

    def _load_providers(self, providers_config: List[ProviderConfig], transformer_loader: TransformerLoader):
        """Load providers from configuration."""
        for provider_config in providers_config:
            try:
                provider = Provider(provider_config, transformer_loader)
                self.providers[provider_config.name] = provider

                # Build model-to-provider mapping
                for model in provider_config.models:
                    self.model_to_provider[model] = provider_config.name

                logger.info(f"Loaded provider '{provider_config.name}' with {len(provider_config.models)} models")

            except Exception as e:
                logger.error(f"Failed to load provider '{provider_config.name}': {e}", exc_info=True)

    def get_provider_for_model(self, model_id: str) -> Optional[Provider]:
        """Get the provider that supports the given model."""
        provider_name = self.model_to_provider.get(model_id)
        if provider_name:
            return self.providers.get(provider_name)
        return None

    def get_provider_by_name(self, name: str) -> Optional[Provider]:
        """Get provider by name."""
        return self.providers.get(name)

    def list_providers(self) -> List[str]:
        """List all provider names."""
        return list(self.providers.keys())

    def list_models(self) -> List[str]:
        """List all supported models."""
        return list(self.model_to_provider.keys())

    async def close_all(self):
        """Close all providers."""
        for provider in self.providers.values():
            await provider.close()
