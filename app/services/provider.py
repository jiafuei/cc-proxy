"""Enhanced provider system for the simplified architecture."""

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
import orjson
from fastapi import Request

from app.common.dumper import Dumper, DumpHandles
from app.common.models import AnthropicRequest
from app.config.log import get_logger
from app.config.user_models import ModelConfig, ProviderConfig
from app.services.transformer_loader import TransformerLoader
from app.services.transformers import RequestTransformer, ResponseTransformer

logger = get_logger(__name__)


@dataclass
class ModelMapping:
    """Maps model alias to provider name and actual model ID."""

    provider_name: str
    model_id: str


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

    async def process_request(self, request: AnthropicRequest, original_request: Request, routing_key: str, dumper: Dumper, dumper_handles: DumpHandles) -> AsyncIterator[bytes]:
        """Process a request through the provider with transformer support.

        Args:
            request: Claude API request

        Yields:
            SSE-formatted response chunks
        """
        # 1. Convert AnthropicRequest to Dict and apply request transformers sequentially
        current_request = request.to_dict()  # Use to_dict() method
        current_headers = dict(original_request.headers)  # Copy headers
        config = self.config.model_copy()

        logger.debug(f'Before transform, headers={current_headers}')
        for transformer in self.request_transformers:
            transform_params = {
                'request': current_request,
                'headers': current_headers,
                'provider_config': config,
                'original_request': original_request,
                'routing_key': routing_key,
            }
            current_request, current_headers = await transformer.transform(transform_params)

        logger.debug(f'Request transformed, stream={current_request.get("stream", False)}, headers={current_headers}')

        # Dump transformed request after all transformers are applied
        dumper.write_transformed_request(dumper_handles, current_request)

        # 2. Check final stream flag after transformations
        should_stream = current_request.get('stream', False)

        # 3. Route based on stream flag
        if should_stream:
            # Initialize chunk params once to preserve transformer state
            chunk_params = {}

            # Stream from provider
            async for chunk in self._stream_request(config, current_request, current_headers):
                # Dump pre-transformed response chunk
                dumper.write_pretransformed_response(dumper_handles, chunk)

                # Apply response transformers to each chunk
                current_chunks = [chunk]
                for transformer in self.response_transformers:
                    next_chunks = []
                    for current_chunk in current_chunks:
                        chunk_params.update(
                            {
                                'chunk': current_chunk,
                                'request': current_request,
                                'final_headers': current_headers,
                                'provider_config': config,
                                'original_request': original_request,
                            }
                        )
                        async for transformed_chunk in transformer.transform_chunk(chunk_params):
                            next_chunks.append(transformed_chunk)
                    current_chunks = next_chunks

                # Yield all final chunks
                for final_chunk in current_chunks:
                    yield final_chunk
        else:
            # Non-streaming request
            response = await self._send_request(config, current_request, current_headers)

            # Dump pre-transformed response
            response_text = response.text
            dumper.write_pretransformed_response(dumper_handles, response_text)

            # Apply response transformers to full response
            transformed_response = response.json()
            response_params = {}  # Initialize once outside loop for consistency
            for transformer in self.response_transformers:
                response_params.update(
                    {
                        'response': transformed_response,
                        'request': current_request,
                        'final_headers': current_headers,
                        'provider_config': config,
                        'original_request': original_request,
                    }
                )
                transformed_response = await transformer.transform_response(response_params)

            # Convert to SSE format for consistent output
            async for chunk in self._convert_response_to_sse(transformed_response):
                yield chunk

    async def _stream_request(self, config: ProviderConfig, request_data: Dict[str, Any], headers: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Make a streaming HTTP request to the provider."""
        # Use headers from transformers (which may include auth)
        final_headers = headers

        logger.debug(f'Streaming request to {config.url}')

        async with self.http_client.stream('POST', config.url, json=request_data, headers=final_headers) as response:
            try:
                response.raise_for_status()

                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
            except httpx.HTTPStatusError:
                await response.aread()
                raise

    async def _send_request(self, config: ProviderConfig, request_data: Dict[str, Any], headers: Dict[str, Any]) -> httpx.Response:
        """Make a non-streaming HTTP request to the provider."""
        # Use headers from transformers (which may include auth)
        final_headers = headers

        logger.debug(f'Non-streaming request to {config.url}')

        response = await self.http_client.post(config.url, json=request_data, headers=final_headers)
        try:
            response.raise_for_status()
        except:
            await response.aread()
            raise

        return response

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

        # Process each content block
        content = response.get('content', [])
        for index, content_block in enumerate(content):
            block_type = content_block.get('type', 'text')

            # Create content_block_start event for this block
            content_block_start = {'type': 'content_block_start', 'index': index, 'content_block': self._create_initial_content_block(content_block, block_type)}
            yield f'event: content_block_start\ndata: {orjson.dumps(content_block_start).decode()}\n\n'.encode()

            # Generate deltas based on content type
            if block_type == 'thinking':
                async for delta_event in self._generate_thinking_deltas(content_block, index):
                    yield delta_event
            elif block_type == 'text':
                async for delta_event in self._generate_text_deltas(content_block, index):
                    yield delta_event
            elif block_type == 'tool_use':
                async for delta_event in self._generate_tool_use_deltas(content_block, index):
                    yield delta_event

            # Create content_block_stop event
            content_block_stop = {'type': 'content_block_stop', 'index': index}
            yield f'event: content_block_stop\ndata: {orjson.dumps(content_block_stop).decode()}\n\n'.encode()

        # Create message_delta event with final stop reason
        message_delta = {
            'type': 'message_delta',
            'delta': {'stop_reason': response.get('stop_reason', 'end_turn'), 'stop_sequence': response.get('stop_sequence')},
            'usage': response.get('usage', {'input_tokens': 0, 'output_tokens': 0}),
        }
        yield f'event: message_delta\ndata: {orjson.dumps(message_delta).decode()}\n\n'.encode()

        # Create message_stop event
        message_stop = {'type': 'message_stop'}
        yield f'event: message_stop\ndata: {orjson.dumps(message_stop).decode()}\n\n'.encode()

    def _create_initial_content_block(self, content_block: Dict[str, Any], block_type: str) -> Dict[str, Any]:
        """Create the initial content block structure for content_block_start event."""
        if block_type == 'thinking':
            return {'type': 'thinking', 'thinking': '', 'signature': ''}
        elif block_type == 'text':
            return {'type': 'text', 'text': ''}
        elif block_type == 'tool_use':
            return {'type': 'tool_use', 'id': content_block.get('id', ''), 'name': content_block.get('name', ''), 'input': {}}
        else:
            # Default fallback
            return {'type': block_type, **{k: '' if isinstance(v, str) else v for k, v in content_block.items() if k != 'type'}}

    async def _generate_thinking_deltas(self, content_block: Dict[str, Any], index: int) -> AsyncIterator[bytes]:
        """Generate delta events for thinking content blocks."""
        thinking_content = content_block.get('thinking', '')
        signature = content_block.get('signature', '')

        # Send thinking content in chunks
        if thinking_content:
            chunk_size = 50
            for i in range(0, len(thinking_content), chunk_size):
                chunk_text = thinking_content[i : i + chunk_size]
                delta = {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'thinking_delta', 'thinking': chunk_text}}
                yield f'event: content_block_delta\ndata: {orjson.dumps(delta).decode()}\n\n'.encode()

        # Send signature as a single delta if present
        if signature:
            signature_delta = {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'signature_delta', 'signature': signature}}
            yield f'event: content_block_delta\ndata: {orjson.dumps(signature_delta).decode()}\n\n'.encode()

    async def _generate_text_deltas(self, content_block: Dict[str, Any], index: int) -> AsyncIterator[bytes]:
        """Generate delta events for text content blocks."""
        text_content = content_block.get('text', '')

        if text_content:
            # Split text into chunks for streaming effect
            chunk_size = 50
            for i in range(0, len(text_content), chunk_size):
                chunk_text = text_content[i : i + chunk_size]
                delta = {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'text_delta', 'text': chunk_text}}
                yield f'event: content_block_delta\ndata: {orjson.dumps(delta).decode()}\n\n'.encode()

    async def _generate_tool_use_deltas(self, content_block: Dict[str, Any], index: int) -> AsyncIterator[bytes]:
        """Generate delta events for tool_use content blocks."""
        tool_input = content_block.get('input', {})

        if tool_input:
            # Send the tool input as JSON delta
            input_json = orjson.dumps(tool_input).decode()

            # Send in chunks for streaming effect
            chunk_size = 100
            for i in range(0, len(input_json), chunk_size):
                chunk_text = input_json[i : i + chunk_size]
                delta = {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': chunk_text}}
                yield f'event: content_block_delta\ndata: {orjson.dumps(delta).decode()}\n\n'.encode()

    async def close(self):
        """Clean up resources."""
        await self.http_client.aclose()

    def __str__(self) -> str:
        return f'Provider(name={self.config.name}, url={self.config.url})'


class ProviderManager:
    """Manages multiple providers."""

    def __init__(self, providers_config: List[ProviderConfig], models_config: List[ModelConfig], transformer_loader: TransformerLoader):
        self.providers: Dict[str, Provider] = {}
        self.alias_mappings: Dict[str, ModelMapping] = {}
        self._load_providers(providers_config, transformer_loader)
        self._build_model_mapping(models_config)

    def _load_providers(self, providers_config: List[ProviderConfig], transformer_loader: TransformerLoader):
        """Load providers from configuration."""
        for provider_config in providers_config:
            try:
                provider = Provider(provider_config, transformer_loader)
                self.providers[provider_config.name] = provider

                logger.info(f"Loaded provider '{provider_config.name}'")

            except Exception as e:
                logger.error(f"Failed to load provider '{provider_config.name}': {e}", exc_info=True)

    def _build_model_mapping(self, models_config: List[ModelConfig]):
        """Build alias-to-provider mapping from top-level models configuration."""
        for model_config in models_config:
            if model_config.provider in self.providers:
                self.alias_mappings[model_config.alias] = ModelMapping(provider_name=model_config.provider, model_id=model_config.id)
            else:
                logger.warning(f"Model '{model_config.alias}' references unknown provider '{model_config.provider}'")

    def get_provider_for_model(self, alias: str) -> Optional[Tuple[Provider, str]]:
        """Get the provider that supports the given alias and return the resolved model ID."""
        mapping = self.alias_mappings.get(alias)
        if not mapping:
            return None

        provider = self.providers.get(mapping.provider_name)
        return (provider, mapping.model_id) if provider else None

    def get_provider_by_name(self, name: str) -> Optional[Provider]:
        """Get provider by name."""
        return self.providers.get(name)

    def list_providers(self) -> List[str]:
        """List all provider names."""
        return list(self.providers.keys())

    def list_models(self) -> List[str]:
        """List all supported model aliases."""
        return list(self.alias_mappings.keys())

    async def close_all(self):
        """Close all providers."""
        for provider in self.providers.values():
            await provider.close()
