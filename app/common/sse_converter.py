"""SSE conversion utilities for converting JSON responses to Server-Sent Events format."""

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict

import orjson

from app.common.dumper import Dumper, DumpHandles
from app.config.log import get_logger


@dataclass
class SSEConfig:
    THINKING_CHUNK_SIZE: int = 50
    TEXT_CHUNK_SIZE: int = 50
    TOOL_INPUT_CHUNK_SIZE: int = 100
    DELTA_DELAY: float = 0.007


SSE_CONFIG = SSEConfig()

logger = get_logger(__name__)


class DeltaStrategy:
    """Abstract base class for delta generation strategies."""

    async def generate(self, content_block: Dict[str, Any], index: int, config: SSEConfig) -> AsyncIterator[bytes]:
        raise NotImplementedError


class ThinkingDeltaStrategy(DeltaStrategy):
    """Delta strategy for thinking content blocks."""

    async def generate(self, content_block: Dict[str, Any], index: int, config: SSEConfig) -> AsyncIterator[bytes]:
        thinking_content = content_block.get('thinking', '')
        signature = content_block.get('signature', '')

        # Send thinking content in chunks
        if thinking_content:
            chunk_size = config.THINKING_CHUNK_SIZE
            for i in range(0, len(thinking_content), chunk_size):
                chunk_text = thinking_content[i : i + chunk_size]
                delta = {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'thinking_delta', 'thinking': chunk_text}}
                yield f'event: content_block_delta\ndata: {orjson.dumps(delta).decode()}\n\n'.encode()

        # Send signature as a single delta if present
        if signature:
            signature_delta = {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'signature_delta', 'signature': signature}}
            yield f'event: content_block_delta\ndata: {orjson.dumps(signature_delta).decode()}\n\n'.encode()


class TextDeltaStrategy(DeltaStrategy):
    """Delta strategy for text content blocks."""

    async def generate(self, content_block: Dict[str, Any], index: int, config: SSEConfig) -> AsyncIterator[bytes]:
        text_content = content_block.get('text', '')

        if text_content:
            # Split text into chunks for streaming effect
            chunk_size = config.TEXT_CHUNK_SIZE
            for i in range(0, len(text_content), chunk_size):
                chunk_text = text_content[i : i + chunk_size]
                delta = {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'text_delta', 'text': chunk_text}}
                yield f'event: content_block_delta\ndata: {orjson.dumps(delta).decode()}\n\n'.encode()


class ToolUseDeltaStrategy(DeltaStrategy):
    """Delta strategy for tool_use content blocks."""

    async def generate(self, content_block: Dict[str, Any], index: int, config: SSEConfig) -> AsyncIterator[bytes]:
        tool_input = content_block.get('input', {})

        if tool_input:
            # Send the tool input as JSON delta
            input_json = orjson.dumps(tool_input).decode()

            # Send in chunks for streaming effect
            chunk_size = config.TOOL_INPUT_CHUNK_SIZE
            for i in range(0, len(input_json), chunk_size):
                chunk_text = input_json[i : i + chunk_size]
                delta = {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': chunk_text}}
                yield f'event: content_block_delta\ndata: {orjson.dumps(delta).decode()}\n\n'.encode()


class DeltaGenerator:
    """Unified delta generator using strategy pattern."""

    def __init__(self, config: SSEConfig = SSE_CONFIG):
        self.config = config

    def _get_delta_strategy(self, block_type: str) -> DeltaStrategy:
        """Get the appropriate delta strategy for a content block type."""
        strategies = {
            'thinking': ThinkingDeltaStrategy(),
            'text': TextDeltaStrategy(),
            'tool_use': ToolUseDeltaStrategy(),
        }
        return strategies.get(block_type, TextDeltaStrategy())  # Default to text strategy

    async def generate_deltas(self, content_block: Dict[str, Any], index: int, block_type: str) -> AsyncIterator[bytes]:
        """Generate delta events for a content block using the appropriate strategy."""
        strategy = self._get_delta_strategy(block_type)
        async for delta in strategy.generate(content_block, index, self.config):
            yield delta


class SSEEventGenerator:
    def __init__(self, config: SSEConfig = SSE_CONFIG):
        self.config = config
        self.delta_generator = DeltaGenerator(config)

    async def convert_json_to_sse(self, response: Dict[str, Any], dumper: Dumper, dumper_handles: DumpHandles) -> AsyncIterator[bytes]:
        """Convert a JSON response to SSE format.

        Args:
            response: JSON response dictionary from LLM provider
            dumper: Dumper instance for logging
            dumper_handles: Dumper handles for the current request

        Yields:
            SSE-formatted response chunks
        """
        # Create message_start event
        async for event in self._generate_message_start(response, dumper, dumper_handles):
            yield event

        # Process each content block
        content = response.get('content', [])
        for index, content_block in enumerate(content):
            block_type = content_block.get('type', 'text')
            async for event in self._process_content_block(content_block, block_type, index, dumper, dumper_handles):
                yield event

        # Create message_delta event with final stop reason
        async for event in self._generate_message_delta(response, dumper, dumper_handles):
            yield event

        # Create message_stop event
        async for event in self._generate_message_stop(dumper, dumper_handles):
            yield event

    async def _generate_message_start(self, response: Dict[str, Any], dumper: Dumper, dumper_handles: DumpHandles) -> AsyncIterator[bytes]:
        """Generate message start SSE events."""
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
        chunk = f'event: message_start\ndata: {orjson.dumps(message_start).decode()}\n\n'.encode()
        dumper.write_response_chunk(dumper_handles, chunk)
        yield chunk

    async def _process_content_block(self, content_block: Dict[str, Any], block_type: str, index: int, dumper: Dumper, dumper_handles: DumpHandles) -> AsyncIterator[bytes]:
        """Process a single content block and generate appropriate SSE events."""
        # Create content_block_start event for this block
        content_block_start = {'type': 'content_block_start', 'index': index, 'content_block': _create_initial_content_block(content_block, block_type)}
        chunk = f'event: content_block_start\ndata: {orjson.dumps(content_block_start).decode()}\n\n'.encode()
        dumper.write_response_chunk(dumper_handles, chunk)
        yield chunk

        # Generate deltas using unified delta generator
        async for delta_event in self.delta_generator.generate_deltas(content_block, index, block_type):
            dumper.write_response_chunk(dumper_handles, delta_event)
            yield delta_event

        # Create content_block_stop event
        content_block_stop = {'type': 'content_block_stop', 'index': index}
        chunk = f'event: content_block_stop\ndata: {orjson.dumps(content_block_stop).decode()}\n\n'.encode()
        dumper.write_response_chunk(dumper_handles, chunk)
        yield chunk
        await asyncio.sleep(self.config.DELTA_DELAY)

    async def _generate_message_delta(self, response: Dict[str, Any], dumper: Dumper, dumper_handles: DumpHandles) -> AsyncIterator[bytes]:
        """Generate message delta SSE event."""
        message_delta = {
            'type': 'message_delta',
            'delta': {'stop_reason': response.get('stop_reason', 'end_turn'), 'stop_sequence': response.get('stop_sequence')},
            'usage': response.get('usage', {'input_tokens': 0, 'output_tokens': 0}),
        }
        chunk = f'event: message_delta\ndata: {orjson.dumps(message_delta).decode()}\n\n'.encode()
        dumper.write_response_chunk(dumper_handles, chunk)
        yield chunk

    async def _generate_message_stop(self, dumper: Dumper, dumper_handles: DumpHandles) -> AsyncIterator[bytes]:
        """Generate message stop SSE event."""
        message_stop = {'type': 'message_stop'}
        chunk = f'event: message_stop\ndata: {orjson.dumps(message_stop).decode()}\n\n'.encode()
        dumper.write_response_chunk(dumper_handles, chunk)
        logger.info('Finished processing request')
        yield chunk


async def convert_json_to_sse(response: Dict[str, Any], dumper: Dumper, dumper_handles: DumpHandles) -> AsyncIterator[bytes]:
    """Convert a JSON response to SSE format.

    Args:
        response: JSON response dictionary from LLM provider
        dumper: Dumper instance for logging
        dumper_handles: Dumper handles for the current request

    Yields:
        SSE-formatted response chunks
    """
    generator = SSEEventGenerator()
    async for event in generator.convert_json_to_sse(response, dumper, dumper_handles):
        yield event


def _create_initial_content_block(content_block: Dict[str, Any], block_type: str) -> Dict[str, Any]:
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


async def _generate_thinking_deltas(content_block: Dict[str, Any], index: int) -> AsyncIterator[bytes]:
    """Generate delta events for thinking content blocks."""
    thinking_content = content_block.get('thinking', '')
    signature = content_block.get('signature', '')

    # Send thinking content in chunks
    if thinking_content:
        chunk_size = SSE_CONFIG.THINKING_CHUNK_SIZE
        for i in range(0, len(thinking_content), chunk_size):
            chunk_text = thinking_content[i : i + chunk_size]
            delta = {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'thinking_delta', 'thinking': chunk_text}}
            yield f'event: content_block_delta\ndata: {orjson.dumps(delta).decode()}\n\n'.encode()

    # Send signature as a single delta if present
    if signature:
        signature_delta = {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'signature_delta', 'signature': signature}}
        yield f'event: content_block_delta\ndata: {orjson.dumps(signature_delta).decode()}\n\n'.encode()


async def _generate_text_deltas(content_block: Dict[str, Any], index: int) -> AsyncIterator[bytes]:
    """Generate delta events for text content blocks."""
    text_content = content_block.get('text', '')

    if text_content:
        # Split text into chunks for streaming effect
        chunk_size = SSE_CONFIG.TEXT_CHUNK_SIZE
        for i in range(0, len(text_content), chunk_size):
            chunk_text = text_content[i : i + chunk_size]
            delta = {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'text_delta', 'text': chunk_text}}
            yield f'event: content_block_delta\ndata: {orjson.dumps(delta).decode()}\n\n'.encode()


async def _generate_tool_use_deltas(content_block: Dict[str, Any], index: int) -> AsyncIterator[bytes]:
    """Generate delta events for tool_use content blocks."""
    tool_input = content_block.get('input', {})

    if tool_input:
        # Send the tool input as JSON delta
        input_json = orjson.dumps(tool_input).decode()

        # Send in chunks for streaming effect
        chunk_size = SSE_CONFIG.TOOL_INPUT_CHUNK_SIZE
        for i in range(0, len(input_json), chunk_size):
            chunk_text = input_json[i : i + chunk_size]
            delta = {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': chunk_text}}
            yield f'event: content_block_delta\ndata: {orjson.dumps(delta).decode()}\n\n'.encode()
