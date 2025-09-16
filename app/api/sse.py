"""Server-sent event helpers used by streaming API endpoints."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict

import orjson

from app.config.log import get_logger
from app.observability.dumper import Dumper, DumpHandles
from app.routing.exchange import ExchangeResponse

logger = get_logger(__name__)


@dataclass
class SSEConfig:
    THINKING_CHUNK_SIZE: int = 50
    TEXT_CHUNK_SIZE: int = 50
    TOOL_INPUT_CHUNK_SIZE: int = 100
    DELTA_DELAY: float = 0.007


SSE_CONFIG = SSEConfig()


class DeltaStrategy:
    """Abstract base class describing how to emit deltas for a content block."""

    async def generate(self, content_block: Dict[str, Any], index: int, config: SSEConfig) -> AsyncIterator[bytes]:
        raise NotImplementedError


class ThinkingDeltaStrategy(DeltaStrategy):
    async def generate(self, content_block: Dict[str, Any], index: int, config: SSEConfig) -> AsyncIterator[bytes]:
        thinking_content = content_block.get('thinking', '')
        signature = content_block.get('signature', '')

        if thinking_content:
            chunk_size = config.THINKING_CHUNK_SIZE
            for start in range(0, len(thinking_content), chunk_size):
                chunk_text = thinking_content[start : start + chunk_size]
                delta = {
                    'type': 'content_block_delta',
                    'index': index,
                    'delta': {'type': 'thinking_delta', 'thinking': chunk_text},
                }
                yield f'event: content_block_delta\ndata: {orjson.dumps(delta).decode()}\n\n'.encode()

        if signature:
            delta = {
                'type': 'content_block_delta',
                'index': index,
                'delta': {'type': 'signature_delta', 'signature': signature},
            }
            yield f'event: content_block_delta\ndata: {orjson.dumps(delta).decode()}\n\n'.encode()


class TextDeltaStrategy(DeltaStrategy):
    async def generate(self, content_block: Dict[str, Any], index: int, config: SSEConfig) -> AsyncIterator[bytes]:
        text_content = content_block.get('text', '')
        if not text_content:
            return

        chunk_size = config.TEXT_CHUNK_SIZE
        for start in range(0, len(text_content), chunk_size):
            chunk_text = text_content[start : start + chunk_size]
            delta = {
                'type': 'content_block_delta',
                'index': index,
                'delta': {'type': 'text_delta', 'text': chunk_text},
            }
            yield f'event: content_block_delta\ndata: {orjson.dumps(delta).decode()}\n\n'.encode()


class ToolUseDeltaStrategy(DeltaStrategy):
    async def generate(self, content_block: Dict[str, Any], index: int, config: SSEConfig) -> AsyncIterator[bytes]:
        tool_input = content_block.get('input', {})
        if not tool_input:
            return

        input_json = orjson.dumps(tool_input).decode()
        chunk_size = config.TOOL_INPUT_CHUNK_SIZE
        for start in range(0, len(input_json), chunk_size):
            chunk_text = input_json[start : start + chunk_size]
            delta = {
                'type': 'content_block_delta',
                'index': index,
                'delta': {'type': 'input_json_delta', 'partial_json': chunk_text},
            }
            yield f'event: content_block_delta\ndata: {orjson.dumps(delta).decode()}\n\n'.encode()


class DeltaGenerator:
    """Selects the appropriate delta strategy per content block."""

    def __init__(self, config: SSEConfig = SSE_CONFIG):
        self.config = config
        self._strategies = {
            'thinking': ThinkingDeltaStrategy(),
            'text': TextDeltaStrategy(),
            'tool_use': ToolUseDeltaStrategy(),
        }

    async def generate_deltas(self, content_block: Dict[str, Any], index: int, block_type: str) -> AsyncIterator[bytes]:
        strategy = self._strategies.get(block_type, TextDeltaStrategy())
        async for delta in strategy.generate(content_block, index, self.config):
            yield delta


class SSEEventGenerator:
    def __init__(self, config: SSEConfig = SSE_CONFIG):
        self.config = config
        self.delta_generator = DeltaGenerator(config)

    async def convert_exchange_response(
        self,
        exchange_response: ExchangeResponse,
        dumper: Dumper,
        dumper_handles: DumpHandles,
    ) -> AsyncIterator[bytes]:
        response = exchange_response.payload or {}

        async for event in self._generate_message_start(response, dumper, dumper_handles):
            yield event

        content = response.get('content', [])
        for index, content_block in enumerate(content):
            block_type = content_block.get('type', 'text')
            async for event in self._process_content_block(content_block, block_type, index, dumper, dumper_handles):
                yield event

        async for event in self._generate_message_delta(response, dumper, dumper_handles):
            yield event

        async for event in self._generate_message_stop(dumper, dumper_handles):
            yield event

        dumper.close(dumper_handles)

    async def _generate_message_start(self, response: Dict[str, Any], dumper: Dumper, handles: DumpHandles) -> AsyncIterator[bytes]:
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
        dumper.write_response_chunk(handles, chunk)
        yield chunk

    async def _process_content_block(
        self,
        content_block: Dict[str, Any],
        block_type: str,
        index: int,
        dumper: Dumper,
        handles: DumpHandles,
    ) -> AsyncIterator[bytes]:
        content_block_start = {
            'type': 'content_block_start',
            'index': index,
            'content_block': _create_initial_content_block(content_block, block_type),
        }
        chunk = f'event: content_block_start\ndata: {orjson.dumps(content_block_start).decode()}\n\n'.encode()
        dumper.write_response_chunk(handles, chunk)
        yield chunk

        async for delta_event in self.delta_generator.generate_deltas(content_block, index, block_type):
            dumper.write_response_chunk(handles, delta_event)
            yield delta_event

        content_block_stop = {'type': 'content_block_stop', 'index': index}
        chunk = f'event: content_block_stop\ndata: {orjson.dumps(content_block_stop).decode()}\n\n'.encode()
        dumper.write_response_chunk(handles, chunk)
        yield chunk
        await asyncio.sleep(self.config.DELTA_DELAY)

    async def _generate_message_delta(self, response: Dict[str, Any], dumper: Dumper, handles: DumpHandles) -> AsyncIterator[bytes]:
        message_delta = {
            'type': 'message_delta',
            'delta': {
                'stop_reason': response.get('stop_reason', 'end_turn'),
                'stop_sequence': response.get('stop_sequence'),
            },
            'usage': response.get('usage', {'input_tokens': 0, 'output_tokens': 0}),
        }
        chunk = f'event: message_delta\ndata: {orjson.dumps(message_delta).decode()}\n\n'.encode()
        dumper.write_response_chunk(handles, chunk)
        yield chunk

    async def _generate_message_stop(self, dumper: Dumper, handles: DumpHandles) -> AsyncIterator[bytes]:
        message_stop = {'type': 'message_stop'}
        chunk = f'event: message_stop\ndata: {orjson.dumps(message_stop).decode()}\n\n'.encode()
        dumper.write_response_chunk(handles, chunk)
        logger.info('Finished processing request')
        yield chunk


async def convert_exchange_to_sse(exchange_response: ExchangeResponse, dumper: Dumper, handles: DumpHandles) -> AsyncIterator[bytes]:
    generator = SSEEventGenerator()
    async for event in generator.convert_exchange_response(exchange_response, dumper, handles):
        yield event


def _create_initial_content_block(content_block: Dict[str, Any], block_type: str) -> Dict[str, Any]:
    if block_type == 'thinking':
        return {'type': 'thinking', 'thinking': '', 'signature': ''}
    if block_type == 'text':
        return {'type': 'text', 'text': ''}
    if block_type == 'tool_use':
        return {'type': 'tool_use', 'id': content_block.get('id', ''), 'name': content_block.get('name', ''), 'input': {}}
    return {'type': block_type, **{key: ('' if isinstance(value, str) else value) for key, value in content_block.items() if key != 'type'}}


__all__ = ['SSE_CONFIG', 'SSEEventGenerator', 'convert_exchange_to_sse']
