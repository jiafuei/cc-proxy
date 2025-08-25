"""OpenAI transformers with real format conversion."""

import json
from typing import Any, Dict, List, Mapping, Tuple

from fastapi import Request

from app.config.log import get_logger
from app.config.user_models import ProviderConfig
from app.services.transformers.interfaces import RequestTransformer, ResponseTransformer

logger = get_logger(__name__)


class OpenAIRequestTransformer(RequestTransformer):
    """Transformer to convert Claude format to OpenAI format."""

    def __init__(self, api_key: str = '', base_url: str = 'https://api.openai.com/v1/chat/completions'):
        """Initialize with OpenAI credentials.

        Args:
            api_key: OpenAI API key
            base_url: OpenAI API endpoint URL
        """
        self.api_key = api_key
        self.base_url = base_url

    async def transform(self, request: Dict[str, Any], headers: Mapping[str, Any], config: ProviderConfig, original_request: Request) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Convert Claude request format to OpenAI format."""

        # Convert messages format
        openai_messages = self._convert_messages(request.get('messages', []))

        # Convert model names
        openai_model = self._convert_model_name(request.get('model', ''))

        # Build OpenAI request (remove Claude-specific fields)
        openai_request = {
            'model': openai_model,
            'messages': openai_messages,
            'max_tokens': request.get('max_tokens'),
            'temperature': request.get('temperature'),
            'stream': request.get('stream', False),
        }

        # Remove None values
        openai_request = {k: v for k, v in openai_request.items() if v is not None}

        # Add OpenAI authentication headers
        auth_headers = {'authorization': f'Bearer {self.api_key or config.api_key}', 'content-type': 'application/json'}

        logger.debug(f'Converted Claude request to OpenAI format: {openai_model}')

        return openai_request, auth_headers

    def _convert_messages(self, claude_messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert Claude message format to OpenAI format."""
        openai_messages = []

        for message in claude_messages:
            role = message.get('role', 'user')
            content = message.get('content', '')

            # Convert Claude content blocks to simple string
            if isinstance(content, list):
                # Claude format: content is list of blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text_parts.append(block.get('text', ''))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content_str = '\n'.join(text_parts)
            else:
                content_str = str(content)

            openai_messages.append({'role': role, 'content': content_str})

        return openai_messages

    def _convert_model_name(self, claude_model: str) -> str:
        """Convert Claude model names to OpenAI equivalents."""
        model_mapping = {
            'claude-3-sonnet-20240229': 'gpt-4',
            'claude-3-haiku-20240307': 'gpt-3.5-turbo',
            'claude-3-opus-20240229': 'gpt-4-turbo',
            'claude-3-5-sonnet-20241022': 'gpt-4',
            'claude-3-5-haiku-20241022': 'gpt-3.5-turbo',
        }

        mapped_model = model_mapping.get(claude_model, 'gpt-4')
        if mapped_model != claude_model:
            logger.debug(f"Mapped Claude model '{claude_model}' to OpenAI model '{mapped_model}'")

        return mapped_model


class OpenAIResponseTransformer(ResponseTransformer):
    """Transformer to convert OpenAI responses to Claude format."""

    async def transform_chunk(self, chunk: bytes) -> bytes:
        """Convert OpenAI streaming chunk to Claude format."""
        try:
            chunk_str = chunk.decode('utf-8')

            # Handle OpenAI SSE format
            if chunk_str.startswith('data: '):
                data_part = chunk_str[6:].strip()

                if data_part == '[DONE]':
                    # Convert OpenAI completion to Claude format
                    claude_done = {'type': 'message_stop'}
                    return f'data: {json.dumps(claude_done)}\n\n'.encode('utf-8')

                # Parse OpenAI chunk
                try:
                    openai_chunk = json.loads(data_part)
                    claude_chunk = self._convert_openai_chunk_to_claude(openai_chunk)
                    return f'data: {json.dumps(claude_chunk)}\n\n'.encode('utf-8')
                except json.JSONDecodeError:
                    logger.warning(f'Failed to parse OpenAI chunk JSON: {data_part[:100]}')
                    return chunk

            return chunk

        except Exception as e:
            logger.error(f'Failed to convert OpenAI chunk: {e}')
            return chunk

    async def transform_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI non-streaming response to Claude format."""
        try:
            # Convert OpenAI response structure to Claude format
            choices = response.get('choices', [])
            if not choices:
                logger.warning('OpenAI response has no choices')
                return response

            choice = choices[0]
            message = choice.get('message', {})
            content = message.get('content', '')

            # Convert to Claude response format
            claude_response = {
                'id': response.get('id', ''),
                'type': 'message',
                'role': 'assistant',
                'content': [{'type': 'text', 'text': content}],
                'model': response.get('model', ''),
                'stop_reason': self._convert_stop_reason(choice.get('finish_reason')),
                'stop_sequence': None,
                'usage': {'input_tokens': response.get('usage', {}).get('prompt_tokens', 0), 'output_tokens': response.get('usage', {}).get('completion_tokens', 0)},
            }

            logger.debug('Converted OpenAI response to Claude format')
            return claude_response

        except Exception as e:
            logger.error(f'Failed to convert OpenAI response: {e}')
            return response

    def _convert_openai_chunk_to_claude(self, openai_chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI streaming chunk to Claude format."""
        choices = openai_chunk.get('choices', [])
        if not choices:
            return {'type': 'ping'}

        choice = choices[0]
        delta = choice.get('delta', {})

        if 'content' in delta and delta['content']:
            return {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['content']}}

        # Handle finish reason
        if choice.get('finish_reason'):
            return {'type': 'message_stop'}

        return {'type': 'ping'}

    def _convert_stop_reason(self, openai_finish_reason: str) -> str:
        """Convert OpenAI finish reason to Claude format."""
        mapping = {'stop': 'end_turn', 'length': 'max_tokens', 'content_filter': 'stop_sequence', 'tool_calls': 'tool_use'}
        return mapping.get(openai_finish_reason, 'end_turn')
