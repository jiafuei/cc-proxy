"""OpenAI transformers with real format conversion."""

from typing import Any, AsyncIterator, Dict, Tuple

import orjson

from app.config.log import get_logger
from app.services.transformers.interfaces import RequestTransformer, ResponseTransformer

logger = get_logger(__name__)


class OpenAIRequestTransformer(RequestTransformer):
    """Transformer to convert Claude format to OpenAI format."""

    # Reasoning effort threshold mapping
    REASONING_EFFORT_THRESHOLDS = [(1024, 'low'), (8192, 'medium'), (float('inf'), 'high')]

    def __init__(self, logger):
        """Initialize transformer.

        API credentials are obtained from provider config during transform.
        """
        self.logger = logger

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Convert Claude request format to OpenAI format."""
        request = params['request']
        headers = params['headers']

        # Build OpenAI request with filtered comprehension
        stream = request.get('stream')
        openai_request = {
            k: v
            for k, v in {
                'model': request.get('model'),
                'temperature': request.get('temperature'),
                'stream': stream,
                'tools': self._convert_tools(request.get('tools')),
                'messages': self._convert_messages(request),
                'max_completion_tokens': request.get('max_tokens'),
                'reasoning_effort': self._get_reasoning_effort(request),
                'stream_options': {'include_usage': True} if stream else None,
            }.items()
            if v is not None
        }
        filtered_headers = {k: v for k, v in headers.items() if any(k.startswith(prefix) for prefix in {'user-', 'accept'})}

        return openai_request, filtered_headers

    def _get_reasoning_effort(self, request):
        """Get reasoning effort from thinking.budget_tokens."""
        if not (thinking := request.get('thinking')):
            return None
        if not (budget_tokens := thinking.get('budget_tokens', 0)) > 0:
            return None
        return next(effort for threshold, effort in self.REASONING_EFFORT_THRESHOLDS if budget_tokens < threshold)

    def _convert_tools(self, claude_tools):
        """Convert Claude tools format to OpenAI functions format."""
        return claude_tools and [
            {'type': 'function', 'function': {'name': tool.get('name', ''), 'description': tool.get('description', ''), 'parameters': tool.get('input_schema', {})}}
            for tool in claude_tools
        ]

    def _convert_messages(self, claude_request):
        """Convert Claude request to OpenAI messages format."""
        messages = []

        # Add system message if present
        if system := claude_request.get('system'):
            system_content = '\n'.join(block['text'] for block in system if isinstance(block, dict) and block.get('type') == 'text' and 'text' in block)
            if system_content:
                messages.append({'role': 'system', 'content': system_content})

        # Process regular messages
        for message in claude_request.get('messages', []):
            messages.extend(self._process_message(message))

        return messages

    def _process_message(self, message):
        """Process a single Claude message into OpenAI format messages."""
        role, content = message.get('role'), message.get('content', [])
        if isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        elif not isinstance(content, list):
            return []

        messages, current_content = [], []

        def flush_content():
            if current_content:
                messages.append({'role': role, 'content': self._convert_content_blocks(current_content)})
                current_content.clear()

        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get('type')
            if block_type == 'tool_result':
                flush_content()
                messages.append(self._convert_tool_result(block))
            elif block_type == 'tool_use' and role == 'assistant':
                flush_content()
                messages.append({'role': 'assistant', 'content': None, 'tool_calls': [self._convert_tool_call(block)]})
            elif block_type in ['text', 'image']:
                current_content.append(block)

        flush_content()
        return messages

    def _convert_content_blocks(self, blocks):
        """Convert list of content blocks to OpenAI format."""
        converted = []
        for block in blocks:
            if block.get('type') == 'text':
                converted.append({'type': 'text', 'text': block.get('text', '')})
            elif block.get('type') == 'image' and (image_block := self._convert_image_block(block)):
                converted.append(image_block)
        return converted

    def _convert_image_block(self, block):
        """Convert Claude image block to OpenAI format."""
        if not (source := block.get('source')):
            return None
        if source.get('type') != 'base64':
            return None

        data = source.get('data', '')
        media_type = source.get('media_type', 'image/jpeg')

        return {'type': 'image_url', 'image_url': {'url': f'data:{media_type};base64,{data}'}}

    def _convert_tool_result(self, block):
        """Convert Claude tool_result to OpenAI tool message."""
        return {'role': 'tool', 'tool_call_id': block.get('tool_use_id'), 'content': block.get('content') or ('Error' if block.get('is_error') else 'Success')}

    def _convert_tool_call(self, block):
        """Convert Claude tool_use to OpenAI tool_call."""
        return {'id': block.get('id'), 'type': 'function', 'function': {'name': block.get('name'), 'arguments': orjson.dumps(block.get('input', {})).decode('utf-8')}}


class OpenAIResponseTransformer(ResponseTransformer):
    """Transformer to convert OpenAI responses to Claude format."""

    # Constants for magic values
    DATA_PREFIX = 'data: '
    DONE_TOKEN = '[DONE]'
    DEFAULT_INDEX = 0

    # Stop reason mapping as class constant
    STOP_REASON_MAPPING = {'stop': 'end_turn', 'length': 'max_tokens', 'content_filter': 'stop_sequence', 'tool_calls': 'tool_use'}

    def _format_sse_event(self, event_type: str, data: Dict[str, Any]) -> bytes:
        """Format data as SSE event with proper headers."""
        return f'event: {event_type}\ndata: {orjson.dumps(data).decode("utf-8")}\n\n'.encode('utf-8')

    async def transform_chunk(self, params: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Convert OpenAI streaming chunk to Claude format."""
        chunk = params['chunk']

        try:
            chunk_str = chunk.decode('utf-8')
            lines = chunk_str.split('\n')

            # Process each line that contains SSE data
            has_sse_data = False
            for line in lines:
                line = line.strip()
                if not line or not line.startswith(self.DATA_PREFIX):
                    continue

                has_sse_data = True
                data_part = line[len(self.DATA_PREFIX) :].strip()

                if data_part == self.DONE_TOKEN:
                    yield self._format_sse_event('message_stop', {'type': 'message_stop'})
                    return  # [DONE] is always at the end

                # Parse OpenAI chunk
                try:
                    openai_chunk = orjson.loads(data_part)
                    claude_chunk = self._convert_openai_chunk_to_claude(openai_chunk)
                    event_type = claude_chunk.get('type', 'ping')
                    yield self._format_sse_event(event_type, claude_chunk)
                except orjson.JSONDecodeError as e:
                    logger.warning(f'Failed to parse OpenAI chunk JSON(): {data_part}')
                    # Continue processing other lines instead of falling back to passthrough

            # If no SSE data was found, pass through the original chunk
            if not has_sse_data:
                yield chunk

        except UnicodeDecodeError:
            logger.error('Failed to decode chunk as UTF-8')
            yield chunk
        except Exception as e:
            logger.error(f'Failed to convert OpenAI chunk: {e}')
            yield chunk

    async def transform_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI non-streaming response to Claude format."""
        response = params['response']

        try:
            choices = response.get('choices', [])
            if not choices:
                logger.warning('OpenAI response has no choices')
                return response

            choice = choices[0]
            message = choice.get('message', {})
            content = message.get('content', '')
            tool_calls = message.get('tool_calls', [])

            # Build Claude content array
            claude_content = []
            if content:
                claude_content.append({'type': 'text', 'text': content})

            # Convert tool calls to Claude tool_use blocks
            for tool_call in tool_calls:
                function = tool_call.get('function', {})
                try:
                    arguments = orjson.loads(function.get('arguments', '{}'))
                except orjson.JSONDecodeError:
                    arguments = {}

                claude_content.append({'type': 'tool_use', 'id': tool_call.get('id', ''), 'name': function.get('name', ''), 'input': arguments})

            # Build usage info more efficiently
            usage = response.get('usage', {})
            claude_usage = {'input_tokens': usage.get('prompt_tokens', 0), 'output_tokens': usage.get('completion_tokens', 0)}

            # Convert to Claude response format
            claude_response = {
                'id': response.get('id', ''),
                'type': 'message',
                'role': 'assistant',
                'content': claude_content,
                'model': response.get('model', ''),
                'stop_reason': self._convert_stop_reason(choice.get('finish_reason')),
                'stop_sequence': None,
                'usage': claude_usage,
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

        # Handle finish reason first (early return)
        if choice.get('finish_reason'):
            return {'type': 'message_stop'}

        # Handle text content
        if content := delta.get('content'):
            return {'type': 'content_block_delta', 'index': self.DEFAULT_INDEX, 'delta': {'type': 'text_delta', 'text': content}}

        # Handle tool calls (streaming)
        if tool_calls := delta.get('tool_calls'):
            return self._convert_tool_call_delta(tool_calls[0])

        return {'type': 'ping'}

    def _convert_tool_call_delta(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI tool call delta to Claude format."""
        index = tool_call.get('index', self.DEFAULT_INDEX)
        function = tool_call.get('function', {})

        # Start of tool call - has name and id
        if function.get('name'):
            return {
                'type': 'content_block_start',
                'index': index,
                'content_block': {'type': 'tool_use', 'id': tool_call.get('id', ''), 'name': function['name'], 'input': {}},
            }

        # Tool call arguments chunk
        if arguments := function.get('arguments'):
            return {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': arguments}}

        return {'type': 'ping'}

    def _convert_stop_reason(self, openai_finish_reason: str) -> str:
        """Convert OpenAI finish reason to Claude format."""
        return self.STOP_REASON_MAPPING.get(openai_finish_reason, 'end_turn')
