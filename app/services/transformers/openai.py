"""OpenAI transformers with real format conversion."""

import copy
from typing import Any, Dict, Tuple

import orjson

from app.config.log import get_logger
from app.services.transformers.interfaces import RequestTransformer, ResponseTransformer

logger = get_logger(__name__)


class OpenAIRequestTransformer(RequestTransformer):
    """Transformer to convert Claude format to OpenAI format."""

    def __init__(self, logger):
        """Initialize transformer.

        API credentials are obtained from provider config during transform.
        """
        self.logger = logger

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Convert Claude request format to OpenAI format."""
        request = params['request']
        headers = params['headers']

        openai_request = {
            'model': request.get('model'),
            'temperature': request.get('temperature'),
            'stream': request.get('stream'),
            'tools': self._convert_tools(request.get('tools')),
            'messages': self._convert_messages(request),
        }

        # Map max_tokens to max_completion_tokens for OpenAI
        if 'max_tokens' in request:
            openai_request['max_completion_tokens'] = request['max_tokens']

        # Map thinking.budget_tokens to reasoning_effort for OpenAI
        thinking = request.get('thinking')
        if thinking and isinstance(thinking, dict):
            budget_tokens = thinking.get('budget_tokens', 0)
            if budget_tokens > 0:
                if budget_tokens < 1024:
                    openai_request['reasoning_effort'] = 'low'
                elif budget_tokens < 8192:
                    openai_request['reasoning_effort'] = 'medium'
                else:
                    openai_request['reasoning_effort'] = 'high'

        # Add stream_options if stream is True
        if request.get('stream') is True:
            openai_request['stream_options'] = {'include_usage': True}

        return openai_request, headers

    def _convert_tools(self, claude_tools):
        """Convert Claude tools format to OpenAI functions format."""
        if not claude_tools:
            return None

        openai_tools = []
        for tool in claude_tools:
            openai_tool = {'type': 'function', 'function': {'name': tool.get('name', ''), 'description': tool.get('description', ''), 'parameters': tool.get('input_schema', {})}}
            openai_tools.append(openai_tool)

        return openai_tools

    def _convert_user_message(self, claude_message):
        """Convert Claude user message to OpenAI format.

        Only handles plain text content for now.

        Args:
            claude_message: Claude message dict with role 'user'

        Returns:
            OpenAI format message dict, or None if not convertible
        """
        if claude_message.get('role') != 'user':
            return None

        content = claude_message.get('content')
        if not content:
            return None

        # Handle string content (simple case)
        if isinstance(content, str):
            return {'role': 'user', 'content': [{'type': 'text', 'text': content}]}

        # Handle list content - extract text and image blocks
        if isinstance(content, list):
            converted_blocks = []
            for block in content:
                if isinstance(block, dict):
                    if block.get('type') == 'text':
                        converted_blocks.append({'type': 'text', 'text': block.get('text', '')})
                    elif block.get('type') == 'image':
                        image_block = self._convert_image_block(block)
                        if image_block:
                            converted_blocks.append(image_block)

            if converted_blocks:
                return {'role': 'user', 'content': converted_blocks}

        return None

    def _convert_image_block(self, claude_image_block):
        """Convert Claude image block to OpenAI format.

        Args:
            claude_image_block: Claude image block dict

        Returns:
            OpenAI format image block dict, or None if not convertible
        """
        if not isinstance(claude_image_block, dict) or claude_image_block.get('type') != 'image':
            return None

        source = claude_image_block.get('source')
        if not source or source.get('type') != 'base64':
            return None

        data = source.get('data', '')
        media_type = source.get('media_type', 'image/jpeg')

        # Convert to data URL format
        data_url = f'data:{media_type};base64,{data}'

        return {'type': 'image_url', 'image_url': {'url': data_url}}

    def _convert_tool_result_to_message(self, tool_result_block):
        """Convert Claude tool_result block to OpenAI tool message.

        Args:
            tool_result_block: Claude tool_result block dict

        Returns:
            OpenAI format tool message dict, or None if not convertible
        """
        if not isinstance(tool_result_block, dict) or tool_result_block.get('type') != 'tool_result':
            return None

        tool_use_id = tool_result_block.get('tool_use_id')
        content = tool_result_block.get('content', '')
        is_error = tool_result_block.get('is_error', False)

        if not tool_use_id:
            return None

        # Apply empty content rules (content is always string)
        if not content:  # Empty string
            content = 'Error' if is_error else 'Success'
        # else: use content as-is (already a string)

        return {
            'role': 'tool',
            'tool_call_id': tool_use_id,  # Keep original ID format
            'content': content,
        }

    def _convert_content_block(self, block):
        """Convert individual content block (text or image) for user messages.

        Args:
            block: Individual content block dict

        Returns:
            Converted content block for OpenAI format, or None if not convertible
        """
        if not isinstance(block, dict):
            return None

        block_type = block.get('type')

        if block_type == 'text':
            return {'type': 'text', 'text': block.get('text', '')}
        elif block_type == 'image':
            return self._convert_image_block(block)

        return None

    def _convert_system_messages(self, claude_system):
        """Convert Claude system messages to single OpenAI system message.

        Args:
            claude_system: List of system message blocks from Claude format

        Returns:
            Single OpenAI system message dict with combined content, or None if no system messages
        """
        if not claude_system:
            return None

        # Combine all system message texts with newlines
        text_blocks = []
        for system_block in claude_system:
            if system_block.get('type') == 'text' and 'text' in system_block:
                text_blocks.append(system_block['text'])

        combined_text = '\n'.join(text_blocks)

        if not combined_text:
            return None

        return {'role': 'system', 'content': combined_text}

    def _convert_tool_use_to_tool_call(self, tool_use_block):
        """Convert Claude tool_use block to OpenAI tool_call format.

        Args:
            tool_use_block: Claude tool_use block dict

        Returns:
            OpenAI format tool_call dict, or None if not convertible
        """
        if not isinstance(tool_use_block, dict) or tool_use_block.get('type') != 'tool_use':
            return None

        tool_use_id = tool_use_block.get('id')
        name = tool_use_block.get('name')
        input_data = tool_use_block.get('input', {})

        if not tool_use_id or not name:
            return None

        return {'id': tool_use_id, 'type': 'function', 'function': {'name': name, 'arguments': orjson.dumps(input_data).decode('utf-8')}}

    def _convert_messages(self, claude_request):
        """Convert Claude request to OpenAI messages format.

        Uses ultra-granular queue-based architecture processing one content block per loop iteration.
        Maintains strict sequence by treating tool_results as message boundaries.

        Args:
            claude_request: Original Claude request dict

        Returns:
            List of OpenAI format messages
        """
        # Deep copy to avoid mutating original request
        claude_messages = copy.deepcopy(claude_request.get('messages', []))
        openai_messages = []

        # Add combined system message first
        system_message = self._convert_system_messages(claude_request.get('system'))
        if system_message:
            openai_messages.append(system_message)

        # Build content block queue - flatten all content blocks from all messages
        content_block_queue = []
        for message in claude_messages:
            if message.get('role') in ['user', 'assistant']:
                content = message.get('content', [])
                if isinstance(content, str):
                    content = [{'type': 'text', 'text': content}]

                # Add each content block to queue with original role context
                for block in content:
                    content_block_queue.append({'block': block, 'original_role': message.get('role')})

                # Add message boundary marker after each message
                content_block_queue.append({'type': 'message_boundary', 'original_role': message.get('role')})

        # Process queue one content block at a time
        current_user_content = []
        current_assistant_content = []
        current_assistant_tool_calls = []

        while content_block_queue:
            queue_item = content_block_queue.pop(0)  # Dequeue one content block

            # Handle message boundary marker
            if queue_item.get('type') == 'message_boundary':
                role = queue_item.get('original_role')

                # Flush accumulated content for this role at message boundary
                if role == 'user' and current_user_content:
                    openai_messages.append({'role': 'user', 'content': current_user_content})
                    current_user_content = []
                elif role == 'assistant':
                    # Assistant messages: handle content and tool_calls exclusivity
                    if current_assistant_content and current_assistant_tool_calls:
                        # Both content and tool_calls exist - create two messages
                        openai_messages.append({'role': 'assistant', 'content': current_assistant_content})
                        openai_messages.append({'role': 'assistant', 'tool_calls': current_assistant_tool_calls, 'content': None})
                    elif current_assistant_content:
                        # Only content
                        openai_messages.append({'role': 'assistant', 'content': current_assistant_content})
                    elif current_assistant_tool_calls:
                        # Only tool_calls
                        openai_messages.append({'role': 'assistant', 'tool_calls': current_assistant_tool_calls, 'content': None})

                    current_assistant_content = []
                    current_assistant_tool_calls = []
                continue

            block = queue_item['block']
            block_type = block.get('type')
            original_role = queue_item.get('original_role')

            if block_type == 'tool_result':
                # Flush accumulated user content (tool_results only come from user messages)
                if current_user_content:
                    openai_messages.append({'role': 'user', 'content': current_user_content})
                    current_user_content = []

                # Convert tool result to separate message
                tool_message = self._convert_tool_result_to_message(block)
                if tool_message:
                    openai_messages.append(tool_message)

            elif block_type in ['text', 'image']:
                # Accumulate content based on original message role
                converted_block = self._convert_content_block(block)
                if converted_block:
                    if original_role == 'user':
                        current_user_content.append(converted_block)
                    elif original_role == 'assistant':
                        current_assistant_content.append(converted_block)

            elif block_type == 'thinking':
                # Skip thinking blocks for OpenAI (not supported)
                pass

            elif block_type == 'tool_use':
                # Convert tool_use block to OpenAI tool_call format
                if original_role == 'assistant':
                    # Only assistant messages can have tool_calls in OpenAI
                    tool_call = self._convert_tool_use_to_tool_call(block)
                    if tool_call:
                        current_assistant_tool_calls.append(tool_call)

        # Final flush of remaining content
        if current_user_content:
            openai_messages.append({'role': 'user', 'content': current_user_content})

        # Final flush for assistant messages with content/tool_calls exclusivity
        if current_assistant_content and current_assistant_tool_calls:
            # Both content and tool_calls exist - create two messages
            openai_messages.append({'role': 'assistant', 'content': current_assistant_content})
            openai_messages.append({'role': 'assistant', 'tool_calls': current_assistant_tool_calls, 'content': None})
        elif current_assistant_content:
            # Only content
            openai_messages.append({'role': 'assistant', 'content': current_assistant_content})
        elif current_assistant_tool_calls:
            # Only tool_calls
            openai_messages.append({'role': 'assistant', 'tool_calls': current_assistant_tool_calls, 'content': None})

        return openai_messages


class OpenAIResponseTransformer(ResponseTransformer):
    """Transformer to convert OpenAI responses to Claude format."""

    async def transform_chunk(self, params: Dict[str, Any]) -> bytes:
        """Convert OpenAI streaming chunk to Claude format."""
        chunk = params['chunk']

        try:
            chunk_str = chunk.decode('utf-8')

            # Handle OpenAI SSE format
            if chunk_str.startswith('data: '):
                data_part = chunk_str[6:].strip()

                if data_part == '[DONE]':
                    # Convert OpenAI completion to Claude format
                    claude_done = {'type': 'message_stop'}
                    return f'data: {orjson.dumps(claude_done).decode("utf-8")}\n\n'.encode('utf-8')

                # Parse OpenAI chunk
                try:
                    openai_chunk = orjson.loads(data_part)
                    claude_chunk = self._convert_openai_chunk_to_claude(openai_chunk)
                    return f'data: {orjson.dumps(claude_chunk).decode("utf-8")}\n\n'.encode('utf-8')
                except orjson.JSONDecodeError:
                    logger.warning(f'Failed to parse OpenAI chunk JSON: {data_part[:100]}')
                    return chunk

            return chunk

        except Exception as e:
            logger.error(f'Failed to convert OpenAI chunk: {e}')
            return chunk

    async def transform_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI non-streaming response to Claude format."""
        response = params['response']

        try:
            # Convert OpenAI response structure to Claude format
            choices = response.get('choices', [])
            if not choices:
                logger.warning('OpenAI response has no choices')
                return response

            choice = choices[0]
            message = choice.get('message', {})
            content = message.get('content', '')

            # Handle tool calls in response
            tool_calls = message.get('tool_calls', [])
            claude_content = []

            # Add text content if present
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

            # Convert to Claude response format
            claude_response = {
                'id': response.get('id', ''),
                'type': 'message',
                'role': 'assistant',
                'content': claude_content,
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

        # Handle text content
        if 'content' in delta and delta['content']:
            return {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['content']}}

        # Handle tool calls (streaming)
        if 'tool_calls' in delta and delta['tool_calls']:
            tool_call = delta['tool_calls'][0]
            function = tool_call.get('function', {})

            if function.get('name'):
                # Start of tool call
                return {
                    'type': 'content_block_start',
                    'index': 0,
                    'content_block': {'type': 'tool_use', 'id': tool_call.get('id', ''), 'name': function.get('name', ''), 'input': {}},
                }
            elif function.get('arguments'):
                # Tool call arguments chunk
                return {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'input_json_delta', 'partial_json': function.get('arguments', '')}}

        # Handle finish reason
        if choice.get('finish_reason'):
            return {'type': 'message_stop'}

        return {'type': 'ping'}

    def _convert_stop_reason(self, openai_finish_reason: str) -> str:
        """Convert OpenAI finish reason to Claude format."""
        mapping = {'stop': 'end_turn', 'length': 'max_tokens', 'content_filter': 'stop_sequence', 'tool_calls': 'tool_use'}
        return mapping.get(openai_finish_reason, 'end_turn')
