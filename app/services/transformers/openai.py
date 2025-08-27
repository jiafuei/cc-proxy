"""OpenAI transformers with real format conversion."""

import json
from typing import Any, Dict, List, Tuple

from app.config.log import get_logger
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

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Convert Claude request format to OpenAI format."""
        request = params['request']
        config = params['provider_config']

        # Convert system messages and combine with regular messages
        openai_messages = self._convert_messages_with_system(request.get('messages', []), request.get('system'))

        # Convert model names
        openai_model = self._convert_model_name(request.get('model', ''))

        # Convert tools and tool_choice
        openai_tools = self._convert_tools(request.get('tools'))
        openai_tool_choice = self._convert_tool_choice(request.get('tool_choice'))

        # Convert stop sequences
        openai_stop = self._convert_stop_sequences(request.get('stop_sequences'))

        # Build OpenAI request (remove Claude-specific fields)
        openai_request = {
            'model': openai_model,
            'messages': openai_messages,
            'max_tokens': request.get('max_tokens'),
            'temperature': request.get('temperature'),
            'top_p': request.get('top_p'),
            'stream': request.get('stream', False),
            'tools': openai_tools,
            'tool_choice': openai_tool_choice,
            'stop': openai_stop,
        }

        # Remove None values
        openai_request = {k: v for k, v in openai_request.items() if v is not None}

        # Add OpenAI authentication headers
        auth_headers = {'authorization': f'Bearer {self.api_key or config.api_key}', 'content-type': 'application/json'}

        logger.debug(f'Converted Claude request to OpenAI format: {openai_model}')

        return openai_request, auth_headers

    def _convert_messages_with_system(self, claude_messages: List[Dict[str, Any]], system: Any) -> List[Dict[str, Any]]:
        """Convert Claude messages with system prompt to OpenAI format."""
        openai_messages = []

        # Convert system prompt first
        if system:
            system_content = self._convert_system_to_content(system)
            if system_content:
                openai_messages.append({'role': 'system', 'content': system_content})

        # Convert regular messages with proper tool handling
        i = 0
        while i < len(claude_messages):
            message = claude_messages[i]
            role = message.get('role', 'user')
            content = message.get('content', '')

            if role == 'user':
                # Handle user messages (may contain images or text)
                converted_msg = self._convert_user_message(message)
                if converted_msg:
                    openai_messages.append(converted_msg)
                i += 1
            elif role == 'assistant':
                # Handle assistant messages (may contain tool_use blocks)
                assistant_msg, tool_messages, next_i = self._convert_assistant_with_tools(claude_messages, i)
                if assistant_msg:
                    openai_messages.append(assistant_msg)
                # Add any tool result messages
                openai_messages.extend(tool_messages)
                i = next_i
            else:
                # Handle other message types normally
                content_str = self._convert_content_blocks(content)
                if content_str:
                    openai_messages.append({'role': role, 'content': content_str})
                i += 1

        return openai_messages

    def _convert_system_to_content(self, system: Any) -> str:
        """Convert Anthropic system format to OpenAI system message content."""
        if isinstance(system, str):
            return system
        elif isinstance(system, list):
            # System is array of text blocks
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
                elif isinstance(block, str):
                    text_parts.append(block)
            return '\n'.join(text_parts)
        else:
            return str(system) if system else ''

    def _convert_user_message(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
        """Convert Claude user message to OpenAI format (supports multimodal)."""
        content = message.get('content', '')
        name = message.get('name')

        if isinstance(content, str):
            # Simple text message
            result = {'role': 'user', 'content': content}
        elif isinstance(content, list):
            # Multimodal message with text and/or images
            openai_content = []
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get('type')
                    if block_type == 'text':
                        openai_content.append({'type': 'text', 'text': block.get('text', '')})
                    elif block_type == 'image':
                        # Convert Anthropic image to OpenAI format
                        image_source = block.get('source', {})
                        if image_source.get('type') == 'base64':
                            media_type = image_source.get('media_type', 'image/jpeg')
                            data = image_source.get('data', '')
                            openai_content.append({'type': 'image_url', 'image_url': {'url': f'data:{media_type};base64,{data}'}})
                elif isinstance(block, str):
                    openai_content.append({'type': 'text', 'text': block})

            if openai_content:
                result = {'role': 'user', 'content': openai_content}
            else:
                return None
        else:
            # Fallback to string conversion
            content_str = str(content) if content else ''
            if not content_str:
                return None
            result = {'role': 'user', 'content': content_str}

        # Add name if provided
        if name:
            result['name'] = name

        return result

    def _convert_assistant_with_tools(self, messages: List[Dict[str, Any]], start_idx: int) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]], int]:
        """Convert assistant message and handle tool use/result pattern.

        Returns: (assistant_message, tool_result_messages, next_message_index)
        """
        message = messages[start_idx]
        content = message.get('content', '')
        name = message.get('name')

        # Separate text content and tool calls
        text_parts = []
        tool_calls = []
        tool_call_counter = 0

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get('type')
                    if block_type == 'text':
                        text_parts.append(block.get('text', ''))
                    elif block_type == 'tool_use':
                        # Convert to OpenAI tool call format
                        tool_call_counter += 1
                        tool_call = {
                            'id': f'call_{tool_call_counter}_{block.get("id", "")}',
                            'type': 'function',
                            'function': {'name': block.get('name', ''), 'arguments': json.dumps(block.get('input', {}))},
                        }
                        tool_calls.append(tool_call)
                elif isinstance(block, str):
                    text_parts.append(block)
        elif isinstance(content, str):
            text_parts.append(content)

        # Build assistant message
        assistant_msg = {'role': 'assistant'}

        text_content = '\n'.join(text_parts).strip()
        if text_content:
            assistant_msg['content'] = text_content
        elif not tool_calls:
            # If no content and no tool calls, this is an empty message
            return None, [], start_idx + 1
        else:
            # Tool calls only, content can be null
            assistant_msg['content'] = None

        if tool_calls:
            assistant_msg['tool_calls'] = tool_calls

        if name:
            assistant_msg['name'] = name

        # Look for tool results in subsequent messages
        tool_result_messages = []
        next_idx = start_idx + 1

        # Check if next messages contain tool results
        while next_idx < len(messages):
            next_msg = messages[next_idx]
            if next_msg.get('role') == 'user':
                # Check if user message contains tool_result blocks
                user_content = next_msg.get('content', '')
                tool_results, remaining_content = self._extract_tool_results(user_content)

                if tool_results:
                    # Add tool result messages
                    tool_result_messages.extend(tool_results)

                    # If there's remaining user content, we'll process it in the next iteration
                    if remaining_content and str(remaining_content).strip():
                        # Create a new user message with remaining content
                        messages[next_idx] = {**next_msg, 'content': remaining_content}
                        break
                    else:
                        # This message was only tool results, continue to next
                        next_idx += 1
                        continue
                else:
                    # No tool results, this is a regular user message
                    break
            else:
                # Different role, stop looking for tool results
                break

        return assistant_msg, tool_result_messages, next_idx

    def _extract_tool_results(self, content: Any) -> Tuple[List[Dict[str, Any]], Any]:
        """Extract tool_result blocks and return them as OpenAI tool messages.

        Returns: (tool_result_messages, remaining_content)
        """
        tool_results = []
        remaining_blocks = []

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'tool_result':
                    # Convert to OpenAI tool message
                    tool_use_id = block.get('tool_use_id', '')
                    result_content = block.get('content', '')

                    # Convert result content to string if it's complex
                    if isinstance(result_content, list):
                        text_parts = []
                        for result_block in result_content:
                            if isinstance(result_block, dict) and result_block.get('type') == 'text':
                                text_parts.append(result_block.get('text', ''))
                        result_content = '\n'.join(text_parts)

                    tool_result = {
                        'role': 'tool',
                        'content': str(result_content),
                        'tool_call_id': f'call_1_{tool_use_id}',  # Match the format used in tool_calls
                    }
                    tool_results.append(tool_result)
                else:
                    remaining_blocks.append(block)

            return tool_results, remaining_blocks if remaining_blocks else ''
        else:
            # Not a list, no tool results
            return [], content

    def _convert_content_blocks(self, content: Any) -> str:
        """Convert Claude content blocks to simple string format (fallback for non-tool content)."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Claude format: content is list of blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get('type')
                    if block_type == 'text':
                        text_parts.append(block.get('text', ''))
                    elif block_type == 'tool_use':
                        # Convert tool use to a readable format for OpenAI
                        tool_name = block.get('name', '')
                        tool_input = block.get('input', {})
                        text_parts.append(f'[Tool: {tool_name} with input: {json.dumps(tool_input)}]')
                    elif block_type == 'tool_result':
                        # Convert tool result to readable format
                        tool_id = block.get('tool_use_id', '')
                        result_content = block.get('content', '')
                        text_parts.append(f'[Tool Result for {tool_id}: {result_content}]')
                    # Skip other block types like 'thinking'
                elif isinstance(block, str):
                    text_parts.append(block)
            return '\n'.join(text_parts)
        else:
            return str(content) if content else ''

    def _convert_tools(self, claude_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]] | None:
        """Convert Claude tools format to OpenAI functions format."""
        if not claude_tools:
            return None

        openai_tools = []
        for tool in claude_tools:
            # Convert Anthropic tool to OpenAI function
            function_def = {'type': 'function', 'function': {'name': tool.get('name', ''), 'description': tool.get('description', ''), 'parameters': tool.get('input_schema', {})}}
            openai_tools.append(function_def)

        return openai_tools if openai_tools else None

    def _convert_tool_choice(self, claude_tool_choice: Any) -> str | Dict[str, Any] | None:
        """Convert Claude tool_choice to OpenAI format."""
        if not claude_tool_choice:
            return None

        if isinstance(claude_tool_choice, str):
            # Simple string values
            if claude_tool_choice == 'auto':
                return 'auto'
            elif claude_tool_choice == 'any':
                return 'required'
            elif claude_tool_choice == 'none':
                return 'none'
        elif isinstance(claude_tool_choice, dict):
            # Specific tool choice
            tool_type = claude_tool_choice.get('type')
            if tool_type == 'tool':
                tool_name = claude_tool_choice.get('name')
                if tool_name:
                    return {'type': 'function', 'function': {'name': tool_name}}
            elif tool_type == 'any':
                return 'required'

        return 'auto'  # Default fallback

    def _convert_stop_sequences(self, claude_stop_sequences: List[str]) -> List[str] | None:
        """Convert Claude stop_sequences to OpenAI stop format."""
        if not claude_stop_sequences or not isinstance(claude_stop_sequences, list):
            return None

        # OpenAI supports up to 4 stop sequences
        return claude_stop_sequences[:4] if claude_stop_sequences else None

    def _convert_model_name(self, claude_model: str) -> str:
        """Use incoming model name as-is since it will already be a GPT model."""
        return claude_model


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
                    arguments = json.loads(function.get('arguments', '{}'))
                except json.JSONDecodeError:
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
