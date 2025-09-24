"""OpenAI transformers with real format conversion."""

import hashlib
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import orjson

from app.config.log import get_logger
from app.transformers.interfaces import ProviderRequestTransformer, ProviderResponseTransformer

logger = get_logger(__name__)


class ClaudeOpenAIRequestTransformer(ProviderRequestTransformer):
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

        tools = request.get('tools') or []
        builtin_tools = [tool for tool in tools if self._is_builtin_tool(tool)]
        callable_tools = [tool for tool in tools if not self._is_builtin_tool(tool)]

        builtin_tool = self._select_builtin_tool(builtin_tools, callable_tools)

        openai_request = self._build_openai_request(
            request,
            tools=None if builtin_tool else callable_tools,
        )

        if builtin_tool:
            self._apply_builtin_tool(openai_request, builtin_tool)

        if 'tools' in openai_request and not openai_request.get('tools'):
            openai_request.pop('tools', None)

        return openai_request, headers

    def _build_openai_request(self, request: Dict[str, Any], tools: Optional[list]) -> Dict[str, Any]:
        """Build the base OpenAI payload shared by all tool paths."""

        stream = request.get('stream')
        openai_request = {
            'model': request.get('model'),
            'temperature': request.get('temperature'),
            'stream': stream,
            'store': False,
            'messages': self._convert_messages(request),
            'max_completion_tokens': request.get('max_tokens'),
            'reasoning_effort': self._get_reasoning_effort(request),
        }

        if stream:
            openai_request['stream_options'] = {'include_usage': True}

        if tools is not None:
            converted_tools = self._convert_tools(tools)
            if converted_tools is not None:
                openai_request['tools'] = converted_tools

        return {k: v for k, v in openai_request.items() if v is not None}

    def _apply_builtin_tool(self, openai_request: Dict[str, Any], tool: dict) -> None:
        """Apply built-in tool specific adjustments to the OpenAI payload."""

        openai_request.pop('tools', None)

        if self._is_websearch_tool(tool):
            web_search_config = self._extract_websearch_config(tool)
            self.logger.debug(f'Extracted WebSearch config: {web_search_config}')
            openai_request['web_search_options'] = web_search_config
            openai_request['model'] = 'gpt-4o-search-preview'

    def _select_builtin_tool(self, builtin_tools: list, callable_tools: list) -> Optional[dict]:
        """Determine if the request should use the built-in tool pathway."""

        if not builtin_tools:
            return None

        if callable_tools:
            self.logger.warning('Ignoring built-in tool path because callable tools are present alongside built-ins')
            return None

        if len(builtin_tools) > 1:
            self.logger.warning(f'Expected single built-in tool but found {len(builtin_tools)}, using first tool only')

        return builtin_tools[0]

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
        for message in claude_request.get('messages') or []:
            messages.extend(self._process_message(message))

        return messages

    def _process_message(self, message):
        """Process a single Claude message into OpenAI format messages."""
        role, content = message.get('role'), message.get('content')
        if content is None:
            content = []
        if isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        elif not isinstance(content, list):
            return []

        messages, current_content, pending_tool_calls = [], [], []

        def flush_content():
            if current_content:
                messages.append({'role': role, 'content': self._convert_content_blocks(current_content)})
                current_content.clear()

        def flush_tool_calls():
            if pending_tool_calls:
                messages.append({'role': 'assistant', 'content': None, 'tool_calls': pending_tool_calls.copy()})
                pending_tool_calls.clear()

        def flush_combined():
            """Flush content and tool_calls, combining them if both exist."""
            if current_content and pending_tool_calls:
                # Combine both content and tool_calls into one message
                messages.append({'role': 'assistant', 'content': self._convert_content_blocks(current_content), 'tool_calls': pending_tool_calls.copy()})
                current_content.clear()
                pending_tool_calls.clear()
            elif current_content:
                flush_content()
            elif pending_tool_calls:
                flush_tool_calls()

        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get('type')
            if block_type == 'tool_result':
                flush_combined()
                messages.append(self._convert_tool_result(block))
            elif block_type == 'tool_use' and role == 'assistant':
                # Don't flush content - allow text + tool_use to combine
                pending_tool_calls.append(self._convert_tool_call(block))
            elif block_type in ['text', 'image']:
                # Don't flush tool_calls - allow text + tool_use to combine
                current_content.append(block)

        flush_combined()
        return messages

    def _convert_content_blocks(self, blocks):
        """Convert list of content blocks to OpenAI format."""
        converted = []
        for block in blocks:
            if block.get('type') == 'text':
                converted.append({'type': 'text', 'text': block.get('text', '')})
            elif block.get('type') == 'image' and (image_block := self._convert_image_block(block)):
                converted.append(image_block)
        if len(converted) == 1 and converted[0].get('type') == 'text':
            converted = converted[0].get('text')
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
        message = {
            'role': 'tool',
            'tool_call_id': block.get('tool_use_id'),
            'content': block.get('content'),
        }
        if isinstance(message['content'], list) and len(message['content']) == 1:
            item = message['content'][0]
            if isinstance(item, dict) and item.get('type') == 'text':
                message['content'] = item.get('text', '')
        if block.get('is_error'):
            message['content'] = f"Error: {message.get('content', '')}"

        return message

    def _convert_tool_call(self, block):
        """Convert Claude tool_use to OpenAI tool_call."""
        return {'id': block.get('id'), 'type': 'function', 'function': {'name': block.get('name'), 'arguments': orjson.dumps(block.get('input', {})).decode('utf-8')}}

    def _extract_websearch_config(self, tool: dict) -> dict:
        """Convert Anthropic WebSearch tool to OpenAI web_search_options format.

        Args:
            tool: Anthropic WebSearch tool dictionary

        Returns:
            OpenAI web_search_options configuration
        """
        config = {}

        # Validate domain filter constraints
        self._validate_domain_filters(tool)

        # Domain filtering
        allowed_domains = tool.get('allowed_domains')
        blocked_domains = tool.get('blocked_domains')

        if allowed_domains or blocked_domains:
            filters = {}
            if allowed_domains:
                filters['allowed_domains'] = allowed_domains
            if blocked_domains:
                filters['blocked_domains'] = blocked_domains
            config['filters'] = filters

        # User location
        if user_location := tool.get('user_location'):
            config['user_location'] = self._convert_user_location(user_location)

        # Search context size (default to medium)
        config['search_context_size'] = 'medium'

        return self._handle_missing_parameters(config)

    def _convert_user_location(self, user_location: dict) -> dict:
        """Convert Anthropic user location to OpenAI format.

        Args:
            user_location: Anthropic user location dictionary

        Returns:
            OpenAI user location format
        """
        openai_location = {'type': 'approximate', 'approximate': {}}

        # Map fields that exist
        approximate = openai_location['approximate']
        for field in ['country', 'city', 'region', 'timezone']:
            if value := user_location.get(field):
                approximate[field] = value

        return openai_location

    def _validate_domain_filters(self, tool: dict):
        """Validate domain filter constraints.

        Args:
            tool: WebSearch tool to validate

        Raises:
            ValueError: If both allowed_domains and blocked_domains are specified
        """
        if tool.get('allowed_domains') and tool.get('blocked_domains'):
            raise ValueError('Cannot use both allowed_domains and blocked_domains in WebSearch')

    def _handle_missing_parameters(self, config: dict) -> dict:
        """Apply defaults for missing parameters.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with defaults applied
        """
        # Ensure search_context_size is set
        config.setdefault('search_context_size', 'medium')

        # Ensure filters exists if not set
        config.setdefault('filters', {})

        return config


class ClaudeOpenAIResponseTransformer(ProviderResponseTransformer):
    """Transformer to convert OpenAI responses to Claude format."""

    # Stop reason mapping as class constant
    STOP_REASON_MAPPING = {'stop': 'end_turn', 'length': 'max_tokens', 'content_filter': 'stop_sequence', 'tool_calls': 'tool_use'}

    # OpenAI SSE constants
    DATA_PREFIX = b'data:'
    DONE_MARKER = b'[DONE]'

    def _init_state(self) -> Dict[str, Any]:
        """Initialize processing state for a new message stream."""
        return {
            'message_id': '',
            'model': '',
            'next_block_index': 0,
            'active_text_block': None,
            'active_tool_block': None,
            'usage_tokens': {},
            'stop_reason': None,
            'message_started': False,
        }

    def _format_anthropic_sse(self, event_type: str, data: Dict[str, Any]) -> bytes:
        """Format data as Anthropic SSE event."""
        event_line = f'event: {event_type}\n'
        data_line = f'data: {orjson.dumps(data).decode()}\n\n'
        return (event_line + data_line).encode()

    async def transform_chunk(self, params: dict[str, Any]) -> AsyncIterator[bytes]:
        """Transform OpenAI SSE chunk to Anthropic SSE format."""
        # Initialize state if not present
        if 'sse_state' not in params:
            params['sse_state'] = self._init_state()

        state = params['sse_state']
        chunk: bytes = params['chunk']

        # Process each line in the chunk
        lines = chunk.splitlines()
        for line in lines:
            if not line:
                continue

            if not line.startswith(self.DATA_PREFIX):
                logger.debug(f'Non-data SSE line: {line}')
                continue

            # Handle [DONE] marker
            data_content = line[len(self.DATA_PREFIX) :].strip()
            if data_content == self.DONE_MARKER:
                yield self._format_anthropic_sse('message_stop', {})
                continue

            # Parse JSON data
            try:
                openai_data = orjson.loads(data_content)
            except orjson.JSONDecodeError as e:
                logger.error(f'Failed to parse OpenAI SSE JSON: {e}', content=data_content)
                continue

            # Process the OpenAI chunk
            async for event_bytes in self._process_openai_chunk(openai_data, state):
                yield event_bytes

    async def transform_response(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI non-streaming response to Claude format."""
        response = params['response']

        # Check for annotations to convert (built-in tools)
        if annotations := response.get('annotations'):
            response = self._convert_annotations_to_anthropic(response, annotations)
            self.logger.debug(f'Converted {len(annotations)} OpenAI annotations to Anthropic format')

        try:
            choices = response.get('choices') or []
            if not choices:
                logger.warning('OpenAI response has no choices')
                return response

            choice = choices[0]
            message = choice.get('message', {})
            tool_calls = message.get('tool_calls') or []

            claude_content: List[Dict[str, Any]] = []

            reasoning_block = self._convert_reasoning(message.get('reasoning'))
            if reasoning_block:
                claude_content.append(reasoning_block)

            message_blocks = self._convert_message_content(message)
            if message_blocks:
                claude_content.extend(message_blocks)

            if tool_calls:
                claude_content.extend(self._convert_tool_calls(tool_calls))

            if not claude_content:
                claude_content = [{'type': 'text', 'text': ''}]

            usage = response.get('usage', {})
            claude_usage = self._convert_openai_usage(usage)

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
            logger.error(f'Failed to convert OpenAI response: {e}', exc_info=True)
            return response

    async def _process_openai_chunk(self, data: Dict[str, Any], state: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Process a single OpenAI data chunk and emit Anthropic events."""
        # Extract basic message info
        message_id = data.get('id', '')
        model = data.get('model', '')
        choices = data.get('choices') or []
        usage = data.get('usage')

        # Update state
        if message_id:
            state['message_id'] = message_id
        if model:
            state['model'] = model
        if usage:
            state['usage_tokens'] = usage

        # Handle usage-only chunks (at end)
        if not choices and usage:
            stop_reason = self._convert_stop_reason(state.get('stop_reason'))
            usage_data = self._convert_openai_usage(usage)

            data = {'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage_data}
            yield self._format_anthropic_sse('message_delta', data)
            return

        # Process first choice (OpenAI typically uses index 0)
        if not choices:
            return

        choice = choices[0]
        delta = choice.get('delta', {})
        finish_reason = choice.get('finish_reason')

        # Handle role (message start)
        if 'role' in delta and not state['message_started']:
            yield self._format_anthropic_sse(
                'message_start',
                {
                    'type': 'message_start',
                    'message': {
                        'id': state['message_id'],
                        'type': 'message',
                        'role': delta['role'],
                        'model': state['model'],
                        'content': [],
                        'stop_reason': None,
                        'stop_sequence': None,
                        'usage': {
                            'input_tokens': 0,
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'cache_creation': {'ephemeral_5m_input_tokens': 0, 'ephemeral_1h_input_tokens': 0},
                            'output_tokens': 0,
                            'service_tier': 'standard',
                        },
                    },
                },
            )
            state['message_started'] = True

        # Handle text content
        if delta.get('content') is not None:
            if state['active_text_block'] is None:
                # Start new text block
                block_index = state['next_block_index']
                state['next_block_index'] += 1
                state['active_text_block'] = block_index

                yield self._format_anthropic_sse('content_block_start', {'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})

            # Stream text delta
            yield self._format_anthropic_sse(
                'content_block_delta', {'type': 'content_block_delta', 'index': state['active_text_block'], 'delta': {'type': 'text_delta', 'text': delta['content']}}
            )

        # Handle tool calls
        if delta.get('tool_calls'):
            tool_calls = delta['tool_calls']
            if tool_calls and len(tool_calls) > 0:
                tool_call = tool_calls[0]
                async for event in self._process_tool_call(tool_call, state):
                    yield event

        # Handle finish reason
        if finish_reason:
            state['stop_reason'] = finish_reason

            # Stop active text block
            if state['active_text_block'] is not None:
                yield self._format_anthropic_sse('content_block_stop', {'type': 'content_block_stop', 'index': state['active_text_block']})
                state['active_text_block'] = None

            # Stop active tool block
            if state['active_tool_block'] is not None:
                yield self._format_anthropic_sse('content_block_stop', {'type': 'content_block_stop', 'index': state['active_tool_block']})
                state['active_tool_block'] = None

    async def _process_tool_call(self, tool_call: Dict[str, Any], state: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Process a single OpenAI tool call and emit appropriate Anthropic events."""
        # Check if this is a new tool call (has type: function)
        tool_id = tool_call.get('id')
        function_info = tool_call.get('function', {})
        tool_name = function_info.get('name')
        call_type = tool_call.get('type', '')

        if call_type == 'function':
            # Stop previous active blocks if starting new tool
            if state['active_text_block'] is not None:
                yield self._format_anthropic_sse('content_block_stop', {'type': 'content_block_stop', 'index': state['active_text_block']})
                state['active_text_block'] = None

            if state['active_tool_block'] is not None:
                yield self._format_anthropic_sse('content_block_stop', {'type': 'content_block_stop', 'index': state['active_tool_block']})

            # Start new tool block
            block_index = state['next_block_index']
            state['next_block_index'] += 1
            state['active_tool_block'] = block_index

            yield self._format_anthropic_sse(
                'content_block_start', {'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': tool_name, 'input': {}}}
            )

        # Stream tool arguments
        arguments = function_info.get('arguments')
        if arguments is not None and state['active_tool_block'] is not None:
            yield self._format_anthropic_sse(
                'content_block_delta', {'type': 'content_block_delta', 'index': state['active_tool_block'], 'delta': {'type': 'input_json_delta', 'partial_json': arguments}}
            )

    def _convert_stop_reason(self, openai_finish_reason: Optional[str]) -> str:
        """Convert OpenAI finish reason to Claude format."""
        if not openai_finish_reason:
            return 'end_turn'
        return self.STOP_REASON_MAPPING.get(openai_finish_reason, 'end_turn')

    def _convert_openai_usage(self, openai_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI usage format to Anthropic usage format."""
        if not openai_usage:
            return {'input_tokens': 0, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'output_tokens': 0}

        anthropic_usage = {}

        # Basic token mapping
        if prompt_tokens := openai_usage.get('prompt_tokens'):
            anthropic_usage['input_tokens'] = prompt_tokens
        if completion_tokens := openai_usage.get('completion_tokens'):
            anthropic_usage['output_tokens'] = completion_tokens

        # Cache tokens from prompt_tokens_details
        if prompt_details := openai_usage.get('prompt_tokens_details', {}):
            if cached_tokens := prompt_details.get('cached_tokens'):
                anthropic_usage['cache_read_input_tokens'] = cached_tokens

        # Reasoning tokens from completion_tokens_details (o1 models)
        if completion_details := openai_usage.get('completion_tokens_details', {}):
            if reasoning_tokens := completion_details.get('reasoning_tokens'):
                # Map reasoning tokens to a reasonable Claude equivalent
                # Note: Claude doesn't have direct reasoning token equivalent, but we preserve the info
                anthropic_usage['reasoning_output_tokens'] = reasoning_tokens

        # Default missing fields
        anthropic_usage.setdefault('cache_creation_input_tokens', 0)
        anthropic_usage.setdefault('cache_read_input_tokens', 0)

        return anthropic_usage

    def _convert_annotations_to_anthropic(self, response: dict, annotations: list) -> dict:
        """Convert OpenAI url_citation annotations to Anthropic web_search_tool_result format.

        Args:
            response: OpenAI response dictionary
            annotations: List of annotation objects

        Returns:
            Response with converted annotations
        """
        # Extract message content for snippet extraction
        choices = response.get('choices', [])
        if not choices:
            return response

        message = choices[0].get('message', {})
        content = message.get('content', '')

        # Convert each url_citation annotation
        tool_results = []
        for annotation in annotations:
            if annotation.get('type') == 'url_citation':
                citation = annotation.get('url_citation', {})
                tool_result = self._create_web_search_result(citation, content)
                if tool_result:
                    tool_results.append(tool_result)

        # Add tool results to response if any were created
        if tool_results:
            # Add to message content as tool_use blocks
            message_content = message.get('content', [])
            if isinstance(message_content, str):
                message_content = [{'type': 'text', 'text': message_content}]
            elif not isinstance(message_content, list):
                message_content = []

            # Add tool results
            message_content.extend(tool_results)
            message['content'] = message_content

        return response

    def _create_web_search_result(self, citation: dict, content: Any) -> Optional[dict]:
        """Create Anthropic web_search_tool_result from OpenAI citation.

        Args:
            citation: OpenAI url_citation object
            content: Full response content for snippet extraction

        Returns:
            Anthropic web_search_tool_result block or None if invalid
        """
        url = citation.get('url')
        title = citation.get('title')

        if not url:
            return None

        # Generate deterministic ID from URL
        result_id = f'search_{hashlib.md5(url.encode()).hexdigest()[:8]}'

        # Extract snippet from content using indices if available
        snippet = self._extract_snippet(content, citation.get('start_index'), citation.get('end_index'))

        return {'type': 'web_search_tool_result', 'id': result_id, 'content': {'type': 'web_search_result', 'url': url, 'title': title or 'Untitled', 'snippet': snippet or ''}}

    def _extract_snippet(self, content: Any, start_index: Optional[int], end_index: Optional[int]) -> str:
        """Extract snippet from content using citation indices."""

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = ''.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        else:
            text = ''

        if not text:
            return ''

        if start_index is not None and end_index is not None:
            try:
                return text[start_index:end_index]
            except (IndexError, TypeError):
                self.logger.warning(f'Invalid citation indices: {start_index}-{end_index}')

        return ''

    def _convert_message_content(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert OpenAI message content into Anthropic content blocks."""

        content = message.get('content')
        if content is None:
            return []

        if isinstance(content, str):
            return [{'type': 'text', 'text': content}] if content else []

        if isinstance(content, dict):
            content = [content]

        if not isinstance(content, list):
            return []

        blocks: List[Dict[str, Any]] = []
        for part in content:
            converted = self._convert_content_part(part)
            if not converted:
                continue
            if isinstance(converted, list):
                blocks.extend(converted)
            else:
                blocks.append(converted)

        return blocks

    def _convert_content_part(self, part: Any) -> Optional[Any]:
        """Convert a single OpenAI content part into Anthropic format."""

        if not isinstance(part, dict):
            return None

        part_type = part.get('type')

        if part_type == 'text':
            return {'type': 'text', 'text': part.get('text', '')}

        if part_type == 'image_url':
            return self._convert_image_part(part)

        if part_type == 'tool_result':
            return self._normalize_tool_result_part(part)

        if part_type in {'web_search_tool_result', 'tool_use', 'thinking', 'image'}:
            return part

        self.logger.debug('Dropping unsupported message content', part_type=part_type)
        return None

    def _normalize_tool_result_part(self, part: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure tool_result content is a single text block."""

        content = part.get('content')

        text_value: str
        if isinstance(content, list):
            texts = [item.get('text', '') for item in content if isinstance(item, dict) and item.get('type') == 'text']
            text_value = '\n'.join(filter(None, texts))
        elif isinstance(content, (dict, list)):
            try:
                text_value = orjson.dumps(content).decode()
            except TypeError:
                text_value = json.dumps(content, default=str)
        else:
            text_value = '' if content is None else str(content)

        normalized = part.copy()
        normalized['content'] = [{'type': 'text', 'text': text_value}]
        return normalized

    def _convert_image_part(self, part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert OpenAI image_url part into an Anthropic image block."""

        image_url = part.get('image_url')
        if not isinstance(image_url, dict):
            return None

        url = image_url.get('url')
        if not url:
            return None

        if url.startswith('data:'):
            parsed = self._parse_data_url(url)
            if not parsed:
                self.logger.warning('Unsupported data URL for image content', url=url)
                return None
            media_type, data = parsed
            return {'type': 'image', 'source': {'type': 'base64', 'media_type': media_type, 'data': data}}

        return {'type': 'image', 'source': {'type': 'url', 'url': url}}

    def _parse_data_url(self, url: str) -> Optional[Tuple[str, str]]:
        """Parse a data URL string into (media_type, base64_data)."""

        try:
            header, data = url.split(',', 1)
        except ValueError:
            return None

        metadata = header[len('data:') :]
        parts = metadata.split(';') if metadata else []
        if not parts:
            return None

        media_type = parts[0] or 'application/octet-stream'
        is_base64 = any(p.lower() == 'base64' for p in parts[1:])
        if not is_base64:
            return None

        return media_type, data

    def _convert_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool_calls array into Anthropic tool_use blocks."""

        blocks: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue

            if tool_call.get('type') and tool_call.get('type') != 'function':
                self.logger.debug('Skipping unsupported tool call type', tool_call_type=tool_call.get('type'))
                continue

            function = tool_call.get('function', {}) or {}
            name = function.get('name', '')
            if not name:
                self.logger.warning('Skipping tool call without name', tool_call_id=tool_call.get('id', ''))
                continue

            arguments_payload = function.get('arguments')
            if isinstance(arguments_payload, (dict, list)):
                parsed_arguments = arguments_payload
            elif isinstance(arguments_payload, str):
                try:
                    parsed_arguments = orjson.loads(arguments_payload)
                except orjson.JSONDecodeError:
                    self.logger.warning('Failed to parse tool call arguments', tool_call_id=tool_call.get('id', ''))
                    parsed_arguments = {}
            else:
                parsed_arguments = {}

            blocks.append({'type': 'tool_use', 'id': tool_call.get('id', ''), 'name': name, 'input': parsed_arguments or {}})

        return blocks

    def _convert_reasoning(self, reasoning: Any) -> Optional[Dict[str, Any]]:
        """Convert OpenAI reasoning traces into Anthropic thinking block."""

        if not reasoning:
            return None

        signature: Optional[str] = None
        segments: List[str] = []

        if isinstance(reasoning, dict):
            signature = reasoning.get('signature')

            tokens = reasoning.get('tokens')
            if isinstance(tokens, list):
                for token in tokens:
                    if isinstance(token, dict):
                        text_value = token.get('text') or token.get('content')
                        if text_value:
                            segments.append(str(text_value))
                    elif isinstance(token, str):
                        segments.append(token)

            for key in ('content', 'output_text', 'text'):
                value = reasoning.get(key)
                if isinstance(value, str):
                    segments.append(value)

        elif isinstance(reasoning, list):
            for item in reasoning:
                if isinstance(item, dict):
                    text_value = item.get('text') or item.get('content')
                    if text_value:
                        segments.append(str(text_value))
                elif isinstance(item, str):
                    segments.append(item)
        elif isinstance(reasoning, str):
            segments.append(reasoning)

        thinking_text = ''.join(segments).strip()
        if not thinking_text:
            return None

        block: Dict[str, Any] = {'type': 'thinking', 'thinking': thinking_text}
        if signature:
            block['signature'] = signature
        return block
