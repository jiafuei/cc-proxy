"""Google Gemini transformers for request/response format conversion."""

from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import orjson

from app.services.transformers.interfaces import RequestTransformer, ResponseTransformer


class GeminiRequestTransformer(RequestTransformer):
    """Transformer to convert Anthropic format to Google Gemini format."""

    def __init__(self, logger):
        """Initialize transformer.

        API credentials are obtained from provider config during transform.
        """
        self.logger = logger

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Convert Anthropic request format to Gemini format."""
        request = params['request']
        headers = params['headers']

        # Build Gemini request with core transformations
        gemini_request = {}

        # Convert system instructions if present
        if system := request.get('system'):
            gemini_request['system_instruction'] = self._convert_system(system)

        # Convert messages to contents
        if messages := request.get('messages'):
            gemini_request['contents'] = self._convert_messages(messages)

        # Convert tools if present
        if tools := request.get('tools'):
            gemini_request['tools'] = self._convert_tools(tools)
            # Add tool configuration with sane defaults
            gemini_request['toolConfig'] = {
                'functionCallingConfig': {
                    'mode': 'AUTO'  # Allow model to decide when to call functions
                }
            }

        # Build generation configuration
        generation_config = self._build_generation_config(request)
        if generation_config:
            gemini_request['generationConfig'] = generation_config

        # Handle thinking parameter - Gemini doesn't have direct equivalent
        if thinking := request.get('thinking'):
            self.logger.info(
                f'Thinking parameter detected (budget_tokens: {thinking.get("budget_tokens", 0)}) '
                'but Gemini API does not have direct reasoning effort equivalent. '
                'Consider using a Gemini model optimized for reasoning tasks.'
            )

        # Filter headers for Gemini API
        filtered_headers = self._filter_headers(headers)

        return gemini_request, filtered_headers

    def _convert_system(self, system: Union[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Convert Anthropic system format to Gemini system_instruction format.

        Args:
            system: Anthropic system message(s) - can be string or array of blocks

        Returns:
            Gemini system_instruction object with parts array
        """
        if isinstance(system, str):
            return {'parts': [{'text': system}]}

        # Handle array of system blocks - concatenate text from all text blocks
        system_texts = []
        for block in system:
            if isinstance(block, dict) and block.get('type') == 'text' and 'text' in block:
                system_texts.append(block['text'])

        combined_text = '\n'.join(system_texts) if system_texts else ''
        return {'parts': [{'text': combined_text}]} if combined_text else {'parts': []}

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Anthropic messages to Gemini contents format.

        Args:
            messages: List of Anthropic message objects

        Returns:
            List of Gemini content objects
        """
        contents = []

        for message in messages:
            role = message.get('role')
            content = message.get('content', [])

            # Map role: assistant -> model, keep user as-is
            gemini_role = 'model' if role == 'assistant' else role

            # Convert content to parts
            parts = self._convert_content_blocks(content)

            if parts:  # Only add content if there are parts
                contents.append({'role': gemini_role, 'parts': parts})

        return contents

    def _convert_content_blocks(self, content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Convert Anthropic content blocks to Gemini parts format.

        Args:
            content: Anthropic content - can be string or array of content blocks

        Returns:
            List of Gemini part objects
        """
        if isinstance(content, str):
            return [{'text': content}]

        if not isinstance(content, list):
            return []

        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get('type')

            if block_type == 'text' and 'text' in block:
                parts.append({'text': block['text']})

            elif block_type == 'image' and 'source' in block:
                image_part = self._convert_image_block(block)
                if image_part:
                    parts.append(image_part)

            elif block_type == 'tool_use':
                tool_part = self._convert_tool_use_block(block)
                if tool_part:
                    parts.append(tool_part)

            elif block_type == 'tool_result':
                tool_result_part = self._convert_tool_result_block(block)
                if tool_result_part:
                    parts.append(tool_result_part)

            # Handle thinking blocks - not supported by Gemini
            elif block_type == 'thinking':
                self.logger.debug('Thinking content block skipped - not supported by Gemini API')

            # Skip other unsupported types
            elif block_type not in ['text', 'image', 'tool_use', 'tool_result', 'thinking']:
                self.logger.debug(f'Unsupported content block type: {block_type}')

        return parts

    def _convert_image_block(self, block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Anthropic image block to Gemini inline_data format.

        Args:
            block: Anthropic image block

        Returns:
            Gemini inline_data part or None if conversion fails
        """
        source = block.get('source', {})
        if source.get('type') != 'base64':
            self.logger.warning(f'Unsupported image source type: {source.get("type")}')
            return None

        data = source.get('data')
        media_type = source.get('media_type', 'image/jpeg')

        if not data:
            self.logger.warning('Image block missing data')
            return None

        return {'inline_data': {'mime_type': media_type, 'data': data}}

    def _convert_tool_use_block(self, block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Anthropic tool_use block to Gemini function call format.

        Args:
            block: Anthropic tool_use block

        Returns:
            Gemini function call part or None if conversion fails
        """
        tool_name = block.get('name')
        tool_input = block.get('input', {})

        if not tool_name:
            self.logger.warning('Tool use block missing name')
            return None

        # Updated format based on Gemini API documentation
        return {'functionCall': {'name': tool_name, 'args': tool_input}}

    def _convert_tool_result_block(self, block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Anthropic tool_result block to Gemini function response format.

        Args:
            block: Anthropic tool_result block

        Returns:
            Gemini function response part or None if conversion fails
        """
        tool_use_id = block.get('tool_use_id')
        content = block.get('content', '')
        is_error = block.get('is_error', False)

        if not tool_use_id:
            self.logger.warning('Tool result block missing tool_use_id')
            return None

        # Handle content format - could be string or structured
        if isinstance(content, str):
            response_content = content
        elif isinstance(content, list):
            # Extract text from content blocks if needed
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif isinstance(item, str):
                    text_parts.append(item)
            response_content = '\n'.join(text_parts)
        else:
            response_content = str(content)

        # Updated format based on Gemini API documentation
        return {
            'functionResponse': {
                'name': tool_use_id,  # In Gemini, this should match the function name
                'response': {'content': response_content, 'success': not is_error},
            }
        }

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Anthropic tools to Gemini function_declarations format.

        Args:
            tools: List of Anthropic tool definitions

        Returns:
            List of Gemini tool objects with function_declarations
        """
        if not tools:
            return []

        function_declarations = []

        for tool in tools:
            name = tool.get('name')
            description = tool.get('description', '')
            input_schema = tool.get('input_schema', {})

            if not name:
                self.logger.warning('Tool missing name, skipping')
                continue

            function_declaration = {'name': name, 'description': description, 'parameters': input_schema}

            function_declarations.append(function_declaration)

        return [{'functionDeclarations': function_declarations}] if function_declarations else []

    def _build_generation_config(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build Gemini generationConfig from Anthropic request parameters.

        Args:
            request: Anthropic request object

        Returns:
            Gemini generationConfig object or None if no config needed
        """
        config = {}

        if temperature := request.get('temperature'):
            config['temperature'] = temperature

        if max_tokens := request.get('max_tokens'):
            config['maxOutputTokens'] = max_tokens

        # Map Anthropic parameters to Gemini equivalents
        if stop_sequences := request.get('stop_sequences'):
            config['stopSequences'] = stop_sequences

        if top_p := request.get('top_p'):
            config['topP'] = top_p

        if top_k := request.get('top_k'):
            config['topK'] = top_k

        # Always set candidateCount to 1 for Anthropic compatibility
        config['candidateCount'] = 1

        return config if config else None

    def _filter_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Filter headers for Gemini API compatibility.

        Args:
            headers: Original request headers

        Returns:
            Filtered headers suitable for Gemini API (no authentication headers)
        """
        filtered_headers = {}

        # Keep basic headers (but not authentication - Gemini uses query params)
        for key, value in headers.items():
            key_lower = key.lower()

            # Keep basic headers but exclude authentication headers
            if any(key_lower.startswith(prefix) for prefix in ['user-agent', 'accept', 'content-type']):
                filtered_headers[key] = value

        return filtered_headers


class GeminiResponseTransformer(ResponseTransformer):
    """Transformer to convert Gemini responses to Anthropic format."""

    # Stop reason mapping from Gemini to Anthropic format
    STOP_REASON_MAPPING = {'STOP': 'end_turn', 'MAX_TOKENS': 'max_tokens', 'SAFETY': 'stop_sequence', 'RECITATION': 'stop_sequence', 'OTHER': 'end_turn'}

    # Gemini SSE constants (may need adjustment based on actual format)
    DATA_PREFIX = b'data:'
    DONE_MARKER = b'[DONE]'

    def __init__(self, logger):
        """Initialize transformer."""
        self.logger = logger

    def _init_state(self) -> Dict[str, Any]:
        """Initialize processing state for streaming response."""
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
        """Transform Gemini SSE chunk to Anthropic SSE format."""
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
                self.logger.debug(f'Non-data SSE line: {line}')
                continue

            # Handle [DONE] marker
            data_content = line[len(self.DATA_PREFIX) :].strip()
            if data_content == self.DONE_MARKER:
                yield self._format_anthropic_sse('message_stop', {})
                continue

            # Parse JSON data
            try:
                gemini_data = orjson.loads(data_content)
            except orjson.JSONDecodeError as e:
                self.logger.error(f'Failed to parse Gemini SSE JSON: {e}', content=data_content)
                continue

            # Process the Gemini chunk
            async for event_bytes in self._process_gemini_chunk(gemini_data, state):
                yield event_bytes

    async def transform_response(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convert Gemini non-streaming response to Anthropic format."""
        response = params['response']

        try:
            # Extract candidates from Gemini response
            candidates = response.get('candidates', [])
            if not candidates:
                self.logger.warning('Gemini response has no candidates')
                return response

            candidate = candidates[0]
            content_parts = candidate.get('content', {}).get('parts', [])
            finish_reason = candidate.get('finishReason')

            # Build Anthropic content array
            anthropic_content = []

            for part in content_parts:
                if 'text' in part:
                    anthropic_content.append({'type': 'text', 'text': part['text']})
                elif 'functionCall' in part:
                    # Convert Gemini function call to Anthropic tool_use
                    function_call = part['functionCall']
                    # Generate an ID since Gemini doesn't provide one
                    tool_id = f'toolu_{hash(str(function_call))}'[:12]
                    anthropic_content.append({'type': 'tool_use', 'id': tool_id, 'name': function_call.get('name', ''), 'input': function_call.get('args', {})})
                # Handle other part types if needed

            # Build usage info
            usage_metadata = response.get('usageMetadata', {})
            anthropic_usage = self._convert_gemini_usage(usage_metadata)

            # Enhanced response field extraction
            anthropic_response = {
                'id': response.get('responseId', response.get('id', f'msg_{hash(str(response))}'))[:32],
                'type': 'message',
                'role': 'assistant',
                'content': anthropic_content,
                'model': response.get('modelVersion', response.get('model', '')),
                'stop_reason': self._convert_stop_reason(finish_reason),
                'stop_sequence': None,
                'usage': anthropic_usage,
            }

            self.logger.debug('Converted Gemini response to Anthropic format')
            return anthropic_response

        except Exception as e:
            self.logger.error(f'Failed to convert Gemini response: {e}')
            return response

    async def _process_gemini_chunk(self, data: Dict[str, Any], state: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Process a single Gemini data chunk and emit Anthropic events."""
        candidates = data.get('candidates', [])
        if not candidates:
            # Handle usage metadata if present
            if usage_metadata := data.get('usageMetadata'):
                state['usage_tokens'] = self._convert_gemini_usage(usage_metadata)
            return

        candidate = candidates[0]
        content = candidate.get('content', {})
        parts = content.get('parts', [])

        # Handle message start
        if not state['message_started'] and (parts or candidate.get('finishReason')):
            message_id = data.get('responseId', data.get('id', f'msg_{hash(str(data))}'))[:32]
            state['message_id'] = message_id

            yield self._format_anthropic_sse(
                'message_start',
                {
                    'type': 'message_start',
                    'message': {
                        'id': message_id,
                        'type': 'message',
                        'role': 'assistant',
                        'model': data.get('modelVersion', data.get('model', '')),
                        'content': [],
                        'stop_reason': None,
                        'stop_sequence': None,
                        'usage': {'input_tokens': 0, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'output_tokens': 0},
                    },
                },
            )
            state['message_started'] = True

        # Process content parts
        for part in parts:
            if 'text' in part:
                # Handle text content
                if state['active_text_block'] is None:
                    block_index = state['next_block_index']
                    state['next_block_index'] += 1
                    state['active_text_block'] = block_index

                    yield self._format_anthropic_sse('content_block_start', {'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})

                # Stream text delta
                yield self._format_anthropic_sse(
                    'content_block_delta', {'type': 'content_block_delta', 'index': state['active_text_block'], 'delta': {'type': 'text_delta', 'text': part['text']}}
                )

            elif 'functionCall' in part:
                # Handle function calls in streaming
                function_call = part['functionCall']
                block_index = state['next_block_index']
                state['next_block_index'] += 1
                state['active_tool_block'] = block_index

                tool_id = f'toolu_{hash(str(function_call))}'[:12]

                yield self._format_anthropic_sse(
                    'content_block_start',
                    {
                        'type': 'content_block_start',
                        'index': block_index,
                        'content_block': {'type': 'tool_use', 'id': tool_id, 'name': function_call.get('name', ''), 'input': function_call.get('args', {})},
                    },
                )

                yield self._format_anthropic_sse('content_block_stop', {'type': 'content_block_stop', 'index': block_index})

                state['active_tool_block'] = None

        # Handle finish reason
        finish_reason = candidate.get('finishReason')
        if finish_reason:
            state['stop_reason'] = finish_reason

            # Stop active blocks
            if state['active_text_block'] is not None:
                yield self._format_anthropic_sse('content_block_stop', {'type': 'content_block_stop', 'index': state['active_text_block']})
                state['active_text_block'] = None

            # Emit usage metadata if available
            usage_metadata = data.get('usageMetadata') or state.get('usage_tokens', {})
            anthropic_usage = self._convert_gemini_usage(usage_metadata)

            yield self._format_anthropic_sse(
                'message_delta', {'type': 'message_delta', 'delta': {'stop_reason': self._convert_stop_reason(finish_reason), 'stop_sequence': None}, 'usage': anthropic_usage}
            )

    def _convert_stop_reason(self, gemini_finish_reason: Optional[str]) -> str:
        """Convert Gemini finish reason to Anthropic format."""
        if not gemini_finish_reason:
            return 'end_turn'
        return self.STOP_REASON_MAPPING.get(gemini_finish_reason, 'end_turn')

    def _convert_gemini_usage(self, usage_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Gemini usage format to Anthropic usage format."""
        if not usage_metadata:
            return {'input_tokens': 0, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'output_tokens': 0}

        anthropic_usage = {}

        # Map basic tokens
        if prompt_tokens := usage_metadata.get('promptTokenCount'):
            anthropic_usage['input_tokens'] = prompt_tokens
        else:
            anthropic_usage['input_tokens'] = 0

        if output_tokens := usage_metadata.get('candidatesTokenCount'):
            anthropic_usage['output_tokens'] = output_tokens
        else:
            anthropic_usage['output_tokens'] = 0

        # Map cached content tokens if available
        if cached_tokens := usage_metadata.get('cachedContentTokenCount'):
            anthropic_usage['cache_read_input_tokens'] = cached_tokens
        else:
            anthropic_usage['cache_read_input_tokens'] = 0

        # Always set cache creation to 0 (Gemini doesn't have equivalent)
        anthropic_usage['cache_creation_input_tokens'] = 0

        return anthropic_usage
