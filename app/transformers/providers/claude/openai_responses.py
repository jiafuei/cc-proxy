"""Claude channel transformers for the OpenAI Responses API."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import orjson

from app.transformers.interfaces import ProviderRequestTransformer, ProviderResponseTransformer


class ClaudeOpenAIResponsesRequestTransformer(ProviderRequestTransformer):
    """Convert Anthropic style payloads into OpenAI Responses API requests."""

    REASONING_EFFORT_THRESHOLDS: Tuple[Tuple[int | float, str], ...] = ((1024, 'low'), (8192, 'medium'), (float('inf'), 'high'))

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        request = copy.deepcopy(params['request'])
        headers = params['headers']

        payload: Dict[str, Any] = {
            'model': request.get('model'),
            'stream': False,
            'store': False,
        }

        instructions, system_items = self._extract_system_sections(request.get('system'))
        message_items = self._convert_messages(request.get('messages') or [])

        input_items = []
        if system_items:
            input_items.extend(system_items)
        if message_items:
            input_items.extend(message_items)

        if instructions:
            payload['instructions'] = instructions

        if input_items:
            payload['input'] = input_items
        else:
            payload['input'] = ''

        reasoning = self._convert_reasoning(request.get('thinking'))
        if reasoning:
            payload['reasoning'] = reasoning

        metadata = self._convert_metadata(request.get('metadata'))
        if metadata:
            payload['metadata'] = metadata

        for scalar_key, target_key, clamp in (
            ('temperature', 'temperature', (0.0, 2.0)),
            ('top_p', 'top_p', (0.0, 1.0)),
            ('top_k', 'top_k', (0, None)),
            ('presence_penalty', 'presence_penalty', (-2.0, 2.0)),
            ('frequency_penalty', 'frequency_penalty', (-2.0, 2.0)),
        ):
            value = request.get(scalar_key)
            if value is None:
                continue
            payload[target_key] = self._clamp_numeric(value, *clamp)

        if (max_tokens := request.get('max_tokens')) is not None:
            payload['max_output_tokens'] = max(0, int(max_tokens))

        tools_data = request.get('tools') or []
        builtin_tools = [tool for tool in tools_data if self._is_builtin_tool(tool)]
        function_tools = [tool for tool in tools_data if not self._is_builtin_tool(tool)]

        converted_functions = self._convert_tools(function_tools)
        converted_builtins = self._convert_builtin_tools(builtin_tools)

        if converted_functions or converted_builtins:
            payload['tools'] = []
            if converted_functions:
                payload['tools'].extend(converted_functions)
            if converted_builtins:
                payload['tools'].extend(converted_builtins)

        tool_choice, parallel_tool_calls = self._convert_tool_choice(request.get('tool_choice'))
        if parallel_tool_calls is not None:
            payload['parallel_tool_calls'] = parallel_tool_calls
        if tool_choice is not None:
            payload['tool_choice'] = tool_choice

        response_format = self._convert_response_format(request.get('response_format'))
        if response_format:
            payload['response_format'] = response_format

        if request.get('modalities'):
            payload['modalities'] = copy.deepcopy(request['modalities'])
        if request.get('attachments'):
            payload['attachments'] = copy.deepcopy(request['attachments'])
        payload['previous_response_id'] = request.get('previous_response_id')

        clean_payload = {k: v for k, v in payload.items() if v is not None}

        return clean_payload, headers

    def _extract_system_sections(self, system_blocks: Optional[Any]) -> Tuple[str, List[Dict[str, Any]]]:
        if not system_blocks:
            return '', []

        if isinstance(system_blocks, str):
            return system_blocks, []

        if not isinstance(system_blocks, list):
            return '', []

        text_fragments: List[str] = []
        residual_items: List[Dict[str, Any]] = []
        for block in system_blocks:
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'text':
                text = block.get('text')
                if text:
                    text_fragments.append(text)
                continue

            residual_items.extend(self._convert_message({'role': 'system', 'content': [block]}))

        instruction_text = '\n'.join(fragment for fragment in text_fragments if fragment)
        return instruction_text, residual_items

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            converted.extend(self._convert_message(message))
        return converted

    def _convert_message(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        role = message.get('role', 'user')
        raw_content = message.get('content', [])
        if isinstance(raw_content, str):
            raw_content = [{'type': 'text', 'text': raw_content}]
        if not isinstance(raw_content, list):
            self.logger.warning('Skipping message with unsupported content type', role=role)
            return []

        items: List[Dict[str, Any]] = []
        message_parts: List[Dict[str, Any]] = []

        for block in raw_content:
            if not isinstance(block, dict):
                continue
            block_type = block.get('type')
            if block_type == 'text':
                part = self._convert_textual_block(role, block)
                if part:
                    message_parts.append(part)
            elif block_type == 'image':
                part = self._convert_image_block(block)
                if part:
                    message_parts.append(part)
            elif block_type == 'tool_use':
                if message_parts:
                    items.append(self._build_message_item(role, message_parts))
                    message_parts = []
                tool_call = self._convert_tool_use_block(block)
                if tool_call:
                    items.append(tool_call)
            elif block_type == 'tool_result':
                if message_parts:
                    items.append(self._build_message_item(role, message_parts))
                    message_parts = []
                result_item = self._convert_tool_result_block(block)
                if result_item:
                    items.append(result_item)
            elif block_type == 'thinking':
                continue
            else:
                self.logger.debug('Dropping unsupported content block', block_type=block_type)

        if message_parts:
            items.append(self._build_message_item(role, message_parts))

        return items

    def _build_message_item(self, role: str, parts: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not parts:
            return {}
        message_item: Dict[str, Any] = {'type': 'message', 'role': role, 'content': parts}
        return message_item

    def _convert_textual_block(self, role: str, block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = block.get('text')
        if text is None:
            return None
        content_type = 'output_text' if role == 'assistant' else 'input_text'
        return {'type': content_type, 'text': text}

    def _convert_image_block(self, block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        source = block.get('source')
        if not isinstance(source, dict):
            return None

        if source.get('type') == 'base64':
            data = source.get('data')
            media_type = source.get('media_type', 'image/png')
            if not data:
                return None
            return {'type': 'input_image', 'image_url': f'data:{media_type};base64,{data}'}

        if source.get('type') == 'url' and source.get('url'):
            return {'type': 'input_image', 'image_url': source['url']}

        return None

    def _convert_tool_use_block(self, block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        call_id = block.get('id') or block.get('tool_call_id')
        name = block.get('name')
        arguments = block.get('input') or {}
        try:
            arguments_json = orjson.dumps(arguments).decode()
        except TypeError:
            arguments_json = json.dumps(arguments, default=str)

        if not name:
            self.logger.warning('Tool use block missing name; dropping call')
            return None

        tool_call: Dict[str, Any] = {'type': 'function_call', 'name': name, 'arguments': arguments_json}
        if call_id:
            tool_call['call_id'] = call_id
        return tool_call

    def _convert_tool_result_block(self, block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        call_id = block.get('tool_use_id') or block.get('id')
        content = block.get('content')

        if isinstance(content, list):
            texts = [part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text']
            output = '\n'.join(texts)
        elif isinstance(content, (dict, list)):
            output = orjson.dumps(content).decode()
        else:
            output = str(content or '')

        result: Dict[str, Any] = {'type': 'function_call_output', 'output': output}
        if call_id:
            result['call_id'] = call_id
        if block.get('is_error'):
            result['is_error'] = True
        return result

    def _convert_reasoning(self, thinking: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not thinking:
            return None

        budget = thinking.get('budget_tokens')
        if not isinstance(budget, (int, float)) or budget <= 0:
            return None

        for threshold, effort in self.REASONING_EFFORT_THRESHOLDS:
            if budget < threshold:
                return {'effort': effort}
        return None

    def _convert_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        metadata_copy = metadata.copy() if isinstance(metadata, dict) else {}
        metadata_copy.setdefault('source', 'cc-proxy')
        return metadata_copy

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if self._is_builtin_tool(tool):
                self.logger.warning('Skipping unsupported built-in tool in Responses transformer', tool=tool.get('name'))
                continue
            name = tool.get('name')
            parameters = tool.get('input_schema') or {}
            description = tool.get('description', '')
            if not name:
                self.logger.warning('Skipping function tool without name')
                continue
            converted.append({'type': 'function', 'name': name, 'description': description, 'parameters': parameters})
        return converted

    def _convert_builtin_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        builtin_entries: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue

            name = (tool.get('name') or '').lower()
            if name == 'web_search':
                entry = self._convert_web_search_tool(tool)
                if entry:
                    builtin_entries.append(entry)
            else:
                self.logger.warning('Unsupported built-in tool in Responses transformer', tool=name)

        return builtin_entries

    def _convert_web_search_tool(self, tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            config = self._extract_websearch_config(tool)
        except ValueError as exc:
            self.logger.warning('Invalid web search configuration; dropping tool', error=str(exc))
            return None

        return {'type': 'web_search', 'web_search': config}

    def _extract_websearch_config(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_domain_filters(tool)

        config: Dict[str, Any] = {}

        allowed_domains = tool.get('allowed_domains')
        blocked_domains = tool.get('blocked_domains')
        if allowed_domains or blocked_domains:
            filters: Dict[str, Any] = {}
            if allowed_domains:
                filters['allowed_domains'] = allowed_domains
            if blocked_domains:
                filters['blocked_domains'] = blocked_domains
            config['filters'] = filters

        if user_location := tool.get('user_location'):
            config['user_location'] = self._convert_user_location(user_location)

        # Search context size defaults to medium but allow override
        config['search_context_size'] = tool.get('search_context_size', 'medium')

        return self._handle_missing_parameters(config)

    def _convert_user_location(self, user_location: Dict[str, Any]) -> Dict[str, Any]:
        openai_location = {'type': 'approximate', 'approximate': {}}
        approximate = openai_location['approximate']

        for field in ['country', 'city', 'region', 'timezone']:
            if value := user_location.get(field):
                approximate[field] = value

        return openai_location

    def _validate_domain_filters(self, tool: Dict[str, Any]) -> None:
        if tool.get('allowed_domains') and tool.get('blocked_domains'):
            raise ValueError('Cannot use both allowed_domains and blocked_domains in WebSearch tool')

    def _handle_missing_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config.setdefault('search_context_size', 'medium')
        config.setdefault('filters', {})
        return config

    def _convert_tool_choice(self, tool_choice: Any) -> Tuple[Optional[Any], Optional[bool]]:
        if tool_choice is None:
            return None, None

        if isinstance(tool_choice, str):
            lowered = tool_choice.lower()
            if lowered in {'auto', 'any'}:
                return 'auto', True
            if lowered in {'none'}:
                return {'type': 'none'}, False
            return {'type': 'function', 'function': {'name': lowered}}, False

        if isinstance(tool_choice, dict):
            choice_type = tool_choice.get('type')
            if choice_type in {'auto', 'any'}:
                return 'auto', True
            if choice_type in {'none'}:
                return {'type': 'none'}, False
            if choice_type in {'tool', 'function'} and tool_choice.get('name'):
                return {'type': 'function', 'function': {'name': tool_choice['name']}}, False

        self.logger.debug('Unable to map tool_choice; defaulting to auto')
        return 'auto', True

    def _convert_response_format(self, response_format: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(response_format, dict):
            return None
        if 'type' not in response_format:
            return None
        allowed_keys = {'type', 'json_schema', 'strict'}
        return {k: v for k, v in response_format.items() if k in allowed_keys}

    def _clamp_numeric(self, value: Any, lower: Optional[float], upper: Optional[float]) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if lower is not None:
            numeric = max(lower, numeric)
        if upper is not None:
            numeric = min(upper, numeric)
        return numeric


class ClaudeOpenAIResponsesResponseTransformer(ProviderResponseTransformer):
    """Convert OpenAI Responses API payloads into Anthropic-compatible responses."""

    STATUS_STOP_REASON = {
        'completed': 'end_turn',
        'failed': 'error',
        'cancelled': 'cancelled',
        'in_progress': 'incomplete',
        'requires_action': 'tool_use',
    }

    async def transform_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        response = params['response']

        if error := response.get('error'):
            return self._convert_error(error)

        return self._convert_success_response(response)

    def _convert_success_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        content_blocks: List[Dict[str, Any]] = []
        for item in response.get('output') or []:
            item_type = item.get('type')
            if item_type == 'message':
                content_blocks.extend(self._convert_message_item(item))
            elif item_type == 'function_call':
                tool_block = self._convert_function_call(item)
                if tool_block:
                    content_blocks.append(tool_block)
            elif item_type == 'function_call_output':
                tool_result = self._convert_function_call_output(item)
                if tool_result:
                    content_blocks.append(tool_result)
            else:
                self.logger.debug('Unhandled Responses output item', item_type=item_type)

        if not content_blocks:
            content_blocks = [{'type': 'text', 'text': ''}]

        usage = self._convert_usage(response.get('usage'))
        stop_reason = self.STATUS_STOP_REASON.get(response.get('status'), 'end_turn')

        claude_response = {
            'id': response.get('id', ''),
            'type': 'message',
            'role': 'assistant',
            'content': content_blocks,
            'model': response.get('model', ''),
            'stop_reason': stop_reason,
            'stop_sequence': None,
            'usage': usage,
        }

        return claude_response

    def _convert_message_item(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = item.get('content')
        if not isinstance(content, list):
            return []

        blocks: List[Dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get('type')
            if part_type == 'output_text':
                blocks.append({'type': 'text', 'text': part.get('text', '')})
            elif part_type == 'output_image':
                image_url = part.get('image_url')
                if image_url:
                    blocks.append({'type': 'image', 'source': {'type': 'url', 'url': image_url}})
            elif part_type == 'reasoning':
                reasoning_block = self._convert_reasoning_part(part)
                if reasoning_block:
                    if isinstance(reasoning_block, list):
                        blocks.extend(reasoning_block)
                    else:
                        blocks.append(reasoning_block)
            elif part_type == 'web_search_result':
                search_blocks = self._convert_web_search_part(part)
                if search_blocks:
                    if isinstance(search_blocks, list):
                        blocks.extend(search_blocks)
                    else:
                        blocks.append(search_blocks)
            else:
                self.logger.debug('Dropping unsupported message content part', part_type=part_type)
        return blocks

    def _convert_function_call(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        name = item.get('name')
        if not name:
            return None
        call_id = item.get('call_id') or item.get('id') or ''
        arguments = item.get('arguments')
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = arguments
        else:
            parsed_arguments = arguments or {}
        return {'type': 'tool_use', 'id': call_id, 'name': name, 'input': parsed_arguments}

    def _convert_function_call_output(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        call_id = item.get('call_id')
        output = item.get('output')
        if output is None:
            return None
        if isinstance(output, (dict, list)):
            try:
                output_text = orjson.dumps(output).decode()
            except TypeError:
                output_text = json.dumps(output, default=str)
        else:
            output_text = str(output or '')

        tool_result: Dict[str, Any] = {
            'type': 'tool_result',
            'tool_use_id': call_id or '',
            'content': [{'type': 'text', 'text': output_text}],
        }
        if item.get('is_error'):
            tool_result['is_error'] = True
        return tool_result

    def _convert_reasoning_part(self, part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        signature = part.get('signature')
        segments: List[str] = []

        reasoning_value = part.get('reasoning')
        if isinstance(reasoning_value, str):
            segments.append(reasoning_value)
        elif isinstance(reasoning_value, list):
            for item in reasoning_value:
                if isinstance(item, str):
                    segments.append(item)
                elif isinstance(item, dict):
                    if text := item.get('text') or item.get('content'):
                        segments.append(str(text))

        for key in ('text', 'content'):
            value = part.get(key)
            if isinstance(value, str):
                segments.append(value)

        details = part.get('details')
        if isinstance(details, dict):
            if detail_text := details.get('text'):
                segments.append(str(detail_text))

        thinking_text = ''.join(segments).strip()
        if not thinking_text:
            return None

        block: Dict[str, Any] = {'type': 'thinking', 'thinking': thinking_text}
        if signature:
            block['signature'] = signature
        return block

    def _convert_web_search_part(self, part: Dict[str, Any]) -> Optional[Any]:
        data: Any = part.get('web_search_result')
        if data is None and 'results' in part:
            data = part

        if isinstance(data, dict):
            results = data.get('results')
            if isinstance(results, list) and results:
                blocks: List[Dict[str, Any]] = []
                for entry in results:
                    block = self._build_web_search_block(entry)
                    if block:
                        blocks.append(block)
                return blocks or None

            return self._build_web_search_block(data)

        if isinstance(data, list):
            blocks: List[Dict[str, Any]] = []
            for entry in data:
                block = self._build_web_search_block(entry)
                if block:
                    blocks.append(block)
            return blocks or None

        return None

    def _build_web_search_block(self, entry: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(entry, dict):
            return None

        url = entry.get('url') or entry.get('link')
        title = entry.get('title') or entry.get('name') or 'Untitled'
        snippet = entry.get('snippet') or entry.get('text') or entry.get('description') or ''

        if not url and not snippet:
            return None

        identifier_source = url or snippet
        digest = hashlib.md5(identifier_source.encode()).hexdigest()[:8]  # nosec - deterministic id only

        content_payload = {'type': 'web_search_result', 'url': url or '', 'title': title, 'snippet': snippet}
        return {'type': 'web_search_tool_result', 'id': f'search_{digest}', 'content': content_payload}

    def _convert_usage(self, usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(usage, dict):
            return {'input_tokens': 0, 'output_tokens': 0}

        converted = {
            'input_tokens': usage.get('input_tokens', 0),
            'output_tokens': usage.get('output_tokens', 0),
        }

        if 'total_tokens' in usage:
            converted['total_tokens'] = usage['total_tokens']

        prompt_details = usage.get('prompt_tokens_details')
        if isinstance(prompt_details, dict) and 'cached_tokens' in prompt_details:
            converted['cache_read_input_tokens'] = prompt_details['cached_tokens']

        details = usage.get('output_tokens_details') or usage.get('completion_tokens_details')
        if isinstance(details, dict) and 'reasoning_tokens' in details:
            converted['reasoning_output_tokens'] = details['reasoning_tokens']

        converted.setdefault('cache_read_input_tokens', 0)

        return converted

    def _convert_error(self, error: Dict[str, Any]) -> Dict[str, Any]:
        message = error.get('message', 'OpenAI Responses API error')
        error_type = error.get('type', 'api_error')
        code = error.get('code')
        details = {'type': 'error', 'error': {'type': error_type, 'message': message}}
        if code:
            details['error']['code'] = code
        return details
