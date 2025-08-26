"""Anthropic transformers - pure passthrough implementations."""

import random
from typing import Any, Dict, Tuple

from app.services.transformers.interfaces import RequestTransformer, ResponseTransformer


class AnthropicAuthTransformer(RequestTransformer):
    """Pure passthrough transformer for Anthropic requests.

    Since incoming requests are already in Claude/Anthropic format,
    no transformation is needed.
    """

    def __init__(self, logger):
        """Initialize transformer.

        API credentials are obtained from provider config during transform.
        """
        self.logger = logger

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Pure passthrough - incoming format is already Anthropic format."""
        request: dict[str, Any] = params['request']
        headers: dict[str, str] = params['headers']
        provider_config = params['provider_config']

        # Get API key from provider config
        api_key = provider_config.api_key

        final_headers = {
            k: v
            for k, v in headers.items()
            if any(
                (
                    k.startswith(prefix)
                    for prefix in (
                        'x-',
                        'anthropic',
                        'user-',
                    )
                )
            )
        }
        final_headers = final_headers | {'authorization': f'Bearer {api_key}'}
        return request, final_headers


class AnthropicResponseTransformer(ResponseTransformer):
    """Pure passthrough transformer for Anthropic responses."""

    async def transform_chunk(self, params: Dict[str, Any]) -> bytes:
        """Pure passthrough - response is already in correct format."""
        return params['chunk']

    async def transform_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Pure passthrough - response is already in correct format."""
        return params['response']


class AnthropicCacheTransformer(RequestTransformer):
    """Optimizes cache breakpoints given the limit of 4 cache breakpoints

    - Removes the cache breakpoint from the 'You are Claude' system message
    - Insert cache breakpoint at last system message
    - Reorder 'tools' array
        1. default tools
        2. MCP tools
    - Insert breakpoints for tools
    - Insert cache breakpoints every 20 content blocks (excluding 'thinking')

    Docs:
    - https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    - https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-thinking-with-prompt-caching

    """

    def __init__(self, logger, max_tools_breakpoints: int = 1):
        self.logger = logger
        self.id = random.random() * 100_000
        self.max_tools_breakpoints = max_tools_breakpoints

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        request, headers, routing_key = params['request'], params['headers'], params['routing_key']

        # Skip processing for background messages
        if routing_key == 'background':
            return request, headers

        # Remove existing cache breakpoints
        self._remove_system_cache_breakpoints(request)
        self._remove_tool_cache_breakpoints(request)
        self._remove_messages_cache_breakpoints(request)

        # Apply optimal caching strategy in correct hierarchy order (tools → system → messages)
        tools_breakpoints = self._reorder_and_cache_tools_array(request)
        system_breakpoints = self._insert_system_cache_bp(request, tools_breakpoints)
        self._insert_messages_cache_bp(request, tools_breakpoints + system_breakpoints)

        # Validate and log cache strategy
        total_breakpoints = self._validate_breakpoint_count(request)
        self.logger.info(f'Applied {total_breakpoints}/4 cache breakpoints for routing_key: {routing_key}')

        return request, headers

    def _remove_system_cache_breakpoints(self, request: dict[str, Any]):
        if 'system' not in request:
            return

        system_arr = [{k: v for k, v in block.items() if k != 'cache_control'} for block in request.get('system', [])]
        request['system'] = system_arr

    def _remove_tool_cache_breakpoints(self, request: dict[str, Any]):
        """Tools seems to be cached even without cache_control"""

        if 'tools' not in request:
            return

        tools_arr = [{k: v for k, v in block.items() if k != 'cache_control'} for block in request.get('tools', [])]
        request['tools'] = tools_arr

    def _remove_messages_cache_breakpoints(self, request: dict[str, Any]):
        if 'messages' not in request:
            return

        messages = []
        for message in request.get('messages', []):
            if 'content' not in message:
                continue

            if isinstance(message['content'], str):
                messages.append({k: v for k, v in message.items() if k != 'cache_control'})
                continue

            if not isinstance(message['content'], list):
                self.logger.warn('unknown message format', message=message)
                messages.append(message)
                continue

            final_content = []
            for content in message['content']:
                final_content.append({k: v for k, v in content.items() if k != 'cache_control'})
            message['content'] = final_content
            messages.append(message)

        request['messages'] = messages

    def _reorder_and_cache_tools_array(self, request: dict[str, Any]) -> int:
        """Reorder tools (default first, MCP second) and add strategic cache breakpoints.

        Returns:
            Number of breakpoints used for tools caching
        """
        tools = request.get('tools', [])
        if not tools:
            return 0

        # Single-pass tool separation for performance
        default_tools = []
        mcp_tools = []
        for tool in tools:
            if tool.get('name', '').startswith('mcp__'):
                mcp_tools.append(tool)
            else:
                default_tools.append(tool)

        # Reorder: defaults first, MCP second
        reordered_tools = default_tools + mcp_tools
        total_tools = len(default_tools + mcp_tools)
        breakpoints_used = 0

        for i in range(0, total_tools, 20):
            if i + 20 > total_tools:
                break
            if breakpoints_used >= self.max_tools_breakpoints:
                break

            reordered_tools[i + 20 - 1]['cache_control'] = {'type': 'ephemeral'}
            breakpoints_used += 1

        if breakpoints_used == 0:
            reordered_tools[-1]['cache_control'] = {'type': 'ephemeral'}
            breakpoints_used += 1

        self.logger.debug(f'Added {breakpoints_used} breakpoint for {total_tools} total tools')
        request['tools'] = reordered_tools
        return breakpoints_used

    def _insert_system_cache_bp(self, request: dict[str, Any], used_breakpoints: int) -> int:
        """Cache only the last system message (largest, most stable content).

        Args:
            used_breakpoints: Number of breakpoints already used by previous stages

        Returns:
            Number of breakpoints used for system caching
        """
        system_messages = request.get('system', [])
        if not system_messages or used_breakpoints >= 4:
            return 0

        # Cache only the last system message
        system_messages[-1]['cache_control'] = {'type': 'ephemeral'}
        self.logger.debug('Added cache breakpoint to last system message')
        return 1

    def _insert_messages_cache_bp(self, request: dict[str, Any], used_breakpoints: int):
        """Add simple, effective cache breakpoints to message history.

        Args:
            used_breakpoints: Number of breakpoints already used by previous stages
        """
        messages = request.get('messages', [])
        if not messages:
            return

        available_breakpoints = 4 - used_breakpoints
        if available_breakpoints <= 0:
            self.logger.debug('No available breakpoints for message caching')
            return

        breakpoints_used = 0

        # Priority 1: Cache last user message (especially valuable for images)
        for i in reversed(range(len(messages))):
            if messages[i].get('role') == 'user':
                if self._add_cache_breakpoint_to_message_content(messages[i]):
                    breakpoints_used += 1
                    self.logger.debug(f'Added cache breakpoint to last user message at index {i}')
                break

        # Priority 2: Add breakpoints every 20 content blocks working backwards
        if breakpoints_used < available_breakpoints:
            content_count = 0
            for i in reversed(range(len(messages))):
                if breakpoints_used >= available_breakpoints:
                    break
                if messages[i].get('cache_control'):  # Skip already cached messages
                    continue

                # Count content blocks in this message
                content = messages[i].get('content', [])
                if isinstance(content, str):
                    content_count += 1
                elif isinstance(content, list):
                    # Count non-thinking content blocks
                    content_count += len([c for c in content if c.get('type') != 'thinking'])

                # Add breakpoint every 20 content blocks
                if content_count >= 20:
                    if self._add_cache_breakpoint_to_message_content(messages[i]):
                        breakpoints_used += 1
                        content_count = 0  # Reset counter
                        self.logger.debug(f'Added cache breakpoint at content block milestone (message {i})')

        self.logger.debug(f'Applied {breakpoints_used} message cache breakpoints')

    def _add_cache_breakpoint_to_message_content(self, message: dict) -> bool:
        """Add cache control to the last content block in a message."""
        content = message.get('content', [])
        if isinstance(content, list) and content:
            # Don't cache thinking blocks
            non_thinking_blocks = [c for c in content if c.get('type') != 'thinking']
            if non_thinking_blocks:
                # Add cache control to last non-thinking content block
                for block in reversed(content):
                    if block.get('type') != 'thinking':
                        block['cache_control'] = {'type': 'ephemeral'}
                        return True
        elif isinstance(content, str):
            # Convert string to list format and add cache control
            message['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
            return True

        return False

    def _validate_breakpoint_count(self, request: dict) -> int:
        """Validate and count total cache breakpoints in request."""
        total_breakpoints = 0

        # Count system breakpoints
        for system_msg in request.get('system', []):
            if 'cache_control' in system_msg:
                total_breakpoints += 1

        # Count tools breakpoints
        for tool in request.get('tools', []):
            if 'cache_control' in tool:
                total_breakpoints += 1

        # Count message content breakpoints
        for message in request.get('messages', []):
            content = message.get('content', [])
            if isinstance(content, list):
                for block in content:
                    if 'cache_control' in block:
                        total_breakpoints += 1

        if total_breakpoints > 4:
            self.logger.warning(f'Cache breakpoint count ({total_breakpoints}) exceeds limit of 4')

        return total_breakpoints
