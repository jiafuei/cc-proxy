"""Anthropic transformers - pure passthrough implementations."""

import random
from typing import Any, AsyncIterator, Dict, Tuple

from app.services.transformers.interfaces import RequestTransformer, ResponseTransformer


class AnthropicHeadersTransformer(RequestTransformer):
    """Anthropic-specific header filtering transformer.

    Filters incoming headers to only include those with specific prefixes
    required by the Anthropic API.
    """

    def __init__(self, logger, auth_header: str):
        """Initialize transformer.

        Args:
            logger: Logger instance
            auth_header: Authentication header to use ('x-api-key' or 'authorization')
        """
        self.logger = logger
        self.auth_header = auth_header

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Filter headers to only include Anthropic-compatible prefixes."""
        request: dict[str, Any] = params['request']
        headers: dict[str, str] = params['headers']

        # Filter headers to only keep Anthropic-compatible ones
        filtered_headers = {
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

        # Inject API key from provider config if available
        provider_config = params.get('provider_config')
        if provider_config and provider_config.api_key:
            # Set the configured auth header
            if self.auth_header == 'x-api-key':
                filtered_headers[self.auth_header] = provider_config.api_key
                # Remove authorization header to avoid conflicts
                filtered_headers.pop('authorization', None)
            elif self.auth_header == 'authorization':
                filtered_headers[self.auth_header] = f'Bearer {provider_config.api_key}'
                # Remove x-api-key header to avoid conflicts
                filtered_headers.pop('x-api-key', None)

        return request, filtered_headers


class AnthropicResponseTransformer(ResponseTransformer):
    """Pure passthrough transformer for Anthropic responses."""

    def __init__(self, logger):
        """Initialize transformer."""
        self.logger = logger

    async def transform_chunk(self, params: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Pure passthrough - response is already in correct format."""
        yield params['chunk']

    async def transform_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Pure passthrough - response is already in correct format."""
        return params['response']


class AnthropicCacheTransformer(RequestTransformer):
    """Optimizes cache breakpoints given the limit of 4 cache breakpoints

    - Removes the cache breakpoint from the 'You are Claude' system message
    - Removes gitStatus suffix from system message to make it more cache-friendly
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
        total_tools = len(reordered_tools)
        breakpoints_used = 0

        # Strategy: Add cache breakpoints every 20 tools (original approach with bug fix)
        for i in range(0, total_tools, 20):
            if breakpoints_used >= self.max_tools_breakpoints:
                break

            # For each chunk, find the end position (either i+19 or the last tool if partial chunk)
            chunk_end = min(i + 20 - 1, total_tools - 1)
            reordered_tools[chunk_end]['cache_control'] = {'type': 'ephemeral', 'ttl': '1h'}
            breakpoints_used += 1

        # Fallback: if no breakpoints were added, add one at the end
        if breakpoints_used == 0 and reordered_tools:
            reordered_tools[-1]['cache_control'] = {'type': 'ephemeral', 'ttl': '1h'}
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


class ClaudeSystemMessageCleanerTransformer(RequestTransformer):
    """Transformer that cleans system messages by removing dynamic content.

    Removes git status information from system messages to make them more
    cache-friendly and reusable across sessions.
    """

    def __init__(self, logger):
        """Initialize transformer."""
        self.logger = logger

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Clean system messages by removing git status suffix."""
        request, headers = params['request'], params['headers']

        # Validate system key exists
        if 'system' not in request:
            return request, headers

        # Validate system array is not empty
        if not request['system']:
            return request, headers

        block = request['system'][-1]
        text = block['text']
        if not isinstance(text, str):
            return request, headers

        text = self._remove_system_git_status_suffix(text)
        text = self._remove_defensive_task_lines(text)

        request['system'][-1]['text'] = text

        return request, headers

    def _remove_system_git_status_suffix(self, text: str):
        """Remove git status suffix from system messages.

        Removes the entire chunk starting with '\ngitStatus: ' from the last
        system message to make it more cache-friendly and reusable.
        """

        # Remove the entire chunk of 'gitStatus: This is the git status at the start of the conversation....'
        # So the system message can be reused for the entire time Claude Code is open
        git_status_pos = text.rfind('\ngitStatus: ')
        if git_status_pos != -1:  # Only truncate if gitStatus is found
            text = text[:git_status_pos]
        return text

    def _remove_defensive_task_lines(self, text: str):
        remove_lines = ['IMPORTANT: Assist with defensive', 'You are powered', 'Assistant knowledge cutoff']
        return '\n'.join((line for line in text.splitlines() if not any(line.startswith(remove_line) for remove_line in remove_lines)))

class ClaudeSoftwareEngineeringSystemMessageTransformer(RequestTransformer):
    """Transformer that replaces the default system prompt for software engineering.

    Sub-agents(output style, agent creation etc..), background tasks should not be affected.
    """

    @classmethod
    def get_default_prompt(cls, env: str):
        return f"""You're an experienced, expert software engineer. Follow these guidelines:

**Response Style:**
- Be concise (1-3 sentences max) and direct
- Use GitHub-flavored markdown for formatting
- AVOID unnecessary explanations, conclusions, preambles, or emojis
- Minimize output tokens while maintaining accuracy

**Core Rules:**
- NEVER guess URLs, use only those supplied by user or Claude Code docs.
- Use WebFetch only when asked about Claude Code features
- Always use TodoWrite to plan and track tasks
- Check code conventions before making changes
- DON'T add comments unless requested

**Task Workflow:**
1. Plan with TodoWrite
2. Research codebase with search tools
3. Implement solution with tools
4. Verify with existing tests
5. Only run lint/typecheck commands if specified

**Tool Usage:**
- Only use provided tools.
- ALWAYS batch independent tool calls and combine commands using && whenever possible
- Prefer Task tool for file searches to save context
- Only commit when explicitly asked
- Process <system-reminder> tags as info, not user input

**Examples:**
- User: "2 + 2" → Assistant: "4"
- User: "what command lists files?" → Assistant: "ls"
- User: "is 11 prime?" → Assistant: "Yes"

{env}

**Remember:** always use **TodoWrite** to track progress, keep responses short, and only act when prompted.

    """
    def __init__(self, logger, prompt = ''):
        super().__init__(logger)
        self.prompt = prompt


    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        request, headers = params['request'], params['headers']

        # Validate system key exists
        if 'system' not in request:
            return request, headers

        # Validate system array is not empty
        if not request['system']:
            return request, headers

        block = request['system'][-1]
        text = block['text']
        if not isinstance(text, str):
            return request, headers

        if not self.is_software_eng_prompt(text):
            return request, headers

        env_text = self.extract_environment_text(text)
        if not self.prompt:
            request['system'][-1]['text'] = self.get_default_prompt(env_text) 
        else:
            request['system'][-1]['text'] = f"{self.prompt}\n{env_text}\n"

        return request, headers

    def is_software_eng_prompt(self, system_msg: str) -> bool:
        if not system_msg.strip().startswith('You are an interactive CLI tool that helps users with software engineering tasks.'):
            return False
        return True

    def extract_environment_text(self, system_msg: str) -> str:
        found = False
        lines = []
        for idx, line in enumerate(system_msg):
            if not found and not line.startswith('Here is useful information about the environment'):
                continue
            if not line.strip():
                return '\n'.join(lines)
            lines.append(line)

        return '\n'.join(lines)

    pass