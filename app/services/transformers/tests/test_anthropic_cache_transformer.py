"""Tests for AnthropicCacheTransformer."""

from unittest.mock import Mock

import pytest

from app.services.transformers.anthropic import AnthropicCacheTransformer


class TestAnthropicCacheTransformer:
    """Test cases for the Anthropic cache transformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance with mock logger."""
        mock_logger = Mock()
        return AnthropicCacheTransformer(mock_logger)

    @pytest.fixture
    def sample_request(self):
        """Sample request with system, tools, and messages."""
        return {
            'model': 'claude-sonnet-4-20250514',
            'system': [{'type': 'text', 'text': 'You are Claude Code'}, {'type': 'text', 'text': 'Large system context with project instructions...'}],
            'tools': [
                {'name': 'Read', 'description': 'Read files', 'input_schema': {'type': 'object', 'properties': {}}},
                {'name': 'Write', 'description': 'Write files', 'input_schema': {'type': 'object', 'properties': {}}},
                {'name': 'mcp__ide__getDiagnostics', 'description': 'Get diagnostics', 'input_schema': {'type': 'object', 'properties': {}}},
            ],
            'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hi there!'}, {'type': 'tool_use', 'name': 'Read', 'id': '1', 'input': {'file': 'test.py'}}]},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': '1', 'content': 'file content'}]},
            ],
        }

    def test_system_cache_breakpoint_only_last_message(self, transformer, sample_request):
        """Test that only the last system message gets cached."""
        breakpoints_used = transformer._insert_system_cache_bp(sample_request, 0)

        # First system message should not have cache control
        assert 'cache_control' not in sample_request['system'][0]

        # Last system message should have cache control
        assert 'cache_control' in sample_request['system'][1]
        assert sample_request['system'][1]['cache_control'] == {'type': 'ephemeral'}

        # Should return 1 breakpoint used
        assert breakpoints_used == 1

    def test_tools_reordering(self, transformer, sample_request):
        """Test that tools are reordered correctly (default first, MCP second)."""
        breakpoints_used = transformer._reorder_and_cache_tools_array(sample_request)

        tools = sample_request['tools']
        default_tools = [t for t in tools if not t['name'].startswith('mcp__')]
        mcp_tools = [t for t in tools if t['name'].startswith('mcp__')]

        # Check that default tools come first
        assert tools[: len(default_tools)] == default_tools
        assert tools[len(default_tools) :] == mcp_tools

        # Should return number of breakpoints used
        assert isinstance(breakpoints_used, int)
        assert breakpoints_used >= 0

    def test_tools_caching_large_array(self, sample_request):
        """Test tools caching with large tool array (>=20 tools)."""
        # Create transformer with max_tools_breakpoints=2 for large arrays
        mock_logger = Mock()
        transformer = AnthropicCacheTransformer(mock_logger, max_tools_breakpoints=2)

        # Add many tools to trigger 2-breakpoint strategy
        sample_request['tools'].extend([{'name': f'Tool{i}', 'description': f'Tool {i}'} for i in range(3, 25)])

        breakpoints_used = transformer._reorder_and_cache_tools_array(sample_request)

        tools = sample_request['tools']

        # Should use 2-breakpoint strategy for large arrays (every 20 tools)
        assert breakpoints_used == 2
        # With 25 tools, should cache at indices 19 and 24 (every 20 tools)
        assert 'cache_control' in tools[19]  # Tool at index 19 (20th tool)
        assert 'cache_control' in tools[24]  # Tool at index 24 (25th tool)
        assert tools[19]['cache_control'] == {'type': 'ephemeral', 'ttl': '1h'}
        assert tools[24]['cache_control'] == {'type': 'ephemeral', 'ttl': '1h'}

    def test_message_cache_breakpoint_addition(self, transformer):
        """Test adding cache breakpoints to message content."""
        # Test with list content
        message_list = {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': 'Response'}, {'type': 'thinking', 'thinking': 'Some thinking'}, {'type': 'text', 'text': 'More text'}],
        }

        result = transformer._add_cache_breakpoint_to_message_content(message_list)
        assert result is True

        # Should cache last non-thinking block
        assert 'cache_control' not in message_list['content'][1]  # thinking block
        assert 'cache_control' in message_list['content'][2]  # last text block

        # Test with string content
        message_string = {'role': 'user', 'content': 'Simple message'}
        result = transformer._add_cache_breakpoint_to_message_content(message_string)
        assert result is True
        assert isinstance(message_string['content'], list)
        assert message_string['content'][0]['cache_control'] == {'type': 'ephemeral'}

    @pytest.mark.asyncio
    async def test_builtin_tool_bypass(self, transformer):
        """Test that built-in tool requests bypass caching."""
        builtin_request = {
            'model': 'claude-sonnet-4-20250514',
            'system': [{'type': 'text', 'text': 'System message'}],
            'tools': [{'type': 'web_search', 'name': 'WebSearch'}, {'type': 'web_fetch', 'name': 'WebFetch'}],
            'messages': [{'role': 'user', 'content': 'Search the web'}],
        }

        params = {'request': builtin_request, 'headers': {}, 'routing_key': 'default'}
        result_request, result_headers = await transformer.transform(params)

        # Should be unchanged (no caching applied)
        assert result_request == builtin_request
        assert transformer._validate_breakpoint_count(result_request) == 0

    @pytest.mark.asyncio
    async def test_background_routing_key_bypass(self, transformer, sample_request):
        """Test that background messages bypass caching."""
        params = {'request': sample_request, 'headers': {}, 'routing_key': 'background'}

        result_request, result_headers = await transformer.transform(params)

        # Should be unchanged
        assert result_request == sample_request
        assert result_headers == {}

    @pytest.mark.asyncio
    async def test_empty_structures(self, transformer):
        """Test handling of empty or missing structures."""
        empty_request = {'model': 'claude-sonnet-4-20250514', 'messages': []}

        params = {'request': empty_request, 'headers': {}, 'routing_key': 'default'}

        # Should not raise exceptions
        result_request, result_headers = await transformer.transform(params)
        assert result_request is not None

        # Test empty system handling
        request = {'system': []}
        breakpoints_used = transformer._insert_system_cache_bp(request, 0)
        assert breakpoints_used == 0

        request = {}
        breakpoints_used = transformer._insert_system_cache_bp(request, 0)
        assert breakpoints_used == 0

    def test_thinking_block_exclusion(self, transformer):
        """Test that thinking blocks are properly excluded from caching."""
        request = {'messages': [{'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'Some thoughts...'}, {'type': 'text', 'text': 'Response'}]}]}

        transformer._insert_messages_cache_bp(request, 0)  # 0 used breakpoints

        # Thinking block should not have cache control
        thinking_block = request['messages'][0]['content'][0]
        assert 'cache_control' not in thinking_block
        assert thinking_block['type'] == 'thinking'
