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
                {'name': 'Read', 'description': 'Read files'},
                {'name': 'Write', 'description': 'Write files'},
                {'name': 'mcp__ide__getDiagnostics', 'description': 'Get diagnostics'},
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

    def test_system_cache_breakpoint_single_message(self, transformer):
        """Test caching with single system message."""
        request = {'system': [{'type': 'text', 'text': 'Single system message'}]}

        breakpoints_used = transformer._insert_system_cache_bp(request, 0)

        assert 'cache_control' in request['system'][0]
        assert request['system'][0]['cache_control'] == {'type': 'ephemeral'}
        assert breakpoints_used == 1

    def test_system_cache_breakpoint_empty_system(self, transformer):
        """Test handling of empty system array."""
        request = {'system': []}
        breakpoints_used = transformer._insert_system_cache_bp(request, 0)
        assert breakpoints_used == 0
        # Should not raise exception

        request = {}
        breakpoints_used = transformer._insert_system_cache_bp(request, 0)
        assert breakpoints_used == 0
        # Should not raise exception

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

    def test_tools_caching_strategy(self, transformer, sample_request):
        """Test tools caching with sufficient default tools."""
        # Add more default tools to trigger caching
        sample_request['tools'].extend([{'name': f'Tool{i}', 'description': f'Tool {i}'} for i in range(3, 10)])

        breakpoints_used = transformer._reorder_and_cache_tools_array(sample_request)

        tools = sample_request['tools']
        # With <20 tools, should use simple 1-breakpoint strategy
        assert breakpoints_used == 1
        assert 'cache_control' in tools[-1]  # Last tool should be cached

    def test_tools_caching_large_array(self, transformer, sample_request):
        """Test tools caching with large tool array (>=20 tools)."""
        # Add many tools to trigger 2-breakpoint strategy
        sample_request['tools'].extend([{'name': f'Tool{i}', 'description': f'Tool {i}'} for i in range(3, 25)])

        breakpoints_used = transformer._reorder_and_cache_tools_array(sample_request)

        tools = sample_request['tools']
        default_tools = [t for t in tools if not t['name'].startswith('mcp__')]
        mcp_tools = [t for t in tools if t['name'].startswith('mcp__')]

        # Should use 2-breakpoint strategy for large arrays
        assert breakpoints_used == 2
        assert 'cache_control' in default_tools[-1]
        if mcp_tools:
            assert 'cache_control' in mcp_tools[-1]

    def test_tool_cluster_identification(self, transformer):
        """Test identification of tool use/result clusters."""
        messages = [
            {'role': 'user', 'content': 'Start task'},
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'name': 'Read', 'id': '1'}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': '1', 'content': 'result'}]},
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'name': 'Write', 'id': '2'}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': '2', 'content': 'result'}]},
            {'role': 'user', 'content': 'Regular message'},
            {'role': 'assistant', 'content': 'Regular response'},
        ]

        clusters = transformer._identify_tool_clusters(messages)

        # Should identify one cluster of indices [1, 2, 3, 4] (tool interactions)
        assert len(clusters) == 1
        assert clusters[0] == [1, 2, 3, 4]

    def test_conversation_milestone_detection(self, transformer):
        """Test detection of workflow milestones."""
        messages = [
            {'role': 'user', 'content': 'Start'},
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'name': 'TodoWrite', 'id': '1', 'input': {'todos': []}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': '1', 'content': 'success'}]},
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'name': 'MultiEdit', 'id': '2', 'input': {'file': 'test.py'}}]},
        ]

        milestones = transformer._find_conversation_milestones(messages)

        # Should detect TodoWrite and MultiEdit as milestones
        assert 1 in milestones  # TodoWrite
        assert 3 in milestones  # MultiEdit

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

    def test_content_block_counting(self, transformer):
        """Test content block counting for cache breakpoint placement."""
        messages = []
        # Create messages with enough content blocks to trigger breakpoint
        for i in range(12):  # 24 non-thinking blocks total (12 * 2)
            messages.append(
                {
                    'role': 'user' if i % 2 == 0 else 'assistant',
                    'content': [
                        {'type': 'text', 'text': f'Message {i}-1'},
                        {'type': 'text', 'text': f'Message {i}-2'},
                        {'type': 'thinking', 'thinking': f'Thinking {i}'},  # Should be excluded
                    ],
                }
            )

        # Simulate no system/tools breakpoints so messages can get breakpoints
        transformer._add_content_block_breakpoints(messages, 2)

        # Should add breakpoints at content milestones (every 20 blocks)
        cached_messages = [msg for msg in messages if any('cache_control' in block for block in msg.get('content', []) if isinstance(block, dict))]
        # With 24 blocks, should hit 20-block milestone once
        assert len(cached_messages) >= 1

    def test_breakpoint_count_validation(self, transformer, sample_request):
        """Test validation of total breakpoint count."""
        # Add cache controls to exceed limit
        sample_request['system'][0]['cache_control'] = {'type': 'ephemeral'}
        sample_request['system'][1]['cache_control'] = {'type': 'ephemeral'}
        sample_request['tools'][0]['cache_control'] = {'type': 'ephemeral'}
        sample_request['tools'][1]['cache_control'] = {'type': 'ephemeral'}
        sample_request['tools'][2]['cache_control'] = {'type': 'ephemeral'}  # This would be 5th

        count = transformer._validate_breakpoint_count(sample_request)
        assert count == 5
        # Should log warning about exceeding limit

    @pytest.mark.asyncio
    async def test_full_transform_integration(self, transformer, sample_request):
        """Test complete transform method integration."""
        params = {'request': sample_request, 'headers': {}, 'routing_key': 'default'}

        result_request, result_headers = await transformer.transform(params)

        # Should have applied caching strategy
        total_breakpoints = transformer._validate_breakpoint_count(result_request)
        assert total_breakpoints <= 4
        assert total_breakpoints > 0

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

    def test_thinking_block_exclusion(self, transformer):
        """Test that thinking blocks are properly excluded from caching."""
        request = {'messages': [{'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'Some thoughts...'}, {'type': 'text', 'text': 'Response'}]}]}

        transformer._insert_messages_cache_bp(request, 0)  # 0 used breakpoints

        # Thinking block should not have cache control
        thinking_block = request['messages'][0]['content'][0]
        assert 'cache_control' not in thinking_block
        assert thinking_block['type'] == 'thinking'
