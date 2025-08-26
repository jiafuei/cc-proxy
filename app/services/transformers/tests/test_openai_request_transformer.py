"""Tests for OpenAIRequestTransformer - comprehensive field mapping tests."""

from unittest.mock import Mock

import pytest

from app.services.transformers.openai import OpenAIRequestTransformer


class TestOpenAIRequestTransformer:
    """Test cases for the enhanced OpenAI request transformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        return OpenAIRequestTransformer(api_key='test-key')

    @pytest.fixture
    def provider_config(self):
        """Mock provider config."""
        config = Mock()
        config.api_key = 'config-key'
        return config

    @pytest.fixture
    def basic_claude_request(self):
        """Basic Claude request for testing."""
        return {
            'model': 'claude-sonnet-4-20250514',
            'messages': [
                {'role': 'user', 'content': 'Hello'}
            ],
            'max_tokens': 100,
            'temperature': 0.7,
            'stream': True
        }

    @pytest.fixture
    def full_claude_request(self):
        """Full Claude request with all fields for testing."""
        return {
            'model': 'claude-3-5-sonnet-20241022',
            'system': [
                {'type': 'text', 'text': 'You are a helpful assistant.'},
                {'type': 'text', 'text': 'Be concise and accurate.'}
            ],
            'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': [
                    {'type': 'text', 'text': 'Hi! Let me help you.'},
                    {'type': 'tool_use', 'id': 'tool1', 'name': 'search', 'input': {'query': 'help'}}
                ]},
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'tool1', 'content': 'Found results'}
                ]}
            ],
            'tools': [
                {
                    'name': 'search',
                    'description': 'Search for information',
                    'input_schema': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string'}
                        }
                    }
                }
            ],
            'tool_choice': {'type': 'tool', 'name': 'search'},
            'stop_sequences': ['STOP', 'END'],
            'max_tokens': 500,
            'temperature': 0.8,
            'stream': False
        }

    @pytest.mark.asyncio
    async def test_basic_transformation(self, transformer, provider_config, basic_claude_request):
        """Test basic transformation with minimal fields."""
        params = {
            'request': basic_claude_request,
            'provider_config': provider_config
        }
        
        result_request, result_headers = await transformer.transform(params)
        
        assert result_request['model'] == 'gpt-4o'  # claude-sonnet-4 -> gpt-4o
        assert result_request['messages'] == [{'role': 'user', 'content': 'Hello'}]
        assert result_request['max_tokens'] == 100
        assert result_request['temperature'] == 0.7
        assert result_request['stream'] is True
        
        assert result_headers['authorization'] == 'Bearer test-key'
        assert result_headers['content-type'] == 'application/json'

    @pytest.mark.asyncio
    async def test_full_transformation(self, transformer, provider_config, full_claude_request):
        """Test full transformation with all fields."""
        params = {
            'request': full_claude_request,
            'provider_config': provider_config
        }
        
        result_request, result_headers = await transformer.transform(params)
        
        # Check model mapping
        assert result_request['model'] == 'gpt-4'  # claude-3-5-sonnet -> gpt-4
        
        # Check system message conversion
        messages = result_request['messages']
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == 'You are a helpful assistant.\nBe concise and accurate.'
        
        # Check regular message conversion
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == 'Hello'
        
        assert messages[2]['role'] == 'assistant'
        assert '[Tool: search with input: {"query": "help"}]' in messages[2]['content']
        
        assert messages[3]['role'] == 'user'
        assert '[Tool Result for tool1: Found results]' in messages[3]['content']
        
        # Check tools conversion
        assert 'tools' in result_request
        tools = result_request['tools']
        assert len(tools) == 1
        assert tools[0]['type'] == 'function'
        assert tools[0]['function']['name'] == 'search'
        assert tools[0]['function']['description'] == 'Search for information'
        assert tools[0]['function']['parameters']['type'] == 'object'
        
        # Check tool_choice conversion
        assert result_request['tool_choice']['type'] == 'function'
        assert result_request['tool_choice']['function']['name'] == 'search'
        
        # Check stop sequences
        assert result_request['stop'] == ['STOP', 'END']

    def test_system_conversion_string(self, transformer):
        """Test system message conversion from string."""
        system_str = "You are a helpful assistant."
        result = transformer._convert_system_to_content(system_str)
        assert result == "You are a helpful assistant."

    def test_system_conversion_array(self, transformer):
        """Test system message conversion from array."""
        system_array = [
            {'type': 'text', 'text': 'First instruction'},
            {'type': 'text', 'text': 'Second instruction'}
        ]
        result = transformer._convert_system_to_content(system_array)
        assert result == "First instruction\nSecond instruction"

    def test_system_conversion_empty(self, transformer):
        """Test system message conversion with empty/None values."""
        assert transformer._convert_system_to_content(None) == ''
        assert transformer._convert_system_to_content('') == ''
        assert transformer._convert_system_to_content([]) == ''

    def test_content_blocks_conversion(self, transformer):
        """Test content blocks conversion."""
        content_blocks = [
            {'type': 'text', 'text': 'Hello'},
            {'type': 'tool_use', 'id': '1', 'name': 'search', 'input': {'q': 'test'}},
            {'type': 'tool_result', 'tool_use_id': '1', 'content': 'result data'},
            {'type': 'thinking', 'thinking': 'some thoughts'}  # Should be skipped
        ]
        
        result = transformer._convert_content_blocks(content_blocks)
        lines = result.split('\n')
        
        assert 'Hello' in lines
        assert '[Tool: search with input: {"q": "test"}]' in lines
        assert '[Tool Result for 1: result data]' in lines
        assert 'some thoughts' not in result  # Thinking blocks are skipped

    def test_tools_conversion(self, transformer):
        """Test tools conversion from Claude to OpenAI format."""
        claude_tools = [
            {
                'name': 'web_search',
                'description': 'Search the web',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string', 'description': 'Search query'}
                    },
                    'required': ['query']
                }
            }
        ]
        
        result = transformer._convert_tools(claude_tools)
        
        assert len(result) == 1
        tool = result[0]
        assert tool['type'] == 'function'
        assert tool['function']['name'] == 'web_search'
        assert tool['function']['description'] == 'Search the web'
        assert tool['function']['parameters']['properties']['query']['type'] == 'string'

    def test_tools_conversion_empty(self, transformer):
        """Test tools conversion with empty input."""
        assert transformer._convert_tools(None) is None
        assert transformer._convert_tools([]) is None

    def test_tool_choice_conversions(self, transformer):
        """Test various tool_choice conversions."""
        # String values
        assert transformer._convert_tool_choice('auto') == 'auto'
        assert transformer._convert_tool_choice('any') == 'required'
        assert transformer._convert_tool_choice('none') == 'none'
        
        # Specific tool choice
        specific_choice = {'type': 'tool', 'name': 'search_tool'}
        result = transformer._convert_tool_choice(specific_choice)
        assert result['type'] == 'function'
        assert result['function']['name'] == 'search_tool'
        
        # Any tool choice
        any_choice = {'type': 'any'}
        assert transformer._convert_tool_choice(any_choice) == 'required'
        
        # Empty/None
        assert transformer._convert_tool_choice(None) is None
        assert transformer._convert_tool_choice('') is None

    def test_stop_sequences_conversion(self, transformer):
        """Test stop sequences conversion."""
        # Normal case
        stop_seqs = ['STOP', 'END', 'TERMINATE']
        result = transformer._convert_stop_sequences(stop_seqs)
        assert result == ['STOP', 'END', 'TERMINATE']
        
        # Too many sequences (should truncate to 4)
        many_stops = ['A', 'B', 'C', 'D', 'E', 'F']
        result = transformer._convert_stop_sequences(many_stops)
        assert result == ['A', 'B', 'C', 'D']
        assert len(result) == 4
        
        # Empty/None
        assert transformer._convert_stop_sequences(None) is None
        assert transformer._convert_stop_sequences([]) is None

    def test_model_name_mappings(self, transformer):
        """Test all model name mappings."""
        test_cases = [
            ('claude-sonnet-4-20250514', 'gpt-4o'),
            ('claude-3-5-sonnet-20250121', 'gpt-4o'),
            ('claude-3-5-sonnet-20241022', 'gpt-4'),
            ('claude-3-sonnet-20240229', 'gpt-4'),
            ('claude-3-haiku-20240307', 'gpt-3.5-turbo'),
            ('claude-3-opus-20240229', 'gpt-4-turbo'),
            ('claude-3-5-haiku-20241022', 'gpt-3.5-turbo'),
            ('unknown-model', 'gpt-4'),  # Default fallback
        ]
        
        for claude_model, expected_openai in test_cases:
            result = transformer._convert_model_name(claude_model)
            assert result == expected_openai, f"Failed for {claude_model}"

    @pytest.mark.asyncio
    async def test_none_field_removal(self, transformer, provider_config):
        """Test that None values are properly removed from request."""
        claude_request = {
            'model': 'claude-sonnet-4-20250514',
            'messages': [{'role': 'user', 'content': 'Hello'}],
            'max_tokens': None,  # This should be removed
            'temperature': 0.7,
            'tools': None,  # This should be removed
            'stream': False
        }
        
        params = {
            'request': claude_request,
            'provider_config': provider_config
        }
        
        result_request, _ = await transformer.transform(params)
        
        # None values should not be present
        assert 'max_tokens' not in result_request
        assert 'tools' not in result_request
        
        # Non-None values should be present
        assert 'temperature' in result_request
        assert 'stream' in result_request

    @pytest.mark.asyncio 
    async def test_api_key_precedence(self, provider_config):
        """Test that transformer api_key takes precedence over config."""
        transformer_with_key = OpenAIRequestTransformer(api_key='transformer-key')
        claude_request = {
            'model': 'claude-sonnet-4-20250514',
            'messages': [{'role': 'user', 'content': 'Hello'}]
        }
        
        params = {
            'request': claude_request,
            'provider_config': provider_config  # Has 'config-key'
        }
        
        _, headers = await transformer_with_key.transform(params)
        assert headers['authorization'] == 'Bearer transformer-key'
        
        # Test fallback to config key
        transformer_no_key = OpenAIRequestTransformer()
        _, headers = await transformer_no_key.transform(params)
        assert headers['authorization'] == 'Bearer config-key'