"""Comprehensive tests for built-in tools transformers."""

from unittest.mock import Mock, patch

import pytest

from app.services.transformers.builtin_tools import (
    AnthropicBuiltinToolsTransformer,
    BuiltinToolsRequestTransformer,
    OpenAIBuiltinToolsResponseTransformer,
    OpenAIBuiltinToolsTransformer,
    SmartBuiltinToolsTransformer,
)


class TestBuiltinToolsRequestTransformer:
    """Test cases for the base built-in tools transformer."""

    @pytest.fixture
    def transformer(self):
        """Create mock transformer instance."""
        mock_logger = Mock()

        class MockTransformer(BuiltinToolsRequestTransformer):
            async def transform(self, params):
                return params['request'], params['headers']

        return MockTransformer(mock_logger)

    def test_is_builtin_tool_detection(self, transformer):
        """Test detection of built-in tools by type field without input_schema."""
        # Built-in tool (has type, no input_schema)
        builtin_tool = {'type': 'web_search_20250305', 'name': 'web_search'}
        assert transformer._is_builtin_tool(builtin_tool) is True

        # Regular tool (has input_schema)
        regular_tool = {'name': 'Read', 'input_schema': {'type': 'object'}}
        assert transformer._is_builtin_tool(regular_tool) is False

        # Invalid tool (not a dict)
        invalid_tool = 'not_a_dict'
        assert transformer._is_builtin_tool(invalid_tool) is False

        # Missing type
        no_type_tool = {'name': 'Something'}
        assert transformer._is_builtin_tool(no_type_tool) is False

    def test_is_websearch_tool_detection(self, transformer):
        """Test specific WebSearch tool detection."""
        # Valid WebSearch tool
        websearch_tool = {'type': 'web_search_20250305', 'name': 'web_search', 'allowed_domains': ['example.com']}
        assert transformer._is_websearch_tool(websearch_tool) is True

        # Built-in tool but not WebSearch
        other_builtin = {'type': 'web_fetch_20250305', 'name': 'web_fetch'}
        assert transformer._is_websearch_tool(other_builtin) is False

        # Regular tool with web_search name
        regular_tool = {'name': 'web_search', 'input_schema': {'type': 'object'}}
        assert transformer._is_websearch_tool(regular_tool) is False

    def test_detect_builtin_tools(self, transformer):
        """Test detection of all built-in tools in request."""
        request = {
            'tools': [
                {'type': 'web_search_20250305', 'name': 'web_search'},
                {'name': 'Read', 'input_schema': {'type': 'object'}},
                {'type': 'web_fetch_20250305', 'name': 'web_fetch'},
                {'name': 'Write', 'input_schema': {'type': 'object'}},
            ]
        }

        builtin_tools = transformer._detect_builtin_tools(request)
        assert len(builtin_tools) == 2
        assert all(transformer._is_builtin_tool(tool) for tool in builtin_tools)

        # Test empty request
        empty_request = {}
        assert transformer._detect_builtin_tools(empty_request) == []

    def test_separate_tools(self, transformer):
        """Test separation of built-in and regular tools."""
        tools = [{'type': 'web_search_20250305', 'name': 'web_search'}, {'name': 'Read', 'input_schema': {'type': 'object'}}, {'type': 'web_fetch_20250305', 'name': 'web_fetch'}]

        builtin_tools, regular_tools = transformer._separate_tools(tools)

        assert len(builtin_tools) == 2
        assert len(regular_tools) == 1
        assert builtin_tools[0]['name'] == 'web_search'
        assert regular_tools[0]['name'] == 'Read'


class TestOpenAIBuiltinToolsTransformer:
    """Test cases for OpenAI built-in tools transformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance with mock logger."""
        mock_logger = Mock()
        return OpenAIBuiltinToolsTransformer(mock_logger)

    @pytest.fixture
    def sample_params(self):
        """Sample transformation parameters."""
        return {
            'request': {
                'model': 'gpt-4o',
                'messages': [{'role': 'user', 'content': 'Search for Python tutorials'}],
                'tools': [
                    {
                        'type': 'web_search_20250305',
                        'name': 'web_search',
                        'allowed_domains': ['python.org', 'docs.python.org'],
                        'user_location': {'country': 'US', 'city': 'San Francisco', 'region': 'California'},
                    },
                    {'name': 'Read', 'description': 'Read files', 'input_schema': {'type': 'object', 'properties': {'file_path': {'type': 'string'}}}},
                ],
            },
            'headers': {'authorization': 'Bearer test-key'},
            'provider_config': Mock(base_url='https://api.openai.com/v1/chat/completions'),
        }

    @pytest.mark.asyncio
    async def test_transform_websearch_to_openai_format(self, transformer, sample_params):
        """Test conversion of WebSearch tool to OpenAI web_search_options."""
        request, headers = await transformer.transform(sample_params)

        # Check web_search_options was added
        assert 'web_search_options' in request
        web_search_opts = request['web_search_options']

        # Check domain filters
        assert 'filters' in web_search_opts
        assert web_search_opts['filters']['allowed_domains'] == ['python.org', 'docs.python.org']

        # Check user location conversion
        assert 'user_location' in web_search_opts
        user_loc = web_search_opts['user_location']
        assert user_loc['type'] == 'approximate'
        assert user_loc['approximate']['country'] == 'US'
        assert user_loc['approximate']['city'] == 'San Francisco'

        # Check search context size default
        assert web_search_opts['search_context_size'] == 'medium'

        # Check model conversion
        assert request['model'] == 'gpt-4o-search-preview'

        # Check tools array (should only have regular tools)
        assert len(request['tools']) == 1
        assert request['tools'][0]['name'] == 'Read'

    @pytest.mark.asyncio
    async def test_transform_no_builtin_tools(self, transformer):
        """Test transformation with no built-in tools."""
        params = {'request': {'model': 'gpt-4o', 'tools': [{'name': 'Read', 'input_schema': {'type': 'object'}}]}, 'headers': {}}

        request, headers = await transformer.transform(params)

        # Should be unchanged
        assert 'web_search_options' not in request
        assert request['model'] == 'gpt-4o'
        assert len(request['tools']) == 1

    @pytest.mark.asyncio
    async def test_transform_only_builtin_tools(self, transformer):
        """Test transformation with only built-in tools."""
        params = {'request': {'model': 'gpt-4o', 'tools': [{'type': 'web_search_20250305', 'name': 'web_search'}]}, 'headers': {}}

        request, headers = await transformer.transform(params)

        # web_search_options should be added
        assert 'web_search_options' in request
        # tools should be removed (no regular tools)
        assert 'tools' not in request

    def test_extract_websearch_config_full_params(self, transformer):
        """Test WebSearch parameter extraction with all parameters."""
        tool = {
            'type': 'web_search_20250305',
            'name': 'web_search',
            'allowed_domains': ['example.com', 'test.com'],
            'user_location': {'country': 'GB', 'city': 'London', 'region': 'England', 'timezone': 'Europe/London'},
        }

        config = transformer._extract_websearch_config(tool)

        assert config['filters']['allowed_domains'] == ['example.com', 'test.com']
        assert config['user_location']['type'] == 'approximate'
        assert config['user_location']['approximate']['country'] == 'GB'
        assert config['user_location']['approximate']['timezone'] == 'Europe/London'
        assert config['search_context_size'] == 'medium'

    def test_extract_websearch_config_minimal_params(self, transformer):
        """Test WebSearch parameter extraction with minimal parameters."""
        tool = {'type': 'web_search_20250305', 'name': 'web_search'}

        config = transformer._extract_websearch_config(tool)

        assert config['filters'] == {}
        assert config['search_context_size'] == 'medium'
        assert 'user_location' not in config

    def test_extract_websearch_config_blocked_domains(self, transformer):
        """Test WebSearch parameter extraction with blocked domains."""
        tool = {'type': 'web_search_20250305', 'name': 'web_search', 'blocked_domains': ['spam.com', 'ads.com']}

        config = transformer._extract_websearch_config(tool)
        assert config['filters']['blocked_domains'] == ['spam.com', 'ads.com']
        assert 'allowed_domains' not in config['filters']

    def test_validate_domain_filters_conflict(self, transformer):
        """Test validation of conflicting domain filters."""
        tool = {'type': 'web_search_20250305', 'name': 'web_search', 'allowed_domains': ['example.com'], 'blocked_domains': ['spam.com']}

        with pytest.raises(ValueError, match='Cannot use both allowed_domains and blocked_domains'):
            transformer._extract_websearch_config(tool)

    def test_ensure_search_model_conversion(self, transformer):
        """Test model conversion to search-preview variants."""
        assert transformer._ensure_search_model('gpt-4o') == 'gpt-4o-search-preview'
        assert transformer._ensure_search_model('gpt-4o-mini') == 'gpt-4o-mini-search-preview'

        # Already search model
        assert transformer._ensure_search_model('gpt-4o-search-preview') == 'gpt-4o-search-preview'

        # Unknown model (should pass through with warning)
        with patch.object(transformer, '_validate_model_compatibility') as mock_validate:
            result = transformer._ensure_search_model('unknown-model')
            assert result == 'unknown-model'
            mock_validate.assert_called_once_with('unknown-model')

    def test_convert_user_location(self, transformer):
        """Test user location format conversion."""
        anthropic_location = {'country': 'JP', 'city': 'Tokyo', 'region': 'Kanto', 'timezone': 'Asia/Tokyo'}

        openai_location = transformer._convert_user_location(anthropic_location)

        assert openai_location['type'] == 'approximate'
        approximate = openai_location['approximate']
        assert approximate['country'] == 'JP'
        assert approximate['city'] == 'Tokyo'
        assert approximate['region'] == 'Kanto'
        assert approximate['timezone'] == 'Asia/Tokyo'

    def test_convert_user_location_partial(self, transformer):
        """Test user location conversion with partial data."""
        anthropic_location = {'country': 'CA'}

        openai_location = transformer._convert_user_location(anthropic_location)

        assert openai_location['type'] == 'approximate'
        assert openai_location['approximate']['country'] == 'CA'
        assert 'city' not in openai_location['approximate']


class TestAnthropicBuiltinToolsTransformer:
    """Test cases for Anthropic built-in tools transformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance with mock logger."""
        mock_logger = Mock()
        return AnthropicBuiltinToolsTransformer(mock_logger)

    @pytest.mark.asyncio
    async def test_passthrough_transformation(self, transformer):
        """Test that Anthropic transformer passes through unchanged."""
        params = {
            'request': {
                'model': 'claude-sonnet-4-20250514',
                'tools': [{'type': 'web_search_20250305', 'name': 'web_search'}, {'name': 'Read', 'input_schema': {'type': 'object'}}],
            },
            'headers': {'x-api-key': 'test-key'},
        }

        original_request = params['request'].copy()
        original_headers = params['headers'].copy()

        request, headers = await transformer.transform(params)

        # Should be exactly the same
        assert request == original_request
        assert headers == original_headers


class TestSmartBuiltinToolsTransformer:
    """Test cases for smart auto-detecting transformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance with mock logger."""
        mock_logger = Mock()
        return SmartBuiltinToolsTransformer(mock_logger)

    def test_detect_anthropic_provider(self, transformer):
        """Test detection of Anthropic provider."""
        config = Mock(base_url='https://api.anthropic.com/v1/messages')
        assert transformer._detect_provider_type(config) == 'anthropic'

    def test_detect_openai_provider(self, transformer):
        """Test detection of OpenAI provider."""
        config = Mock(base_url='https://api.openai.com/v1/chat/completions')
        assert transformer._detect_provider_type(config) == 'openai'

    def test_detect_unknown_provider(self, transformer):
        """Test detection of unknown provider."""
        config = Mock(base_url='https://api.unknown.com/v1/chat')
        assert transformer._detect_provider_type(config) == 'unknown'

    @pytest.mark.asyncio
    async def test_auto_detect_openai_transformation(self, transformer):
        """Test auto-detection and OpenAI transformation."""
        params = {
            'request': {'model': 'gpt-4o', 'tools': [{'type': 'web_search_20250305', 'name': 'web_search'}]},
            'headers': {},
            'provider_config': Mock(base_url='https://api.openai.com/v1/chat/completions'),
        }

        with patch.object(transformer.transformers['openai'], 'transform') as mock_transform:
            mock_transform.return_value = (params['request'], params['headers'])

            await transformer.transform(params)
            mock_transform.assert_called_once_with(params)

    @pytest.mark.asyncio
    async def test_auto_detect_anthropic_passthrough(self, transformer):
        """Test auto-detection and Anthropic passthrough."""
        params = {
            'request': {'model': 'claude-sonnet-4-20250514', 'tools': [{'type': 'web_search_20250305', 'name': 'web_search'}]},
            'headers': {},
            'provider_config': Mock(base_url='https://api.anthropic.com/v1/messages'),
        }

        with patch.object(transformer.transformers['anthropic'], 'transform') as mock_transform:
            mock_transform.return_value = (params['request'], params['headers'])

            await transformer.transform(params)
            mock_transform.assert_called_once_with(params)

    @pytest.mark.asyncio
    async def test_unknown_provider_passthrough(self, transformer):
        """Test unknown provider defaults to passthrough."""
        params = {'request': {'model': 'unknown-model'}, 'headers': {}, 'provider_config': Mock(base_url='https://api.unknown.com/v1/chat')}

        original_request = params['request'].copy()
        request, headers = await transformer.transform(params)

        assert request == original_request

    @pytest.mark.asyncio
    async def test_missing_provider_config(self, transformer):
        """Test handling of missing provider config."""
        params = {'request': {'model': 'test-model'}, 'headers': {}}

        original_request = params['request'].copy()
        request, headers = await transformer.transform(params)

        assert request == original_request


class TestOpenAIBuiltinToolsResponseTransformer:
    """Test cases for OpenAI response transformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance with mock logger."""
        mock_logger = Mock()
        return OpenAIBuiltinToolsResponseTransformer(mock_logger)

    @pytest.mark.asyncio
    async def test_transform_response_with_annotations(self, transformer):
        """Test conversion of OpenAI annotations to Anthropic format."""
        params = {
            'request': {'web_search_options': {'filters': {}}},  # Indicates web search
            'response': {
                'choices': [{'message': {'content': 'Here are some Python tutorials from python.org.', 'role': 'assistant'}}],
                'annotations': [{'type': 'url_citation', 'url_citation': {'url': 'https://python.org/tutorial', 'title': 'Python Tutorial', 'start_index': 32, 'end_index': 44}}],
            },
        }

        response = await transformer.transform_response(params)

        # Check that annotation was converted
        message = response['choices'][0]['message']
        content = message['content']

        assert isinstance(content, list)
        assert len(content) == 2  # Original text + tool result

        # Check text block
        assert content[0]['type'] == 'text'
        assert content[0]['text'] == 'Here are some Python tutorials from python.org.'

        # Check tool result block
        tool_result = content[1]
        assert tool_result['type'] == 'web_search_tool_result'
        assert tool_result['content']['url'] == 'https://python.org/tutorial'
        assert tool_result['content']['title'] == 'Python Tutorial'
        assert tool_result['content']['snippet'] == 'rom python.o'

    @pytest.mark.asyncio
    async def test_transform_response_no_web_search(self, transformer):
        """Test response without web search is unchanged."""
        params = {
            'request': {'model': 'gpt-4o'},  # No web_search_options
            'response': {'choices': [{'message': {'content': 'Hello world'}}]},
        }

        original_response = params['response'].copy()
        response = await transformer.transform_response(params)

        assert response == original_response

    @pytest.mark.asyncio
    async def test_transform_response_no_annotations(self, transformer):
        """Test web search response without annotations."""
        params = {'request': {'web_search_options': {}}, 'response': {'choices': [{'message': {'content': 'No search results found.'}}]}}

        original_response = params['response'].copy()
        response = await transformer.transform_response(params)

        assert response == original_response

    def test_create_web_search_result(self, transformer):
        """Test creation of web search result block."""
        citation = {'url': 'https://example.com/page', 'title': 'Example Page', 'start_index': 0, 'end_index': 10}
        content = 'Example text content'

        result = transformer._create_web_search_result(citation, content)

        assert result['type'] == 'web_search_tool_result'
        assert result['content']['url'] == 'https://example.com/page'
        assert result['content']['title'] == 'Example Page'
        assert result['content']['snippet'] == 'Example te'
        assert 'id' in result

    def test_create_web_search_result_no_url(self, transformer):
        """Test creation fails without URL."""
        citation = {'title': 'No URL'}
        result = transformer._create_web_search_result(citation, '')
        assert result is None

    def test_extract_snippet(self, transformer):
        """Test snippet extraction from content."""
        content = 'This is a test content string'

        # Valid indices
        snippet = transformer._extract_snippet(content, 5, 9)
        assert snippet == 'is a'

        # Invalid indices
        snippet = transformer._extract_snippet(content, None, None)
        assert snippet == ''

        snippet = transformer._extract_snippet(content, 100, 200)
        assert snippet == ''

    @pytest.mark.asyncio
    async def test_transform_chunk_passthrough(self, transformer):
        """Test streaming chunk transformation (currently passthrough)."""
        params = {'chunk': b'test chunk data'}

        chunks = []
        async for chunk in transformer.transform_chunk(params):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == b'test chunk data'
