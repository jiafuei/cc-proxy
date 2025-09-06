"""Tests for Gemini request transformer."""

from unittest.mock import MagicMock

import pytest

from app.services.transformers.gemini import GeminiRequestTransformer


class TestGeminiRequestTransformer:
    """Test cases for GeminiRequestTransformer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = MagicMock()
        self.transformer = GeminiRequestTransformer(self.logger)

    @pytest.mark.asyncio
    async def test_basic_text_message_conversion(self):
        """Test conversion of basic text messages."""
        anthropic_request = {
            'model': 'claude-3-5-sonnet-20241022',
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hello world'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hi there!'}]}],
            'temperature': 0.7,
            'max_tokens': 1000,
        }

        headers = {'authorization': 'Bearer test-api-key', 'content-type': 'application/json', 'accept': 'application/json', 'user-agent': 'test-client'}
        params = {'request': anthropic_request, 'headers': headers}

        gemini_request, filtered_headers = await self.transformer.transform(params)

        # Check message conversion
        assert 'contents' in gemini_request
        contents = gemini_request['contents']
        assert len(contents) == 2

        # Check user message
        assert contents[0]['role'] == 'user'
        assert contents[0]['parts'] == [{'text': 'Hello world'}]

        # Check assistant -> model role mapping
        assert contents[1]['role'] == 'model'
        assert contents[1]['parts'] == [{'text': 'Hi there!'}]

        # Check generation config - now includes default values
        assert 'generationConfig' in gemini_request
        gen_config = gemini_request['generationConfig']
        assert gen_config['temperature'] == 0.7
        assert gen_config['maxOutputTokens'] == 1000
        # Note: topP and topK are not set by default in current implementation
        assert gen_config['candidateCount'] == 1

        # Check header filtering - should NOT contain auth headers (uses query params)
        assert 'authorization' not in filtered_headers
        assert 'x-goog-api-key' not in filtered_headers
        assert 'user-agent' in filtered_headers
        assert 'accept' in filtered_headers
        assert 'content-type' in filtered_headers

    @pytest.mark.asyncio
    async def test_string_content_conversion(self):
        """Test conversion of string content to parts format."""
        anthropic_request = {'messages': [{'role': 'user', 'content': 'Simple string message'}]}

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        contents = gemini_request['contents']
        assert len(contents) == 1
        assert contents[0]['parts'] == [{'text': 'Simple string message'}]

    @pytest.mark.asyncio
    async def test_system_instruction_conversion(self):
        """Test conversion of system messages."""
        anthropic_request = {
            'system': [{'type': 'text', 'text': 'You are a helpful assistant.'}, {'type': 'text', 'text': 'Be concise and accurate.'}],
            'messages': [{'role': 'user', 'content': 'Test'}],
        }

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        assert 'system_instruction' in gemini_request
        system_instruction = gemini_request['system_instruction']
        assert system_instruction['parts'] == [{'text': 'You are a helpful assistant.\nBe concise and accurate.'}]

    @pytest.mark.asyncio
    async def test_string_system_conversion(self):
        """Test conversion of string system message."""
        anthropic_request = {'system': 'You are a helpful assistant.', 'messages': [{'role': 'user', 'content': 'Test'}]}

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        assert 'system_instruction' in gemini_request
        system_instruction = gemini_request['system_instruction']
        assert system_instruction['parts'] == [{'text': 'You are a helpful assistant.'}]

    @pytest.mark.asyncio
    async def test_image_block_conversion(self):
        """Test conversion of image blocks."""
        anthropic_request = {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Describe this image:'},
                        {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': 'iVBORw0KGgoAAAANSUhEUgA...'}},
                    ],
                }
            ]
        }

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        contents = gemini_request['contents']
        parts = contents[0]['parts']

        assert len(parts) == 2
        assert parts[0] == {'text': 'Describe this image:'}
        assert parts[1] == {'inline_data': {'mime_type': 'image/jpeg', 'data': 'iVBORw0KGgoAAAANSUhEUgA...'}}

    @pytest.mark.asyncio
    async def test_tools_conversion(self):
        """Test conversion of tool definitions."""
        anthropic_request = {
            'messages': [{'role': 'user', 'content': 'Use tools'}],
            'tools': [
                {
                    'name': 'get_weather',
                    'description': 'Get weather information',
                    'input_schema': {
                        'type': 'object',
                        'properties': {'city': {'type': 'string'}, 'units': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}},
                        'required': ['city'],
                    },
                }
            ],
        }

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        assert 'tools' in gemini_request
        tools = gemini_request['tools']
        assert len(tools) == 1

        function_declarations = tools[0]['functionDeclarations']
        assert len(function_declarations) == 1

        func_decl = function_declarations[0]
        assert func_decl['name'] == 'get_weather'
        assert func_decl['description'] == 'Get weather information'
        assert func_decl['parameters'] == anthropic_request['tools'][0]['input_schema']

    @pytest.mark.asyncio
    async def test_tool_use_conversion(self):
        """Test conversion of tool use blocks."""
        anthropic_request = {
            'messages': [
                {
                    'role': 'assistant',
                    'content': [
                        {'type': 'text', 'text': "I'll get the weather for you."},
                        {'type': 'tool_use', 'id': 'toolu_123', 'name': 'get_weather', 'input': {'city': 'New York', 'units': 'fahrenheit'}},
                    ],
                }
            ]
        }

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        contents = gemini_request['contents']
        parts = contents[0]['parts']

        assert len(parts) == 2
        assert parts[0] == {'text': "I'll get the weather for you."}

        function_call = parts[1]['functionCall']
        assert function_call['name'] == 'get_weather'
        assert function_call['args'] == {'city': 'New York', 'units': 'fahrenheit'}

    @pytest.mark.asyncio
    async def test_tool_result_conversion(self):
        """Test conversion of tool result blocks."""
        anthropic_request = {
            'messages': [{'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_123', 'content': 'The weather in New York is 72°F and sunny.'}]}]
        }

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        contents = gemini_request['contents']
        parts = contents[0]['parts']

        assert len(parts) == 1
        function_response = parts[0]['functionResponse']
        assert function_response['name'] == 'toolu_123'
        assert function_response['response']['content'] == 'The weather in New York is 72°F and sunny.'
        assert function_response['response']['success'] is True

    @pytest.mark.asyncio
    async def test_thinking_block_skipped(self):
        """Test that thinking blocks are skipped in conversion."""
        anthropic_request = {
            'messages': [{'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'Let me think about this...'}, {'type': 'text', 'text': 'Here is my response.'}]}]
        }

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        contents = gemini_request['contents']
        parts = contents[0]['parts']

        # Only text block should remain
        assert len(parts) == 1
        assert parts[0] == {'text': 'Here is my response.'}

    @pytest.mark.asyncio
    async def test_empty_messages_handling(self):
        """Test handling of messages with no valid content."""
        anthropic_request = {
            'messages': [
                {
                    'role': 'user',
                    'content': [],  # Empty content array
                },
                {
                    'role': 'assistant',
                    'content': [
                        {'type': 'thinking', 'thinking': 'Only thinking'}  # Only unsupported blocks
                    ],
                },
            ]
        }

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        # Messages with no valid parts should be filtered out
        assert 'contents' in gemini_request
        assert len(gemini_request['contents']) == 0

    @pytest.mark.asyncio
    async def test_header_filtering(self):
        """Test header filtering for Gemini API."""
        headers = {
            'authorization': 'Bearer sk-test123',
            'x-goog-api-key': 'goog-key-456',
            'user-agent': 'test-agent',
            'accept': 'application/json',
            'x-anthropic-beta': 'some-beta',  # Should be filtered out
            'custom-header': 'value',  # Should be filtered out
        }

        params = {'request': {}, 'headers': headers}
        _, filtered_headers = await self.transformer.transform(params)

        # Should NOT keep auth headers (Gemini uses query params)
        assert 'x-goog-api-key' not in filtered_headers
        assert 'authorization' not in filtered_headers

        # Should keep allowed headers
        assert filtered_headers['user-agent'] == 'test-agent'
        assert filtered_headers['accept'] == 'application/json'

        # Should filter out unwanted headers
        assert 'x-anthropic-beta' not in filtered_headers
        assert 'custom-header' not in filtered_headers

    @pytest.mark.asyncio
    async def test_authorization_to_api_key_conversion(self):
        """Test conversion of authorization header to API key."""
        headers = {'authorization': 'Bearer sk-test123'}

        params = {'request': {}, 'headers': headers}
        _, filtered_headers = await self.transformer.transform(params)

        assert 'authorization' not in filtered_headers
        assert 'x-goog-api-key' not in filtered_headers

    @pytest.mark.asyncio
    async def test_no_generation_config_when_empty(self):
        """Test that generationConfig is not included when no parameters are set."""
        anthropic_request = {'messages': [{'role': 'user', 'content': 'Test'}]}

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        # With new implementation, generationConfig is always included with defaults
        assert 'generationConfig' in gemini_request
        gen_config = gemini_request['generationConfig']
        assert gen_config['topP'] == 1.0
        assert gen_config['topK'] == 40
        assert gen_config['candidateCount'] == 1

    @pytest.mark.asyncio
    async def test_invalid_image_source_handling(self):
        """Test handling of invalid image source types."""
        anthropic_request = {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                'type': 'url',  # Unsupported type
                                'url': 'https://example.com/image.jpg',
                            },
                        }
                    ],
                }
            ]
        }

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        # Invalid image should be filtered out
        contents = gemini_request['contents']
        if contents:  # Check if any contents exist
            assert len(contents[0]['parts']) == 0
        else:
            assert len(contents) == 0  # No contents should be present

    @pytest.mark.asyncio
    async def test_tool_without_name_skipped(self):
        """Test that tools without names are skipped."""
        anthropic_request = {
            'messages': [{'role': 'user', 'content': 'Test'}],
            'tools': [
                {'description': 'A tool without a name', 'input_schema': {'type': 'object'}},
                {'name': 'valid_tool', 'description': 'A valid tool', 'input_schema': {'type': 'object'}},
            ],
        }

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        tools = gemini_request['tools']
        function_declarations = tools[0]['functionDeclarations']

        # Only the valid tool should be included
        assert len(function_declarations) == 1
        assert function_declarations[0]['name'] == 'valid_tool'

    @pytest.mark.asyncio
    async def test_thinking_parameter_handling(self):
        """Test that thinking parameter is logged but not passed to Gemini."""
        anthropic_request = {'messages': [{'role': 'user', 'content': 'Test thinking'}], 'thinking': {'type': 'enabled', 'budget_tokens': 2000}}

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        # Thinking parameter should not appear in Gemini request
        assert 'thinking' not in gemini_request

        # Should have logged the thinking parameter detection
        self.logger.info.assert_called_with(
            'Thinking parameter detected (budget_tokens: 2000) '
            'but Gemini API does not have direct reasoning effort equivalent. '
            'Consider using a Gemini model optimized for reasoning tasks.'
        )

    @pytest.mark.asyncio
    async def test_generation_config_comprehensive(self):
        """Test comprehensive generation config parameter mapping."""
        anthropic_request = {
            'messages': [{'role': 'user', 'content': 'Test comprehensive config'}],
            'temperature': 0.8,
            'max_tokens': 2000,
            'stop_sequences': ['END', 'STOP'],
            'top_p': 0.95,
            'top_k': 50,
        }

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        assert 'generationConfig' in gemini_request
        gen_config = gemini_request['generationConfig']

        # Check all mapped parameters
        assert gen_config['temperature'] == 0.8
        assert gen_config['maxOutputTokens'] == 2000
        assert gen_config['stopSequences'] == ['END', 'STOP']
        assert gen_config['topP'] == 0.95
        assert gen_config['topK'] == 50
        assert gen_config['candidateCount'] == 1

    @pytest.mark.asyncio
    async def test_generation_config_defaults(self):
        """Test that generation config includes default values when parameters not provided."""
        anthropic_request = {
            'messages': [{'role': 'user', 'content': 'Test defaults'}],
            'temperature': 0.7,  # Only provide temperature
        }

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        assert 'generationConfig' in gemini_request
        gen_config = gemini_request['generationConfig']

        # Check provided parameter
        assert gen_config['temperature'] == 0.7

        # Check defaults are applied
        assert gen_config['topP'] == 1.0
        assert gen_config['topK'] == 40
        assert gen_config['candidateCount'] == 1

        # Parameters not provided should not be in config
        assert 'maxOutputTokens' not in gen_config
        assert 'stopSequences' not in gen_config

    @pytest.mark.asyncio
    async def test_generation_config_partial_parameters(self):
        """Test generation config with only some parameters provided."""
        anthropic_request = {
            'messages': [{'role': 'user', 'content': 'Test partial params'}],
            'max_tokens': 1500,
            'top_p': 0.9,
            # No temperature, top_k, or stop_sequences
        }

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        gen_config = gemini_request['generationConfig']

        # Check provided parameters
        assert gen_config['maxOutputTokens'] == 1500
        assert gen_config['topP'] == 0.9

        # Check defaults for missing parameters
        assert gen_config['topK'] == 40
        assert gen_config['candidateCount'] == 1

        # Parameters not provided should not be in config
        assert 'temperature' not in gen_config
        assert 'stopSequences' not in gen_config

    @pytest.mark.asyncio
    async def test_tool_config_added_with_tools(self):
        """Test that toolConfig is added when tools are present."""
        anthropic_request = {
            'messages': [{'role': 'user', 'content': 'Use tools'}],
            'tools': [{'name': 'test_tool', 'description': 'A test tool', 'input_schema': {'type': 'object', 'properties': {}}}],
        }

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        # Should have tools
        assert 'tools' in gemini_request
        assert len(gemini_request['tools']) == 1

        # Should have toolConfig
        assert 'toolConfig' in gemini_request
        tool_config = gemini_request['toolConfig']
        assert tool_config['functionCallingConfig']['mode'] == 'AUTO'

    @pytest.mark.asyncio
    async def test_no_tool_config_without_tools(self):
        """Test that toolConfig is not added when no tools are present."""
        anthropic_request = {'messages': [{'role': 'user', 'content': 'No tools needed'}]}

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        # Should not have tools or toolConfig
        assert 'tools' not in gemini_request
        assert 'toolConfig' not in gemini_request

    @pytest.mark.asyncio
    async def test_thinking_content_block_handling(self):
        """Test that thinking content blocks are properly skipped with logging."""
        anthropic_request = {
            'messages': [
                {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'Let me think...', 'signature': 'sig'}, {'type': 'text', 'text': 'Here is my response.'}]}
            ]
        }

        params = {'request': anthropic_request, 'headers': {}}
        gemini_request, _ = await self.transformer.transform(params)

        contents = gemini_request['contents']
        parts = contents[0]['parts']

        # Only text block should remain, thinking block should be skipped
        assert len(parts) == 1
        assert parts[0] == {'text': 'Here is my response.'}

        # Should have logged the thinking content block skip
        self.logger.debug.assert_called_with('Thinking content block skipped - not supported by Gemini API')
