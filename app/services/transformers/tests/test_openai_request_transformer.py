"""Tests for OpenAI Request Transformer."""

from unittest.mock import Mock

import pytest

from app.services.transformers.openai import OpenAIRequestTransformer


class TestOpenAIRequestTransformer:
    """Test cases for the OpenAI request transformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance with mock logger."""
        mock_logger = Mock()
        return OpenAIRequestTransformer(mock_logger)

    @pytest.fixture
    def sample_claude_request(self):
        """Sample Claude request with system messages."""
        return {
            'model': 'claude-sonnet-4-20250514',
            'temperature': 1.0,
            'stream': True,
            'system': [
                {'type': 'text', 'text': 'You are Claude Code', 'cache_control': {'type': 'ephemeral'}},
                {'type': 'text', 'text': ' - a helpful assistant for coding.', 'cache_control': {'type': 'ephemeral'}},
            ],
            'messages': [{'role': 'user', 'content': 'Hello'}, {'role': 'user', 'content': 'Hi there!'}],
            'tools': [{'name': 'Read', 'description': 'Read files', 'input_schema': {'type': 'object', 'properties': {'file_path': {'type': 'string'}}}}],
        }

    def test_convert_system_messages_multiple_blocks(self, transformer):
        """Test combining multiple system message blocks into one."""
        claude_request = {
            'system': [
                {'type': 'text', 'text': 'You are Claude Code', 'cache_control': {'type': 'ephemeral'}},
                {'type': 'text', 'text': ' - a helpful assistant.'},
            ]
        }

        result = transformer._convert_messages(claude_request)

        assert len(result) == 1
        assert result[0] == {'role': 'system', 'content': 'You are Claude Code\n - a helpful assistant.'}

    def test_convert_system_messages_single_block(self, transformer):
        """Test single system message conversion."""
        claude_request = {'system': [{'type': 'text', 'text': 'You are a helpful assistant.'}]}

        result = transformer._convert_messages(claude_request)

        assert len(result) == 1
        assert result[0] == {'role': 'system', 'content': 'You are a helpful assistant.'}

    def test_convert_system_messages_empty(self, transformer):
        """Test empty system messages."""
        assert transformer._convert_messages({}) == []
        assert transformer._convert_messages({'system': []}) == []

    def test_convert_system_messages_no_text_blocks(self, transformer):
        """Test system array with no text blocks."""
        claude_request = {'system': [{'type': 'other', 'data': 'some data'}]}

        result = transformer._convert_messages(claude_request)
        assert result == []

    def test_convert_user_message_string_content(self, transformer):
        """Test user message with string content."""
        claude_request = {'messages': [{'role': 'user', 'content': 'Hello, how are you?'}]}

        result = transformer._convert_messages(claude_request)

        assert len(result) == 1
        assert result[0] == {'role': 'user', 'content': 'Hello, how are you?'}

    def test_convert_user_message_list_content_text_only(self, transformer):
        """Test user message with list content containing only text blocks."""
        claude_request = {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'First part'}, {'type': 'text', 'text': 'Second part'}]}]}

        result = transformer._convert_messages(claude_request)

        assert len(result) == 1
        assert result[0] == {'role': 'user', 'content': [{'type': 'text', 'text': 'First part'}, {'type': 'text', 'text': 'Second part'}]}

    def test_convert_user_message_list_content_mixed_blocks(self, transformer):
        """Test user message with mixed content blocks (text and images converted)."""
        claude_request = {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Text part'},
                        {'type': 'image', 'source': {'type': 'base64', 'data': 'abc123', 'media_type': 'image/jpeg'}},
                        {'type': 'text', 'text': 'More text'},
                    ],
                }
            ]
        }

        result = transformer._convert_messages(claude_request)

        assert len(result) == 1
        assert result[0] == {
            'role': 'user',
            'content': [{'type': 'text', 'text': 'Text part'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,abc123'}}, {'type': 'text', 'text': 'More text'}],
        }

    def test_convert_user_message_non_user_role(self, transformer):
        """Test that assistant messages are properly converted."""
        claude_request = {'messages': [{'role': 'assistant', 'content': 'I am an assistant'}]}

        result = transformer._convert_messages(claude_request)

        assert len(result) == 1
        assert result[0] == {'role': 'assistant', 'content': 'I am an assistant'}

    def test_convert_user_message_empty_content(self, transformer):
        """Test user message with empty content."""
        claude_request = {'messages': [{'role': 'user', 'content': ''}]}

        result = transformer._convert_messages(claude_request)

        assert len(result) == 1
        assert result[0] == {'role': 'user', 'content': [{'type': 'text', 'text': ''}]}

    def test_convert_user_message_no_convertible_blocks(self, transformer):
        """Test user message with no convertible blocks (unsupported types)."""
        claude_request = {'messages': [{'role': 'user', 'content': [{'type': 'tool_use', 'id': '123', 'name': 'Read'}, {'type': 'unsupported', 'data': 'some data'}]}]}

        result = transformer._convert_messages(claude_request)

        # Should create assistant message with tool call, not user message
        assert len(result) == 0  # No convertible content for user

    def test_convert_image_block_valid(self, transformer):
        """Test converting valid Claude image block to OpenAI format."""
        claude_image = {'type': 'image', 'source': {'type': 'base64', 'data': 'iVBORw0KGgoAAAANSUhEUgAA', 'media_type': 'image/png'}}

        result = transformer._convert_image_block(claude_image)

        assert result == {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA'}}

    def test_convert_image_block_invalid_type(self, transformer):
        """Test that non-image blocks are not converted."""
        claude_block = {'type': 'text', 'text': 'Not an image'}

        result = transformer._convert_image_block(claude_block)
        assert result is None

    def test_convert_image_block_invalid_source(self, transformer):
        """Test image block with invalid source."""
        claude_image = {
            'type': 'image',
            'source': {
                'type': 'url',  # Not base64
                'url': 'https://example.com/image.jpg',
            },
        }

        result = transformer._convert_image_block(claude_image)
        assert result is None

    def test_convert_image_block_missing_source(self, transformer):
        """Test image block with missing source."""
        claude_image = {'type': 'image'}

        result = transformer._convert_image_block(claude_image)
        assert result is None

    def test_convert_user_message_only_images(self, transformer):
        """Test user message with only image content."""
        claude_request = {'messages': [{'role': 'user', 'content': [{'type': 'image', 'source': {'type': 'base64', 'data': 'xyz789', 'media_type': 'image/gif'}}]}]}

        result = transformer._convert_messages(claude_request)

        assert len(result) == 1
        assert result[0] == {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/gif;base64,xyz789'}}]}

    def test_convert_system_messages_mixed_blocks(self, transformer):
        """Test system array with mixed block types."""
        claude_request = {'system': [{'type': 'text', 'text': 'First part'}, {'type': 'other', 'data': 'ignored'}, {'type': 'text', 'text': ' second part'}]}

        result = transformer._convert_messages(claude_request)

        assert len(result) == 1
        assert result[0] == {'role': 'system', 'content': 'First part\n second part'}

    def test_convert_messages_with_system(self, transformer, sample_claude_request):
        """Test messages conversion with system messages."""
        result = transformer._convert_messages(sample_claude_request)

        assert len(result) == 3  # system + 2 user messages
        assert result[0] == {'role': 'system', 'content': 'You are Claude Code\n - a helpful assistant for coding.'}
        assert result[1] == {'role': 'user', 'content': [{'type': 'text', 'text': 'Hello'}]}
        assert result[2] == {'role': 'user', 'content': [{'type': 'text', 'text': 'Hi there!'}]}

    def test_convert_messages_no_system(self, transformer):
        """Test messages conversion without system messages."""
        claude_request = {'messages': [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi!'}]}

        result = transformer._convert_messages(claude_request)

        assert len(result) == 2  # user and assistant messages both converted
        assert result[0] == {'role': 'user', 'content': [{'type': 'text', 'text': 'Hello'}]}
        assert result[1] == {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hi!'}]}

    def test_convert_messages_empty_system(self, transformer):
        """Test messages conversion with empty system array."""
        claude_request = {'system': [], 'messages': [{'role': 'user', 'content': 'Hello'}]}

        result = transformer._convert_messages(claude_request)

        assert len(result) == 1  # 1 user message converted
        assert result[0] == {'role': 'user', 'content': [{'type': 'text', 'text': 'Hello'}]}

    def test_convert_messages_no_messages_field(self, transformer):
        """Test messages conversion when messages field is missing."""
        claude_request = {'system': [{'type': 'text', 'text': 'System message'}]}

        result = transformer._convert_messages(claude_request)

        assert len(result) == 1
        assert result[0] == {'role': 'system', 'content': 'System message'}

    @pytest.mark.asyncio
    async def test_transform_preserves_original_request(self, transformer, sample_claude_request):
        """Test that transform doesn't modify the original request."""
        original_system = sample_claude_request['system'].copy()
        original_messages = sample_claude_request['messages'].copy()

        params = {'request': sample_claude_request, 'headers': {'content-type': 'application/json'}}

        openai_request, headers = await transformer.transform(params)

        # Original request should be unchanged
        assert sample_claude_request['system'] == original_system
        assert sample_claude_request['messages'] == original_messages
        assert 'system' in sample_claude_request  # system field still exists in original

    def test_convert_tool_result_to_message_valid(self, transformer):
        """Test converting valid tool_result block to OpenAI tool message."""
        tool_result_block = {'type': 'tool_result', 'tool_use_id': 'toolu_123abc', 'content': 'File contents here', 'is_error': False}

        result = transformer._convert_tool_result(tool_result_block)

        assert result == {'role': 'tool', 'tool_call_id': 'toolu_123abc', 'content': 'File contents here'}

    def test_convert_tool_result_to_message_empty_content_success(self, transformer):
        """Test tool_result with empty content and no is_error field (defaults to Success)."""
        tool_result_block = {'type': 'tool_result', 'tool_use_id': 'toolu_456def', 'content': ''}

        result = transformer._convert_tool_result(tool_result_block)

        assert result == {'role': 'tool', 'tool_call_id': 'toolu_456def', 'content': 'Success'}

    def test_convert_tool_result_to_message_empty_content_error(self, transformer):
        """Test tool_result with empty content and is_error=True."""
        tool_result_block = {'type': 'tool_result', 'tool_use_id': 'toolu_789ghi', 'content': '', 'is_error': True}

        result = transformer._convert_tool_result(tool_result_block)

        assert result == {'role': 'tool', 'tool_call_id': 'toolu_789ghi', 'content': 'Error'}

    def test_convert_tool_result_to_message_empty_content_is_error_false(self, transformer):
        """Test tool_result with empty content and explicit is_error=False."""
        tool_result_block = {'type': 'tool_result', 'tool_use_id': 'toolu_abc123', 'content': '', 'is_error': False}

        result = transformer._convert_tool_result(tool_result_block)

        assert result == {'role': 'tool', 'tool_call_id': 'toolu_abc123', 'content': 'Success'}

    def test_convert_tool_result_to_message_non_empty_content_with_error_flag(self, transformer):
        """Test tool_result with actual content and is_error=True (content preserved)."""
        tool_result_block = {'type': 'tool_result', 'tool_use_id': 'toolu_preserve', 'content': 'Actual error message', 'is_error': True}

        result = transformer._convert_tool_result(tool_result_block)

        assert result == {'role': 'tool', 'tool_call_id': 'toolu_preserve', 'content': 'Actual error message'}

    def test_convert_tool_result_to_message_invalid_type(self, transformer):
        """Test that non-tool_result blocks are not converted."""
        block = {'type': 'text', 'text': 'Not a tool result'}

        # This test no longer applies since _convert_tool_result doesn't validate type
        # The validation is now done at the message processing level
        result = transformer._convert_tool_result(block)
        assert result['role'] == 'tool'  # Will still create a tool message with None ID

    def test_convert_tool_result_to_message_missing_id(self, transformer):
        """Test tool_result block with missing tool_use_id."""
        tool_result_block = {'type': 'tool_result', 'content': 'Some content'}

        result = transformer._convert_tool_result(tool_result_block)
        assert result['tool_call_id'] is None

    def test_convert_content_block_text(self, transformer):
        """Test converting text content block through _convert_content_blocks."""
        blocks = [{'type': 'text', 'text': 'Hello world'}]

        result = transformer._convert_content_blocks(blocks)

        assert result == [{'type': 'text', 'text': 'Hello world'}]

    def test_convert_content_block_image(self, transformer):
        """Test converting image content block through _convert_content_blocks."""
        blocks = [{'type': 'image', 'source': {'type': 'base64', 'data': 'xyz789', 'media_type': 'image/png'}}]

        result = transformer._convert_content_blocks(blocks)

        assert result == [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,xyz789'}}]

    def test_convert_content_block_unsupported(self, transformer):
        """Test unsupported content block type through _convert_content_blocks."""
        blocks = [{'type': 'unknown', 'data': 'something'}]

        result = transformer._convert_content_blocks(blocks)
        assert result == []

    def test_queue_processing_with_tool_result_boundary(self, transformer):
        """Test queue processing with tool_result creating message boundaries."""
        claude_request = {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Before tool'},
                        {'type': 'tool_result', 'tool_use_id': 'toolu_123', 'content': 'Tool result'},
                        {'type': 'text', 'text': 'After tool'},
                    ],
                }
            ]
        }

        result = transformer._convert_messages(claude_request)

        # Should create 3 messages: user → tool → user
        assert len(result) == 3
        assert result[0] == {'role': 'user', 'content': [{'type': 'text', 'text': 'Before tool'}]}
        assert result[1] == {'role': 'tool', 'tool_call_id': 'toolu_123', 'content': 'Tool result'}
        assert result[2] == {'role': 'user', 'content': [{'type': 'text', 'text': 'After tool'}]}

    def test_queue_processing_multiple_tool_results(self, transformer):
        """Test queue processing with multiple tool_results in sequence."""
        claude_request = {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Start'},
                        {'type': 'tool_result', 'tool_use_id': 'toolu_A', 'content': 'Result A'},
                        {'type': 'tool_result', 'tool_use_id': 'toolu_B', 'content': 'Result B'},
                        {'type': 'text', 'text': 'End'},
                    ],
                }
            ]
        }

        result = transformer._convert_messages(claude_request)

        # Should create 4 messages: user → tool → tool → user
        assert len(result) == 4
        assert result[0]['role'] == 'user'
        assert result[0]['content'] == [{'type': 'text', 'text': 'Start'}]
        assert result[1]['role'] == 'tool'
        assert result[1]['tool_call_id'] == 'toolu_A'
        assert result[2]['role'] == 'tool'
        assert result[2]['tool_call_id'] == 'toolu_B'
        assert result[3]['role'] == 'user'
        assert result[3]['content'] == [{'type': 'text', 'text': 'End'}]

    def test_queue_processing_only_tool_results(self, transformer):
        """Test queue processing with message containing only tool_results."""
        claude_request = {'messages': [{'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_only', 'content': 'Only result'}]}]}

        result = transformer._convert_messages(claude_request)

        # Should create 1 tool message (no user message since no user content)
        assert len(result) == 1
        assert result[0] == {'role': 'tool', 'tool_call_id': 'toolu_only', 'content': 'Only result'}

    def test_queue_processing_mixed_content_with_images_and_tools(self, transformer):
        """Test complex queue processing with text, images, and tool_results."""
        claude_request = {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Look at this:'},
                        {'type': 'image', 'source': {'type': 'base64', 'data': 'img123', 'media_type': 'image/jpeg'}},
                        {'type': 'tool_result', 'tool_use_id': 'toolu_mixed', 'content': 'Analysis complete'},
                        {'type': 'text', 'text': 'What do you think?'},
                    ],
                }
            ]
        }

        result = transformer._convert_messages(claude_request)

        # Should create 3 messages: user(text+image) → tool → user(text)
        assert len(result) == 3

        # First message: text + image
        assert result[0]['role'] == 'user'
        assert len(result[0]['content']) == 2
        assert result[0]['content'][0]['type'] == 'text'
        assert result[0]['content'][1]['type'] == 'image_url'

        # Second message: tool result
        assert result[1]['role'] == 'tool'
        assert result[1]['tool_call_id'] == 'toolu_mixed'

        # Third message: text
        assert result[2]['role'] == 'user'
        assert result[2]['content'] == [{'type': 'text', 'text': 'What do you think?'}]

    def test_queue_processing_preserves_message_boundaries(self, transformer):
        """Test that separate user messages remain separate (message boundary preservation)."""
        claude_request = {'messages': [{'role': 'user', 'content': 'First message'}, {'role': 'user', 'content': 'Second message'}]}

        result = transformer._convert_messages(claude_request)

        # Should create 2 separate user messages
        assert len(result) == 2
        assert result[0] == {'role': 'user', 'content': [{'type': 'text', 'text': 'First message'}]}
        assert result[1] == {'role': 'user', 'content': [{'type': 'text', 'text': 'Second message'}]}

    def test_convert_assistant_message_text_only(self, transformer):
        """Test assistant message with only text content."""
        claude_request = {'messages': [{'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hello, I can help you with that.'}]}]}

        result = transformer._convert_messages(claude_request)

        assert len(result) == 1
        assert result[0] == {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hello, I can help you with that.'}]}

    def test_convert_assistant_message_with_thinking_blocks(self, transformer):
        """Test assistant message with thinking blocks (should be filtered out)."""
        claude_request = {
            'messages': [
                {
                    'role': 'assistant',
                    'content': [
                        {'type': 'thinking', 'thinking': 'Let me think about this...', 'signature': 'some_signature'},
                        {'type': 'text', 'text': 'Based on my analysis...'},
                        {'type': 'thinking', 'thinking': 'More thoughts...', 'signature': 'another_signature'},
                        {'type': 'text', 'text': 'Here is my conclusion.'},
                    ],
                }
            ]
        }

        result = transformer._convert_messages(claude_request)

        # Should only include text blocks, thinking blocks filtered out
        assert len(result) == 1
        assert result[0] == {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Based on my analysis...'}, {'type': 'text', 'text': 'Here is my conclusion.'}]}

    def test_convert_assistant_message_only_thinking_blocks(self, transformer):
        """Test assistant message with only thinking blocks (should create no message)."""
        claude_request = {'messages': [{'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'Just thinking...', 'signature': 'sig'}]}]}

        result = transformer._convert_messages(claude_request)

        # Should create no messages since all blocks were thinking blocks
        assert len(result) == 0

    def test_convert_mixed_user_and_assistant_messages(self, transformer):
        """Test conversion with both user and assistant messages."""
        claude_request = {
            'messages': [
                {'role': 'user', 'content': 'What is the weather today?'},
                {
                    'role': 'assistant',
                    'content': [
                        {'type': 'thinking', 'thinking': 'User asking about weather...', 'signature': 'sig'},
                        {'type': 'text', 'text': 'I can help you check the weather.'},
                    ],
                },
                {'role': 'user', 'content': 'Thank you!'},
            ]
        }

        result = transformer._convert_messages(claude_request)

        # Should create 3 messages: user → assistant → user
        assert len(result) == 3
        assert result[0] == {'role': 'user', 'content': [{'type': 'text', 'text': 'What is the weather today?'}]}
        assert result[1] == {'role': 'assistant', 'content': [{'type': 'text', 'text': 'I can help you check the weather.'}]}
        assert result[2] == {'role': 'user', 'content': [{'type': 'text', 'text': 'Thank you!'}]}

    def test_convert_assistant_message_string_content(self, transformer):
        """Test assistant message with string content (converted to array)."""
        claude_request = {'messages': [{'role': 'assistant', 'content': 'Simple string response'}]}

        result = transformer._convert_messages(claude_request)

        assert len(result) == 1
        assert result[0] == {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Simple string response'}]}

    @pytest.mark.asyncio
    async def test_full_transform_integration(self, transformer, sample_claude_request):
        """Test complete transform method integration."""
        params = {'request': sample_claude_request, 'headers': {'content-type': 'application/json'}}

        openai_request, headers = await transformer.transform(params)

        # Check OpenAI format structure
        assert 'messages' in openai_request
        assert len(openai_request['messages']) == 3  # system + 2 user messages

        # Check system message conversion
        system_msg = openai_request['messages'][0]
        assert system_msg['role'] == 'system'
        assert system_msg['content'] == 'You are Claude Code\n - a helpful assistant for coding.'

        # Check user message conversion
        user_msg1 = openai_request['messages'][1]
        assert user_msg1['role'] == 'user'
        assert user_msg1['content'] == [{'type': 'text', 'text': 'Hello'}]

        user_msg2 = openai_request['messages'][2]
        assert user_msg2['role'] == 'user'
        assert user_msg2['content'] == [{'type': 'text', 'text': 'Hi there!'}]

        # Check other fields are preserved
        assert openai_request['model'] == 'claude-sonnet-4-20250514'
        assert openai_request['temperature'] == 1.0
        assert openai_request['stream'] is True
        assert openai_request['stream_options'] == {'include_usage': True}

        # Check tools conversion
        assert openai_request['tools'] is not None
        assert len(openai_request['tools']) == 1

    def test_convert_tool_use_to_tool_call_valid(self, transformer):
        """Test converting valid tool_use block to OpenAI tool_call format."""
        tool_use_block = {'type': 'tool_use', 'id': 'toolu_123abc', 'name': 'Read', 'input': {'file_path': '/path/to/file.txt'}}

        result = transformer._convert_tool_call(tool_use_block)

        assert result == {'id': 'toolu_123abc', 'type': 'function', 'function': {'name': 'Read', 'arguments': '{"file_path":"/path/to/file.txt"}'}}

    def test_convert_tool_use_to_tool_call_empty_input(self, transformer):
        """Test tool_use with empty input object."""
        tool_use_block = {'type': 'tool_use', 'id': 'toolu_456def', 'name': 'Ping', 'input': {}}

        result = transformer._convert_tool_call(tool_use_block)

        assert result == {'id': 'toolu_456def', 'type': 'function', 'function': {'name': 'Ping', 'arguments': '{}'}}

    def test_convert_tool_use_to_tool_call_invalid_type(self, transformer):
        """Test that non-tool_use blocks still create tool call structure."""
        block = {'type': 'text', 'text': 'Not a tool use'}

        result = transformer._convert_tool_call(block)
        assert result['type'] == 'function'
        assert result['id'] is None

    def test_convert_tool_use_to_tool_call_missing_id(self, transformer):
        """Test tool_use block with missing id."""
        tool_use_block = {'type': 'tool_use', 'name': 'Read', 'input': {'file_path': '/path/to/file.txt'}}

        result = transformer._convert_tool_call(tool_use_block)
        assert result['id'] is None
        assert result['function']['name'] == 'Read'

    def test_convert_tool_use_to_tool_call_missing_name(self, transformer):
        """Test tool_use block with missing name."""
        tool_use_block = {'type': 'tool_use', 'id': 'toolu_123abc', 'input': {'file_path': '/path/to/file.txt'}}

        result = transformer._convert_tool_call(tool_use_block)
        assert result['id'] == 'toolu_123abc'
        assert result['function']['name'] is None

    def test_convert_assistant_message_with_tool_use_only(self, transformer):
        """Test assistant message with only tool_use blocks."""
        claude_request = {
            'messages': [
                {
                    'role': 'assistant',
                    'content': [
                        {'type': 'tool_use', 'id': 'toolu_read123', 'name': 'Read', 'input': {'file_path': 'test.py'}},
                        {'type': 'tool_use', 'id': 'toolu_write456', 'name': 'Write', 'input': {'file_path': 'output.txt', 'content': 'Hello'}},
                    ],
                }
            ]
        }

        result = transformer._convert_messages(claude_request)

        # Should create 1 assistant message with merged tool_calls
        assert len(result) == 1
        assert result[0] == {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {'id': 'toolu_read123', 'type': 'function', 'function': {'name': 'Read', 'arguments': '{"file_path":"test.py"}'}},
                {'id': 'toolu_write456', 'type': 'function', 'function': {'name': 'Write', 'arguments': '{"file_path":"output.txt","content":"Hello"}'}},
            ],
        }

    def test_convert_assistant_message_with_thinking_and_tool_use(self, transformer):
        """Test assistant message with thinking blocks and tool_use blocks."""
        claude_request = {
            'messages': [
                {
                    'role': 'assistant',
                    'content': [
                        {'type': 'thinking', 'thinking': 'I need to read the file first...', 'signature': 'sig1'},
                        {'type': 'tool_use', 'id': 'toolu_read', 'name': 'Read', 'input': {'file_path': 'config.yaml'}},
                    ],
                }
            ]
        }

        result = transformer._convert_messages(claude_request)

        # Should create 1 assistant message with only tool_calls (thinking blocks filtered)
        assert len(result) == 1
        assert result[0] == {
            'role': 'assistant',
            'tool_calls': [{'id': 'toolu_read', 'type': 'function', 'function': {'name': 'Read', 'arguments': '{"file_path":"config.yaml"}'}}],
            'content': None,
        }

    def test_convert_complex_conversation_with_tool_use(self, transformer):
        """Test complex conversation with user, assistant text, and assistant tool_use."""
        claude_request = {
            'messages': [
                {'role': 'user', 'content': 'Please read the config file'},
                {
                    'role': 'assistant',
                    'content': [
                        {'type': 'text', 'text': "I'll read the config file for you."},
                        {'type': 'tool_use', 'id': 'toolu_config', 'name': 'Read', 'input': {'file_path': 'config.yaml'}},
                    ],
                },
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_config', 'content': 'port: 8080\ndebug: true'}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': 'The config shows port 8080 and debug mode enabled.'}]},
            ]
        }

        result = transformer._convert_messages(claude_request)

        # Should create 4 messages: user → assistant(content+tool_calls) → tool → assistant(content)
        assert len(result) == 4

        # User message
        assert result[0]['role'] == 'user'
        assert result[0]['content'] == 'Please read the config file'

        # Assistant message with both content and tool_calls combined
        assert result[1]['role'] == 'assistant'
        assert result[1]['content'] == "I'll read the config file for you."
        assert len(result[1]['tool_calls']) == 1
        assert result[1]['tool_calls'][0]['id'] == 'toolu_config'

        # Tool result message
        assert result[2]['role'] == 'tool'
        assert result[2]['tool_call_id'] == 'toolu_config'
        assert result[2]['content'] == 'port: 8080\ndebug: true'

        # Final assistant message
        assert result[3]['role'] == 'assistant'
        assert result[3]['content'] == 'The config shows port 8080 and debug mode enabled.'

    def test_user_tool_use_blocks_ignored(self, transformer):
        """Test that tool_use blocks in user messages are ignored (not converted)."""
        claude_request = {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Here is some text'},
                        {
                            'type': 'tool_use',  # This should be ignored for user messages
                            'id': 'toolu_invalid',
                            'name': 'SomeFunction',
                            'input': {'param': 'value'},
                        },
                    ],
                }
            ]
        }

        result = transformer._convert_messages(claude_request)

        # Should create 1 user message with only text content
        assert len(result) == 1
        assert result[0] == {'role': 'user', 'content': [{'type': 'text', 'text': 'Here is some text'}]}

    def test_user_text_followed_by_assistant_tool_call(self, transformer):
        """Test user text message followed by assistant tool call."""
        claude_request = {
            'messages': [
                {'role': 'user', 'content': 'Please read the config file'},
                {
                    'role': 'assistant',
                    'content': [
                        {'type': 'tool_use', 'id': 'toolu_read123', 'name': 'Read', 'input': {'file_path': 'config.yaml'}},
                    ],
                },
            ]
        }

        result = transformer._convert_messages(claude_request)

        # Should create 2 messages: user text → assistant tool call
        assert len(result) == 2
        
        # User message (string content gets converted to simple string format)
        assert result[0] == {'role': 'user', 'content': 'Please read the config file'}
        
        # Assistant message with tool call
        assert result[1] == {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {'id': 'toolu_read123', 'type': 'function', 'function': {'name': 'Read', 'arguments': '{"file_path":"config.yaml"}'}}
            ],
        }
