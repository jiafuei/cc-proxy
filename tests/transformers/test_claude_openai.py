import hashlib

import pytest

from app.config.log import get_logger
from app.transformers.providers.claude.openai import (
    ClaudeOpenAIRequestTransformer,
    ClaudeOpenAIResponseTransformer,
)


@pytest.mark.asyncio
async def test_chat_request_sets_store_false():
    transformer = ClaudeOpenAIRequestTransformer(get_logger(__name__))

    claude_request = {
        'model': 'gpt-4o-mini',
        'messages': [
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'Hello!'}],
            }
        ],
    }

    payload, headers = await transformer.transform({'request': claude_request, 'headers': {}})

    assert payload['store'] is False
    assert payload['messages'][0]['role'] == 'user'
    assert headers == {}


@pytest.mark.asyncio
async def test_chat_response_preserves_reasoning_images_and_tool_calls():
    transformer = ClaudeOpenAIResponseTransformer(get_logger(__name__))

    openai_response = {
        'id': 'chatcmpl-123',
        'model': 'gpt-4o-mini',
        'choices': [
            {
                'finish_reason': 'tool_calls',
                'message': {
                    'role': 'assistant',
                    'content': [
                        {'type': 'text', 'text': 'Hello there'},
                        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,Zm9v'}},
                    ],
                    'tool_calls': [
                        {
                            'id': 'call_1',
                            'type': 'function',
                            'function': {'name': 'lookup_tool', 'arguments': '{"topic": "weather"}'},
                        }
                    ],
                    'reasoning': {
                        'signature': 'sig123',
                        'tokens': [{'type': 'text', 'text': 'Thinking... '}],
                    },
                },
            }
        ],
        'annotations': [
            {
                'type': 'url_citation',
                'url_citation': {
                    'url': 'https://example.com',
                    'title': 'Example',
                    'start_index': 0,
                    'end_index': 5,
                },
            }
        ],
        'usage': {
            'prompt_tokens': 12,
            'completion_tokens': 20,
            'prompt_tokens_details': {'cached_tokens': 3},
            'completion_tokens_details': {'reasoning_tokens': 4},
        },
    }

    claude_response = await transformer.transform_response({'response': openai_response})

    assert claude_response['id'] == 'chatcmpl-123'
    assert claude_response['stop_reason'] == 'tool_use'

    content_types = [block['type'] for block in claude_response['content']]
    assert content_types == ['thinking', 'text', 'image', 'web_search_tool_result', 'tool_use']

    thinking_block = claude_response['content'][0]
    assert thinking_block['thinking'].strip() == 'Thinking...'
    assert thinking_block['signature'] == 'sig123'

    image_block = claude_response['content'][2]
    assert image_block['source']['type'] == 'base64'
    assert image_block['source']['media_type'] == 'image/png'
    assert image_block['source']['data'] == 'Zm9v'

    expected_id = f"search_{hashlib.md5('https://example.com'.encode()).hexdigest()[:8]}"
    search_block = claude_response['content'][3]
    assert search_block['id'] == expected_id
    assert search_block['content']['url'] == 'https://example.com'
    assert search_block['content']['title'] == 'Example'
    assert search_block['content']['snippet'] == 'Hello'

    tool_use_block = claude_response['content'][4]
    assert tool_use_block['name'] == 'lookup_tool'
    assert tool_use_block['input'] == {'topic': 'weather'}

    usage = claude_response['usage']
    assert usage['input_tokens'] == 12
    assert usage['output_tokens'] == 20
    assert usage['cache_read_input_tokens'] == 3
    assert usage['reasoning_output_tokens'] == 4
