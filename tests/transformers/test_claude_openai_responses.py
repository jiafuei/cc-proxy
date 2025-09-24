import hashlib
import json

import pytest

from app.config.log import get_logger
from app.transformers.providers.claude.openai_responses import (
    ClaudeOpenAIResponsesRequestTransformer,
    ClaudeOpenAIResponsesResponseTransformer,
)


@pytest.mark.asyncio
async def test_request_transform_text_only():
    transformer = ClaudeOpenAIResponsesRequestTransformer(get_logger(__name__))

    claude_request = {
        'model': 'gpt-4.1-mini',
        'system': [{'type': 'text', 'text': 'You are a helpful AI.'}],
        'messages': [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Hello!'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hi there!'}]},
        ],
        'temperature': 0.8,
        'max_tokens': 256,
    }

    payload, headers = await transformer.transform({'request': claude_request, 'headers': {}, 'provider_config': None, 'original_request': None, 'routing_key': None, 'exchange_request': None})

    assert payload['model'] == 'gpt-4.1-mini'
    assert payload['stream'] is False
    assert payload['store'] is False
    assert payload['instructions'] == 'You are a helpful AI.'
    assert payload['max_output_tokens'] == 256
    assert pytest.approx(payload['temperature'], rel=0.0, abs=1e-9) == 0.8

    assert len(payload['input']) == 2
    first_message = payload['input'][0]
    second_message = payload['input'][1]

    assert first_message['role'] == 'user'
    assert first_message['content'][0]['type'] == 'input_text'
    assert first_message['content'][0]['text'] == 'Hello!'

    assert second_message['role'] == 'assistant'
    assert second_message['content'][0]['type'] == 'output_text'
    assert second_message['content'][0]['text'] == 'Hi there!'

    assert headers == {}


@pytest.mark.asyncio
async def test_request_transform_with_tool_calls():
    transformer = ClaudeOpenAIResponsesRequestTransformer(get_logger(__name__))

    claude_request = {
        'model': 'gpt-4.1-mini',
        'messages': [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'tool_use', 'id': 'call_1', 'name': 'get_weather', 'input': {'location': 'San Francisco'}},
                ],
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'tool_result', 'tool_use_id': 'call_1', 'content': '{"temperature": "70F"}'},
                ],
            },
        ],
        'tools': [
            {
                'name': 'get_weather',
                'description': 'Fetch the current weather',
                'input_schema': {
                    'type': 'object',
                    'properties': {'location': {'type': 'string'}},
                    'required': ['location'],
                },
            }
        ],
        'tool_choice': {'type': 'tool', 'name': 'get_weather'},
    }

    payload, _ = await transformer.transform({'request': claude_request, 'headers': {}, 'provider_config': None, 'original_request': None, 'routing_key': None, 'exchange_request': None})

    assert payload['tools'][0]['type'] == 'function'
    assert payload['tools'][0]['name'] == 'get_weather'
    assert payload['tool_choice'] == {'type': 'function', 'function': {'name': 'get_weather'}}
    assert payload['parallel_tool_calls'] is False

    function_call = next(item for item in payload['input'] if item['type'] == 'function_call')
    assert function_call['name'] == 'get_weather'
    assert json.loads(function_call['arguments']) == {'location': 'San Francisco'}

    function_result = next(item for item in payload['input'] if item['type'] == 'function_call_output')
    assert function_result['call_id'] == 'call_1'
    assert function_result['output'] == '{"temperature": "70F"}'


@pytest.mark.asyncio
async def test_request_transform_with_builtin_web_search():
    transformer = ClaudeOpenAIResponsesRequestTransformer(get_logger(__name__))

    claude_request = {
        'model': 'gpt-4.1-mini',
        'messages': [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Find the news.'}]},
        ],
        'tools': [
            {
                'name': 'web_search',
                'type': 'tool',
                'allowed_domains': ['example.com'],
                'user_location': {'country': 'US'},
            }
        ],
    }

    payload, _ = await transformer.transform({'request': claude_request, 'headers': {}, 'provider_config': None, 'original_request': None, 'routing_key': None, 'exchange_request': None})

    tools = payload.get('tools')
    assert tools is not None and len(tools) == 1
    web_search_tool = tools[0]
    assert web_search_tool['type'] == 'web_search'
    assert web_search_tool['web_search']['filters']['allowed_domains'] == ['example.com']
    assert web_search_tool['web_search']['user_location']['type'] == 'approximate'
    assert web_search_tool['web_search']['user_location']['approximate']['country'] == 'US'


@pytest.mark.asyncio
async def test_response_transform_with_tool_call():
    transformer = ClaudeOpenAIResponsesResponseTransformer(get_logger(__name__))

    responses_payload = {
        'id': 'resp_123',
        'model': 'gpt-4.1-mini',
        'status': 'completed',
        'output': [
            {
                'type': 'message',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': 'Here is the forecast.'}],
            },
            {
                'type': 'function_call',
                'name': 'get_weather',
                'call_id': 'call_1',
                'arguments': '{"location": "San Francisco"}',
            },
            {
                'type': 'function_call_output',
                'call_id': 'call_1',
                'output': '{"temperature": "70F"}',
            },
        ],
        'usage': {'input_tokens': 42, 'output_tokens': 16, 'total_tokens': 58},
    }

    claude_response = await transformer.transform_response({'response': responses_payload, 'request': None, 'final_headers': {}, 'provider_config': None, 'original_request': None, 'exchange_request': None})

    assert claude_response['id'] == 'resp_123'
    assert claude_response['model'] == 'gpt-4.1-mini'
    assert claude_response['stop_reason'] == 'end_turn'

    text_block = next(block for block in claude_response['content'] if block['type'] == 'text')
    assert text_block['text'] == 'Here is the forecast.'

    tool_call_block = next(block for block in claude_response['content'] if block['type'] == 'tool_use')
    assert tool_call_block['name'] == 'get_weather'
    assert tool_call_block['id'] == 'call_1'
    assert tool_call_block['input'] == {'location': 'San Francisco'}

    tool_result_block = next(block for block in claude_response['content'] if block['type'] == 'tool_result')
    assert tool_result_block['tool_use_id'] == 'call_1'
    assert tool_result_block['content'] == [{'type': 'text', 'text': '{"temperature": "70F"}'}]

    assert claude_response['usage']['input_tokens'] == 42
    assert claude_response['usage']['output_tokens'] == 16
    assert claude_response['usage']['total_tokens'] == 58


@pytest.mark.asyncio
async def test_response_transform_with_reasoning_image_and_web_search():
    transformer = ClaudeOpenAIResponsesResponseTransformer(get_logger(__name__))

    responses_payload = {
        'id': 'resp_999',
        'model': 'gpt-4.1-mini',
        'status': 'requires_action',
        'output': [
            {
                'type': 'message',
                'role': 'assistant',
                'content': [
                    {'type': 'reasoning', 'reasoning': 'Step 1. Step 2.', 'signature': 'sig-abc'},
                    {'type': 'output_text', 'text': 'Answer ready.'},
                    {'type': 'output_image', 'image_url': 'https://example.com/image.png'},
                    {
                        'type': 'web_search_result',
                        'web_search_result': {
                            'url': 'https://example.com',
                            'title': 'Example',
                            'snippet': 'Hello world',
                        },
                    },
                ],
            },
            {
                'type': 'function_call',
                'name': 'lookup_tool',
                'call_id': 'call_42',
                'arguments': {'subject': 'test'},
            },
            {
                'type': 'function_call_output',
                'call_id': 'call_42',
                'output': {'result': 'ok'},
            },
        ],
        'usage': {
            'input_tokens': 12,
            'output_tokens': 34,
            'prompt_tokens_details': {'cached_tokens': 5},
            'completion_tokens_details': {'reasoning_tokens': 7},
        },
    }

    claude_response = await transformer.transform_response({'response': responses_payload, 'request': None, 'final_headers': {}, 'provider_config': None, 'original_request': None, 'exchange_request': None})

    assert claude_response['stop_reason'] == 'tool_use'

    content_types = [block['type'] for block in claude_response['content']]
    assert content_types == ['thinking', 'text', 'image', 'web_search_tool_result', 'tool_use', 'tool_result']

    thinking_block = claude_response['content'][0]
    assert thinking_block['thinking'] == 'Step 1. Step 2.'
    assert thinking_block['signature'] == 'sig-abc'

    image_block = claude_response['content'][2]
    assert image_block['source']['url'] == 'https://example.com/image.png'

    expected_id = f"search_{hashlib.md5('https://example.com'.encode()).hexdigest()[:8]}"
    search_block = claude_response['content'][3]
    assert search_block['id'] == expected_id
    assert search_block['content']['url'] == 'https://example.com'
    assert search_block['content']['title'] == 'Example'
    assert search_block['content']['snippet'] == 'Hello world'

    tool_result_block = claude_response['content'][5]
    assert tool_result_block['content'] == [{'type': 'text', 'text': '{"result":"ok"}'}]

    usage = claude_response['usage']
    assert usage['input_tokens'] == 12
    assert usage['output_tokens'] == 34
    assert usage['cache_read_input_tokens'] == 5
    assert usage['reasoning_output_tokens'] == 7
