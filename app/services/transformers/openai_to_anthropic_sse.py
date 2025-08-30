"""OpenAI to Anthropic SSE transformer for streaming response conversion."""

from typing import Any, AsyncIterator, Dict, Optional

import orjson

from app.config.log import get_logger
from app.services.transformers.interfaces import ResponseTransformer

logger = get_logger(__name__)


class OpenAIToAnthropicSSETransformer(ResponseTransformer):
    """Convert OpenAI streaming responses to Anthropic SSE format.
    
    Handles sequential conversion from OpenAI Chat Completions streaming format
    to Anthropic Messages API streaming format with proper event-based SSE structure.
    """

    # OpenAI SSE constants
    DATA_PREFIX = b'data: '
    DONE_MARKER = b'[DONE]'
    
    # Stop reason mapping
    STOP_REASON_MAPPING = {
        'stop': 'end_turn',
        'length': 'max_tokens', 
        'content_filter': 'stop_sequence',
        'tool_calls': 'tool_use'
    }

    def __init__(self, logger):
        """Initialize the transformer with logger."""
        super().__init__(logger)

    def _init_state(self) -> Dict[str, Any]:
        """Initialize processing state for a new message stream."""
        return {
            'message_id': '',
            'model': '',
            'next_block_index': 0,
            'active_text_block': None,
            'active_tool_block': None,
            'usage_tokens': {},
            'stop_reason': None,
            'message_started': False
        }

    def _format_anthropic_sse(self, event_type: str, data: Dict[str, Any]) -> bytes:
        """Format data as Anthropic SSE event."""
        event_line = f'event: {event_type}\n'
        data_line = f'data: {orjson.dumps(data).decode()}\n\n'
        return (event_line + data_line).encode()

    async def transform_chunk(self, params: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Transform OpenAI SSE chunk to Anthropic SSE format."""
        # Initialize state if not present
        if 'sse_state' not in params:
            params['sse_state'] = self._init_state()
        
        state = params['sse_state']
        chunk: bytes = params['chunk']
        
        # Process each line in the chunk
        lines = chunk.splitlines()
        for line in lines:
            if not line:
                continue
                
            if not line.startswith(self.DATA_PREFIX):
                self.logger.debug(f'Non-data SSE line: {line}')
                continue
                
            # Handle [DONE] marker
            data_content = line[len(self.DATA_PREFIX):]
            if data_content == self.DONE_MARKER:
                yield self._format_anthropic_sse('message_stop', {})
                continue
            
            # Parse JSON data
            try:
                openai_data = orjson.loads(data_content)
            except orjson.JSONDecodeError as e:
                self.logger.error(f'Failed to parse OpenAI SSE JSON: {e}')
                continue
                
            # Process the OpenAI chunk
            async for event_bytes in self._process_openai_chunk(openai_data, state):
                yield event_bytes

    async def _process_openai_chunk(self, data: Dict[str, Any], state: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Process a single OpenAI data chunk and emit Anthropic events."""
        # Extract basic message info
        message_id = data.get('id', '')
        model = data.get('model', '')
        choices = data.get('choices', [])
        usage = data.get('usage')
        
        # Update state
        if message_id:
            state['message_id'] = message_id
        if model:
            state['model'] = model
        if usage:
            state['usage_tokens'] = usage

        # Handle usage-only chunks (at end)
        if not choices and usage:
            stop_reason = self._convert_stop_reason(state.get('stop_reason'))
            usage_data = self._convert_openai_usage(usage)
            
            data = {
                'type': 'message_delta',
                'delta': {'stop_reason': stop_reason, 'stop_sequence': None},
                'usage': usage_data
            }
            yield self._format_anthropic_sse('message_delta', data)
            return

        # Process first choice (OpenAI typically uses index 0)  
        if not choices:
            return
            
        choice = choices[0]
        delta = choice.get('delta', {})
        finish_reason = choice.get('finish_reason')
        
        # Handle role (message start)
        if 'role' in delta and not state['message_started']:
            yield self._format_anthropic_sse('message_start', {
                'type': 'message_start',
                'message': {
                    'id': state['message_id'],
                    'type': 'message', 
                    'role': delta['role'],
                    'model': state['model'],
                    'content': [],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'usage': {
                        'input_tokens': 0,
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'cache_creation': {
                            'ephemeral_5m_input_tokens': 0,
                            'ephemeral_1h_input_tokens': 0
                        },
                        'output_tokens': 0,
                        'service_tier': 'standard'
                    }
                }
            })
            state['message_started'] = True

        # Handle text content
        if delta.get('content') is not None:
            if state['active_text_block'] is None:
                # Start new text block
                block_index = state['next_block_index'] 
                state['next_block_index'] += 1
                state['active_text_block'] = block_index
                
                yield self._format_anthropic_sse('content_block_start', {
                    'type': 'content_block_start',
                    'index': block_index,
                    'content_block': {
                        'type': 'text',
                        'text': ''
                    }
                })
            
            # Stream text delta
            yield self._format_anthropic_sse('content_block_delta', {
                'type': 'content_block_delta',
                'index': state['active_text_block'],
                'delta': {
                    'type': 'text_delta',
                    'text': delta['content']
                }
            })

        # Handle tool calls
        if 'tool_calls' in delta:
            tool_call = delta['tool_calls'][0]
            async for event in self._process_tool_call(tool_call, state):
                yield event

        # Handle finish reason
        if finish_reason:
            state['stop_reason'] = finish_reason
            
            # Stop active text block
            if state['active_text_block'] is not None:
                yield self._format_anthropic_sse('content_block_stop', {
                    'type': 'content_block_stop',
                    'index': state['active_text_block']
                })
                state['active_text_block'] = None
            
            # Stop active tool block  
            if state['active_tool_block'] is not None:
                yield self._format_anthropic_sse('content_block_stop', {
                    'type': 'content_block_stop', 
                    'index': state['active_tool_block']
                })
                state['active_tool_block'] = None

    async def _process_tool_call(self, tool_call: Dict[str, Any], state: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Process a single OpenAI tool call and emit appropriate Anthropic events."""
        # Check if this is a new tool call (has type: function)
        tool_id = tool_call.get('id')
        function_info = tool_call.get('function', {})
        tool_name = function_info.get('name')
        call_type = tool_call.get('type', '')
        
        if call_type == 'function':
            # Stop previous active blocks if starting new tool
            if state['active_text_block'] is not None:
                yield self._format_anthropic_sse('content_block_stop', {
                    'type': 'content_block_stop',
                    'index': state['active_text_block'] 
                })
                state['active_text_block'] = None
                
            if state['active_tool_block'] is not None:
                yield self._format_anthropic_sse('content_block_stop', {
                    'type': 'content_block_stop',
                    'index': state['active_tool_block']
                })
                
            # Start new tool block
            block_index = state['next_block_index']
            state['next_block_index'] += 1
            state['active_tool_block'] = block_index
            
            yield self._format_anthropic_sse('content_block_start', {
                'type': 'content_block_start',
                'index': block_index,
                'content_block': {
                    'type': 'tool_use',
                    'id': tool_id,
                    'name': tool_name,
                    'input': {}
                }
            })
        
        # Stream tool arguments
        arguments = function_info.get('arguments')
        if arguments is not None and state['active_tool_block'] is not None:
            yield self._format_anthropic_sse('content_block_delta', {
                'type': 'content_block_delta',
                'index': state['active_tool_block'],
                'delta': {
                    'type': 'input_json_delta',
                    'partial_json': arguments
                }
            })

    def _convert_stop_reason(self, openai_reason: Optional[str]) -> str:
        """Convert OpenAI finish reason to Anthropic stop reason."""
        if not openai_reason:
            return 'end_turn'
        return self.STOP_REASON_MAPPING.get(openai_reason, 'end_turn')

    def _convert_openai_usage(self, openai_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI usage format to Anthropic usage format."""
        if not openai_usage:
            return {
                'input_tokens': 0,
                'cache_creation_input_tokens': 0, 
                'cache_read_input_tokens': 0,
                'output_tokens': 0
            }

        anthropic_usage = {}
        
        # Basic token mapping
        if prompt_tokens := openai_usage.get('prompt_tokens'):
            anthropic_usage['input_tokens'] = prompt_tokens
        if completion_tokens := openai_usage.get('completion_tokens'):
            anthropic_usage['output_tokens'] = completion_tokens
            
        # Cache tokens from prompt_tokens_details
        if prompt_details := openai_usage.get('prompt_tokens_details', {}):
            if cached_tokens := prompt_details.get('cached_tokens'):
                anthropic_usage['cache_read_input_tokens'] = cached_tokens
        
        # Default missing fields
        anthropic_usage.setdefault('cache_creation_input_tokens', 0)
        anthropic_usage.setdefault('cache_read_input_tokens', 0)
        
        return anthropic_usage

    async def transform_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Transform non-streaming OpenAI response to Anthropic format.
        
        Note: This transformer is designed for streaming responses.
        For non-streaming, use the existing OpenAIResponseTransformer.
        """
        response = params['response']
        self.logger.warning('OpenAIToAnthropicSSETransformer used for non-streaming response')
        return response