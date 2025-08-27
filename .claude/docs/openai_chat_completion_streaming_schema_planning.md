# OpenAI Chat Completion Streaming Response Schema

## Streaming Event Types
All streaming events use Server-Sent Events (SSE) format with `data:` fields containing JSON chunks.

### Core Event Flow
1. **Initial chunk** - Chat completion with empty/minimal content
2. **Content deltas** (repeated for incremental updates):
   - Choice delta chunks with content updates
   - Function call argument streaming (if applicable)
3. **Final chunk** - Completion with finish reason and final usage
4. **Termination** - `data: [DONE]` marker indicating stream end

### Event Structure

#### Chat Completion Chunk
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1677652288,
  "model": "gpt-4",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "Hello"
      },
      "logprobs": null,
      "finish_reason": null
    }
  ],
  "usage": null
}
```

#### Delta Object Types

##### Text Content Delta
```json
{
  "role": "assistant",
  "content": "incremental text content"
}
```

##### Function Call Delta
```json
{
  "tool_calls": [
    {
      "index": 0,
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"San"
      }
    }
  ]
}
```

##### Role Assignment Delta (First Chunk)
```json
{
  "role": "assistant"
}
```

#### Choice Object
```json
{
  "index": 0,
  "delta": {
    // Delta object content
  },
  "logprobs": {
    "content": [
      {
        "token": "Hello",
        "logprob": -0.31725305,
        "bytes": [72, 101, 108, 108, 111],
        "top_logprobs": []
      }
    ]
  },
  "finish_reason": "stop|length|function_call|tool_calls|content_filter"
}
```

#### Final Chunk with Usage
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1677652288,
  "model": "gpt-4",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [
    {
      "index": 0,
      "delta": {},
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 150,
    "total_tokens": 175
  }
}
```

#### Stream Termination
```
data: [DONE]
```

## Content Types

### Text Streaming
- **First chunk**: Contains role assignment (`"role": "assistant"`)
- **Content chunks**: Incremental text in `content` field
- **Final chunk**: Empty delta with `finish_reason`

### Function/Tool Call Streaming
- **Start chunk**: Tool call initiation with `id`, `type`, and `name`
- **Argument chunks**: Incremental JSON in `arguments` field
- **Completion chunk**: Empty delta with `finish_reason: "tool_calls"`

### Multi-Choice Responses
- Each choice streams independently with separate `index` values
- Choices can finish at different times with individual `finish_reason`

## Finish Reasons

- **`stop`** - Natural completion or stop sequence reached
- **`length`** - Maximum token limit reached
- **`function_call`** - Legacy function calling (deprecated)
- **`tool_calls`** - Tool/function calls completed
- **`content_filter`** - Content filtered by safety systems

## Usage Tracking
- **No usage in intermediate chunks** - Usage only appears in final chunk
- **Complete token counts** - Prompt, completion, and total tokens
- **Model-specific usage** - Different models may have additional usage fields

## LogProbs Structure
```json
{
  "content": [
    {
      "token": "text_token",
      "logprob": -0.31725305,
      "bytes": [72, 101, 108, 108, 111],
      "top_logprobs": [
        {
          "token": "alternative",
          "logprob": -1.2345,
          "bytes": [65, 108, 116]
        }
      ]
    }
  ]
}
```

## Potential Edge Cases

1. **Network Interruption**: Partial streams requiring reconnection and resumption
2. **Function Argument Parsing**: Accumulating partial JSON across multiple chunks
3. **Multiple Tool Calls**: Managing concurrent tool call streaming in single response
4. **Choice Index Management**: Ensuring proper choice tracking in multi-choice scenarios
5. **Token Limit Scenarios**: Handling `length` finish reason with incomplete responses
6. **Content Filtering**: Processing filtered content and appropriate error handling
7. **Large Function Arguments**: Managing memory usage with large tool parameters
8. **LogProbs Processing**: Handling optional log probability data efficiently
9. **System Fingerprint Tracking**: Monitoring model version consistency across chunks
10. **Rate Limit Recovery**: Graceful handling of rate limit errors during streaming
11. **Partial JSON Recovery**: Reconstructing function arguments after interruption
12. **Choice Completion Timing**: Handling choices that finish at different rates
13. **Streaming Buffer Management**: Memory efficiency with long responses
14. **Tool Call ID Consistency**: Ensuring tool call IDs remain consistent across chunks
15. **Delta Accumulation**: Properly concatenating incremental content updates
16. **Empty Delta Handling**: Processing chunks with empty delta objects
17. **Model Switching**: Handling potential model changes mid-stream
18. **Usage Calculation Accuracy**: Ensuring final token counts match actual usage
19. **LogProbs Byte Encoding**: Proper handling of multi-byte character tokens
20. **Error Recovery Patterns**: Resuming from partial assistant responses

## Key Differences from Anthropic Streaming

### Event Structure
- **OpenAI**: Single event type (`chat.completion.chunk`) with varying delta content
- **Anthropic**: Multiple distinct event types (`message_start`, `content_block_delta`, etc.)

### Content Organization
- **OpenAI**: Content deltas within choice objects
- **Anthropic**: Separate content block start/delta/stop events with indexing

### Usage Reporting
- **OpenAI**: Usage only in final chunk
- **Anthropic**: Usage in both initial and delta events

### Tool Calling
- **OpenAI**: Streaming tool arguments as partial JSON strings
- **Anthropic**: Structured tool use events with proper JSON delta parsing

### Multi-Response Handling
- **OpenAI**: Multiple choices with independent streaming
- **Anthropic**: Single message with multiple content blocks