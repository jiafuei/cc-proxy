# OpenAI Chat Completion Non-Streaming Response Schema

## Core Response Structure

The OpenAI Chat Completion API returns a complete response object for non-streaming requests.

### Main Response Object
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4o",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [...],
  "usage": {...}
}
```

### Response Fields

| Field | Type | Description | Always Present |
|-------|------|-------------|----------------|
| `id` | string | Unique completion identifier | ✅ |
| `object` | string | Always "chat.completion" | ✅ |
| `created` | integer | Unix timestamp of creation | ✅ |
| `model` | string | Model used for generation | ✅ |
| `system_fingerprint` | string | Model version identifier | ✅ |
| `choices` | array | Array of completion choices | ✅ |
| `usage` | object | Token usage information | ✅ |

## Choice Object Structure

Each choice in the choices array represents a potential response:

```json
{
  "index": 0,
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help you today?",
    "tool_calls": null
  },
  "logprobs": null,
  "finish_reason": "stop"
}
```

### Choice Fields

| Field | Type | Description | Always Present |
|-------|------|-------------|----------------|
| `index` | integer | Choice index (0-based) | ✅ |
| `message` | object | Assistant message object | ✅ |
| `logprobs` | object\|null | Log probability information | ✅ |
| `finish_reason` | string | Reason for completion | ✅ |

## Message Object Types

### Text Response Message
```json
{
  "role": "assistant",
  "content": "Paris is the capital of France.",
  "tool_calls": null
}
```

### Function Call Response Message
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"Boston, MA\"}"
      }
    }
  ]
}
```

### Mixed Content Response Message
```json
{
  "role": "assistant", 
  "content": "I'll check the weather for you.",
  "tool_calls": [
    {
      "id": "call_def456",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"Boston, MA\"}"
      }
    }
  ]
}
```


## Tool Call Structure

### Tool Call Object
```json
{
  "id": "call_abc123",
  "type": "function",
  "function": {
    "name": "function_name",
    "arguments": "{\"param\": \"value\"}"
  }
}
```

### Tool Call Fields

| Field | Type | Description | Notes |
|-------|------|-------------|--------|
| `id` | string | Unique call identifier | Usually format: `call_<alphanumeric>` |
| `type` | string | Always "function" | Only function calls currently supported |
| `function.name` | string | Function name to call | Must match tool definition |
| `function.arguments` | string | JSON string of arguments | **Not a JSON object** |

## Finish Reasons

### Standard Finish Reasons

| Reason | Description | Context |
|--------|-------------|---------|
| `stop` | Natural completion or stop sequence | Normal conversation end |
| `length` | Maximum token limit reached | Response was truncated |
| `tool_calls` | Function calls were made | Tools need to be executed |
| `content_filter` | Content filtered by safety systems | Policy violation detected |

### Legacy Finish Reason
| Reason | Description | Status |
|--------|-------------|--------|
| `function_call` | Legacy function calling | Deprecated, use `tool_calls` |

## Usage Object

### Standard Usage
```json
{
  "prompt_tokens": 56,
  "completion_tokens": 31,
  "total_tokens": 87
}
```

### Extended Usage (Some Models)
```json
{
  "prompt_tokens": 56,
  "completion_tokens": 31, 
  "total_tokens": 87,
  "prompt_tokens_details": {
    "cached_tokens": 12
  },
  "completion_tokens_details": {
    "reasoning_tokens": 15
  }
}
```

### Usage Fields

| Field | Type | Description | Always Present |
|-------|------|-------------|----------------|
| `prompt_tokens` | integer | Input tokens consumed | ✅ |
| `completion_tokens` | integer | Output tokens generated | ✅ |
| `total_tokens` | integer | Sum of prompt + completion | ✅ |
| `prompt_tokens_details` | object | Detailed prompt token breakdown | ❌ (Some models) |
| `completion_tokens_details` | object | Detailed completion breakdown | ❌ (o1 models) |

## LogProbs Structure

When `logprobs: true` is specified in the request:

```json
{
  "logprobs": {
    "content": [
      {
        "token": "Hello",
        "logprob": -0.31725305,
        "bytes": [72, 101, 108, 108, 111],
        "top_logprobs": [
          {
            "token": "Hi",
            "logprob": -1.3862944,
            "bytes": [72, 105]
          }
        ]
      }
    ]
  }
}
```

### LogProbs Fields

| Field | Type | Description |
|-------|------|-------------|
| `content` | array | Array of token probability information |
| `token` | string | The actual token |
| `logprob` | number | Log probability of the token |
| `bytes` | array | UTF-8 byte representation |
| `top_logprobs` | array | Alternative tokens with probabilities |

## Response Examples

### Simple Text Response
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4o",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Paris is the capital of France."
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 11,
    "completion_tokens": 7,
    "total_tokens": 18
  }
}
```

### Function Call Response
```json
{
  "id": "chatcmpl-456",
  "object": "chat.completion", 
  "created": 1677652289,
  "model": "gpt-4o",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"Boston, MA\", \"unit\": \"fahrenheit\"}"
            }
          }
        ]
      },
      "logprobs": null,
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 82,
    "completion_tokens": 18,
    "total_tokens": 100
  }
}
```

### Multiple Choice Response
```json
{
  "id": "chatcmpl-789",
  "object": "chat.completion",
  "created": 1677652290,
  "model": "gpt-4o", 
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Option 1: Paris is the capital of France."
      },
      "logprobs": null,
      "finish_reason": "stop"
    },
    {
      "index": 1, 
      "message": {
        "role": "assistant",
        "content": "Option 2: The capital city of France is Paris."
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 11,
    "completion_tokens": 24,
    "total_tokens": 35
  }
}
```

### JSON Mode Response
```json
{
  "id": "chatcmpl-json",
  "object": "chat.completion",
  "created": 1677652291,
  "model": "gpt-4o",
  "system_fingerprint": "fp_44709d6fcb", 
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"capital\": \"Paris\", \"country\": \"France\", \"population\": 2161000}"
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 35,
    "completion_tokens": 20,
    "total_tokens": 55
  }
}
```

### LogProbs Response
```json
{
  "id": "chatcmpl-logprobs",
  "object": "chat.completion",
  "created": 1677652292,
  "model": "gpt-4o",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant", 
        "content": "Hello there!"
      },
      "logprobs": {
        "content": [
          {
            "token": "Hello",
            "logprob": -0.31725305,
            "bytes": [72, 101, 108, 108, 111],
            "top_logprobs": [
              {
                "token": "Hi",
                "logprob": -1.3862944,
                "bytes": [72, 105]
              }
            ]
          },
          {
            "token": " there",
            "logprob": -0.02380908,
            "bytes": [32, 116, 104, 101, 114, 101],
            "top_logprobs": [
              {
                "token": "!",
                "logprob": -2.3025851,
                "bytes": [33]
              }
            ]
          },
          {
            "token": "!",
            "logprob": -0.04566316,
            "bytes": [33],
            "top_logprobs": []
          }
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 3,
    "total_tokens": 12
  }
}
```

### Reasoning Response (o1 Models)
```json
{
  "id": "chatcmpl-o1-reasoning",
  "object": "chat.completion",
  "created": 1677652293,
  "model": "o1-preview",
  "system_fingerprint": "fp_o1_reasoning",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Looking at this step by step:\n\n1. First segment: 120 km in 1.5 hours\n   Speed = 120 ÷ 1.5 = 80 km/h\n\n2. Second segment: speeds up by 20 km/h for 2 hours\n   New speed = 80 + 20 = 100 km/h\n   Distance = 100 × 2 = 200 km\n\n3. Total distance = 120 + 200 = 320 km"
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 95,
    "total_tokens": 140,
    "completion_tokens_details": {
      "reasoning_tokens": 15
    }
  }
}
```

## Error Response Structure

### API Error Response
```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "invalid_parameter",
    "message": "Invalid value for 'temperature': must be between 0 and 2",
    "param": "temperature"
  }
}
```

### Error Fields
| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Error category |
| `code` | string | Specific error code |
| `message` | string | Human-readable error description |
| `param` | string | Parameter that caused the error |

### Common Error Types
- `invalid_request_error` - Request validation failed
- `authentication_error` - Invalid API key
- `permission_error` - Insufficient permissions
- `not_found_error` - Resource not found
- `rate_limit_error` - Rate limit exceeded
- `api_error` - Server-side error
- `overloaded_error` - Server temporarily overloaded

## Potential Edge Cases

### 1. Content and Tool Call Combinations
```json
// ✅ Valid - content with tool calls
{
  "role": "assistant",
  "content": "I'll search for that information.",
  "tool_calls": [...]
}

// ✅ Valid - null content with tool calls
{
  "role": "assistant", 
  "content": null,
  "tool_calls": [...]
}

// ❌ Invalid - both null/empty
{
  "role": "assistant",
  "content": null,
  "tool_calls": null
}
```

### 2. Token Limit Edge Cases
- **Length finish reason**: Response truncated due to `max_tokens`
- **Context limit**: Total tokens exceed model context window
- **Token counting accuracy**: Slight variations between estimated vs actual usage

### 3. Function Call Argument Issues
```json
// ❌ Invalid JSON in arguments
{
  "function": {
    "name": "get_weather",
    "arguments": "{invalid json"
  }
}

// ❌ Model hallucinated parameters
{
  "function": {
    "name": "get_weather", 
    "arguments": "{\"nonexistent_param\": \"value\"}"
  }
}
```

### 4. Multiple Choice Edge Cases
- **Different finish reasons**: Choices can have different completion states
- **Varying content types**: Some choices text, others with tool calls
- **Usage calculation**: Total usage across all choices

### 5. LogProbs Edge Cases
- **Large logprobs data**: Memory usage with `top_logprobs: 5`
- **Unicode handling**: Multi-byte character token representation
- **Token boundary issues**: Subword tokenization artifacts

### 6. System Fingerprint Changes
- **Mid-conversation changes**: Model updates during extended conversations
- **Consistency tracking**: Ensuring same model version across related calls

### 7. Model-Specific Variations
- **o1 models**: Additional reasoning tokens in usage (internal reasoning not exposed)
- **Legacy models**: Deprecated function calling format
- **Fine-tuned models**: Custom model identifiers

### 8. Response Size Limits
- **Large responses**: Memory constraints with extensive content
- **Tool call volume**: Multiple simultaneous function calls
- **LogProbs overhead**: Additional data size with probability information

### 9. Encoding and Character Handling
- **UTF-8 compliance**: Proper encoding of international characters  
- **Special tokens**: Model-specific special tokens in responses
- **Byte representation**: Accurate byte arrays in logprobs

### 10. Timing and Concurrency
- **Response timing**: Created timestamp accuracy and timezone handling
- **Concurrent requests**: ID uniqueness across parallel calls
- **Request-response correlation**: Matching responses to original requests

### 11. Content Filtering Edge Cases
- **Partial filtering**: Content modified vs completely blocked
- **Filter trigger points**: Specific content that triggers filtering
- **False positives**: Legitimate content incorrectly flagged

### 12. JSON Mode Validation
- **Invalid JSON output**: Model produces malformed JSON despite instruction
- **Schema compliance**: JSON structure matching expected format
- **Nested complexity**: Deep object structures and array handling

## Key Differences from Anthropic Responses

### Structure Differences
- **OpenAI**: Single message object per choice with optional tool_calls array
- **Anthropic**: Content blocks array with different block types

### Usage Reporting
- **OpenAI**: Simple token counts with optional reasoning breakdown
- **Anthropic**: Detailed cache usage and server tool statistics

### Tool Handling
- **OpenAI**: Tool calls as structured objects within message
- **Anthropic**: Separate tool use content blocks with results

### Content Organization
- **OpenAI**: String content with separate tool_calls array
- **Anthropic**: Mixed content blocks including text, thinking, and tool use

### Error Handling
- **OpenAI**: Structured error objects with specific error codes
- **Anthropic**: Error information within streaming events or HTTP status codes