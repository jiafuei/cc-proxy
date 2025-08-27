# Anthropic Messages API Streaming Response Schema

## Streaming Event Types
All streaming events use Server-Sent Events (SSE) format with `event:` and `data:` fields.

### Core Event Flow
1. **`message_start`** - Initial message with empty content
2. **Content blocks** (repeated for each content block):
   - `content_block_start` - Start of content block
   - `content_block_delta` (multiple) - Incremental content updates 
   - `content_block_stop` - End of content block
3. **`message_delta`** - Top-level message changes (stop_reason, usage)
4. **`message_stop`** - Final event indicating completion

### Event Types

#### Message Start Event
```json
{
  "type": "message_start",
  "message": {
    "id": "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY",
    "type": "message", 
    "role": "assistant",
    "content": [],
    "model": "claude-sonnet-4-20250514",
    "stop_reason": null,
    "stop_sequence": null,
    "usage": {"input_tokens": 25, "output_tokens": 1}
  }
}
```

#### Content Block Start Event
```json
{
  "type": "content_block_start",
  "index": 0,
  "content_block": {
    "type": "text|tool_use|thinking|server_tool_use",
    // type-specific fields
  }
}
```

#### Content Block Delta Event
```json
{
  "type": "content_block_delta",
  "index": 0,
  "delta": {
    "type": "text_delta|input_json_delta|thinking_delta|signature_delta",
    // delta-specific fields
  }
}
```

#### Content Block Stop Event
```json
{
  "type": "content_block_stop",
  "index": 0
}
```

#### Message Delta Event
```json
{
  "type": "message_delta",
  "delta": {
    "stop_reason": "end_turn|max_tokens|stop_sequence|tool_use",
    "stop_sequence": null
  },
  "usage": {"output_tokens": 15}
}
```

#### Message Stop Event
```json
{
  "type": "message_stop"
}
```

#### Ping Event
```json
{
  "type": "ping"
}
```

#### Error Event
```json
{
  "type": "error",
  "error": {
    "type": "overloaded_error|rate_limit_error|...",
    "message": "Error description"
  }
}
```

## Content Block Types

### Text Content Block
- **Start**: `{"type": "text", "text": ""}`
- **Delta**: `{"type": "text_delta", "text": "Hello"}`

### Tool Use Content Block
- **Start**: `{"type": "tool_use", "id": "toolu_...", "name": "tool_name", "input": {}}`
- **Delta**: `{"type": "input_json_delta", "partial_json": "{\"param\": \"val"}`

### Thinking Content Block (Extended Thinking)
- **Start**: `{"type": "thinking", "thinking": ""}`
- **Delta**: `{"type": "thinking_delta", "thinking": "Let me think..."}`
- **Signature**: `{"type": "signature_delta", "signature": "EqQBCgIYAh..."}`

### Server Tool Use Content Block
- **Start**: `{"type": "server_tool_use", "id": "srvtoolu_...", "name": "web_search", "input": {}}`
- **Delta**: `{"type": "input_json_delta", "partial_json": "{\"query\": \"..."}`

### Web Search Tool Result Block
- **Start**: `{"type": "web_search_tool_result", "tool_use_id": "...", "content": [...]}`
- **No deltas** - Results appear complete in start event

## Delta Types

### Text Delta
```json
{
  "type": "text_delta",
  "text": "incremental text content"
}
```

### Input JSON Delta
```json
{
  "type": "input_json_delta", 
  "partial_json": "{\"param\": \"partial value"
}
```

### Thinking Delta
```json
{
  "type": "thinking_delta",
  "thinking": "reasoning content"
}
```

### Signature Delta
```json
{
  "type": "signature_delta",
  "signature": "cryptographic signature for thinking block"
}
```

## Usage Tracking
- **Initial usage**: Appears in `message_start` event
- **Cumulative usage**: Updated in `message_delta` event
- **Final usage**: Complete token counts and server tool usage

## Potential Edge Cases
1. **Network Interruption**: Partial streams requiring recovery/resumption
2. **Tool JSON Parsing**: Accumulating partial JSON fragments across deltas
3. **Multiple Content Blocks**: Managing different block types in single response
4. **Thinking Block Integrity**: Signature validation for extended thinking
5. **Server Tool Results**: Handling large tool result payloads
6. **Error Recovery**: Resuming from partial assistant responses
7. **Event Ordering**: Ensuring proper sequence of start/delta/stop events
8. **Ping Event Handling**: Ignoring keepalive pings in processing
9. **Unknown Event Types**: Graceful handling of future event additions  
10. **Token Count Accuracy**: Cumulative vs incremental usage tracking
11. **Cache Usage Reporting**: Complex cache read/write token accounting
12. **Multi-server Tool Coordination**: MCP tool result synchronization
13. **Streaming Buffer Management**: Memory usage with large responses
14. **Content Block Index Tracking**: Ensuring deltas match correct blocks
15. **Partial JSON Recovery**: Tool use parameter reconstruction after interruption