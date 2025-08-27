# Anthropic Messages API Schema Reference

## Core Required Fields
- **`model`** (string): AI model identifier (e.g., "claude-sonnet-4-20250514")
- **`messages`** (array): Conversation history with role/content pairs
- **`max_tokens`** (integer): Maximum tokens to generate (≥1)

## Core Optional Fields
- **`system`** (string|array): System prompt/instructions - can be string or array of text blocks
- **`temperature`** (number): Randomness control (0.0-1.0)
- **`stream`** (boolean): Enable streaming response
- **`stop_sequences`** (array of strings): Custom stop tokens
- **`tools`** (array): Tool definitions with name, description, input_schema
- **`tool_choice`** (string|object): Tool usage control ("auto", "any", "none", or specific tool)

## Advanced Fields
- **`thinking`** (boolean): Enable extended reasoning mode
- **`top_k`** (integer): Token sampling parameter
- **`top_p`** (number): Nucleus sampling parameter
- **`metadata`** (object): Additional request metadata
- **`service_tier`** (string): Request priority level

## Message Content Structure
Messages support complex content blocks:
- **Text blocks**: `{type: "text", text: "..."}`
- **Image blocks**: `{type: "image", source: {type: "base64", media_type: "image/jpeg|png|gif|webp", data: "..."}}`
- **Tool use blocks**: `{type: "tool_use", name: "...", input: {...}}`
- **Tool result blocks**: `{type: "tool_result", tool_use_id: "...", content: "..."}`
- **Thinking blocks**: `{type: "thinking", content: "..."}` (internal reasoning)

## Potential Edge Cases
1. **Cache Control**: Ephemeral cache breakpoints on any content block (`cache_control: {type: "ephemeral"}`)
2. **Mixed Content Types**: Messages can mix text, images, tool calls, and results in single content array
3. **Image Processing**: Base64 encoding/decoding, media type validation, size limits
4. **Image Caching**: Large images with cache breakpoints for optimization
5. **System Message Formats**: System can be string or structured array format
6. **Tool Schema Complexity**: Tools use JSON Schema for input validation
7. **MCP Tool Prefixes**: Tools with `mcp__` prefixes need special handling
8. **Empty/Null Content**: Handling of empty messages or null content blocks
9. **Unicode/Encoding**: Special character handling in text content
10. **Token Limits**: Max tokens interaction with content length validation (images count as tokens)
11. **Streaming Interruption**: Partial responses and reconnection scenarios
12. **Role Validation**: Strict alternating user/assistant pattern enforcement
13. **Image Format Support**: Different providers support different image formats/sizes
14. **Multi-modal Context**: Images affecting token count and context window usage