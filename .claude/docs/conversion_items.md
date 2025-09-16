# Anthropic to Provider Conversion Checklist

## Basic Message Structure
- [ ] **Role mapping**: Convert `user`, `assistant`, `system` roles to target provider format
- [ ] **Message alternation**: Ensure user/assistant message alternation requirements are met
- [ ] **System message placement**: Handle system prompts (first message vs separate parameter)
- [ ] **Content structure**: Convert content arrays to target provider's text/multimodal format
- [ ] **Empty message handling**: Handle messages with empty or null content
- [ ] **Message ordering**: Preserve conversation flow and context

## Tool Usage & Function Calling
- [ ] **Tool definition schema**: Convert Anthropic tool definitions to target format (OpenAI functions, Google function declarations)
- [ ] **Tool use blocks**: Transform `tool_use` content blocks to provider-specific function call format
- [ ] **Tool result blocks**: Convert `tool_result` content blocks to function response format
- [ ] **Multiple tool calls**: Handle parallel tool execution in single assistant message
- [ ] **Tool call IDs**: Generate/map tool call identifiers across providers
- [ ] **Tool choice parameter**: Map `tool_choice` options (auto, any, specific tool)
- [ ] **Tool parameters**: Validate and convert tool parameter schemas (JSON Schema differences)
- [ ] **Tool error handling**: Handle tool execution errors and timeouts

## Thinking Modes & Internal Reasoning
- [ ] **Thinking mode support**: Check if target provider supports thinking/reasoning modes
- [ ] **Thinking block extraction**: Strip or preserve thinking content based on provider capability
- [ ] **Interleaved thinking**: Handle thinking blocks between regular content
- [ ] **Auto thinking**: Convert automatic reasoning to provider-specific patterns
- [ ] **Thinking visibility**: Ensure thinking content is properly hidden from end users
- [ ] **Reasoning chains**: Preserve logical flow when converting thinking to other formats

## Caching Mechanisms
- [ ] **Cache control support**: Check if target provider supports prompt caching
- [ ] **Cache breakpoints**: Handle `cache_control` blocks for optimization
- [ ] **Ephemeral vs persistent**: Map caching types to provider capabilities
- [ ] **Cache fallback**: Remove cache controls for non-supporting providers
- [ ] **Cache key generation**: Generate appropriate cache identifiers
- [ ] **Cache invalidation**: Handle cache refresh scenarios

## Multimodal Content
- [ ] **Image support**: Verify target provider supports image inputs
- [ ] **Image encoding**: Convert between base64, URL, and file reference formats
- [ ] **Media type mapping**: Transform MIME types to provider-specific formats
- [ ] **Image size limits**: Validate and resize images for provider constraints
- [ ] **Format support**: Convert between supported image formats (JPEG, PNG, GIF, WebP)
- [ ] **Multiple images**: Handle multiple images in single message
- [ ] **Image-text interleaving**: Preserve image placement within text content

## Streaming & Response Handling
- [ ] **Streaming protocol**: Convert SSE format to provider-specific streaming
- [ ] **Event type mapping**: Transform message_start, content_block_delta, etc.
- [ ] **Partial content**: Handle incremental content delivery
- [ ] **Stream completion**: Properly close and finalize streaming responses
- [ ] **Error events**: Convert streaming error events
- [ ] **Non-streaming fallback**: Support providers without streaming capability
- [ ] **Stop sequences**: Handle custom stop sequence configurations

## Model Parameters & Configuration
- [ ] **Model name mapping**: Convert Anthropic model names to provider equivalents
- [ ] **Parameter mapping**: Transform max_tokens, temperature, top_p, top_k
- [ ] **Parameter ranges**: Validate and clamp values to provider-supported ranges
- [ ] **Unsupported parameters**: Handle provider-specific limitations gracefully
- [ ] **Default values**: Apply appropriate defaults for missing parameters
- [ ] **Model capabilities**: Check target model supports required features (tools, multimodal)
- [ ] **Context length**: Validate message length against model limits

## Authentication & Headers
- [ ] **API key format**: Convert between header formats (x-api-key vs Authorization)
- [ ] **Authentication method**: Handle different auth schemes (API key, OAuth, subscription key)
- [ ] **Version headers**: Map anthropic-version to provider-specific versioning
- [ ] **User-agent strings**: Set appropriate user-agent for target provider
- [ ] **Content-type**: Ensure correct content-type headers
- [ ] **Custom headers**: Handle provider-specific required headers
- [ ] **Rate limiting**: Implement provider-specific rate limit handling

## Error Handling & Status Codes
- [ ] **Error format mapping**: Convert error response structures
- [ ] **HTTP status codes**: Map status codes between providers
- [ ] **Error type classification**: Transform error categories (auth, rate limit, validation)
- [ ] **Error message translation**: Provide meaningful error messages
- [ ] **Retry logic**: Implement appropriate retry mechanisms
- [ ] **Quota exceeded**: Handle usage limit errors
- [ ] **Service unavailable**: Handle provider downtime scenarios

## Edge Cases & Special Handling
- [ ] **Very long messages**: Handle context length overflow
- [ ] **Unicode support**: Ensure proper character encoding
- [ ] **Special characters**: Handle markdown, code blocks, formatting
- [ ] **Empty responses**: Handle cases where model returns no content
- [ ] **Malformed inputs**: Validate and sanitize input data
- [ ] **Backwards compatibility**: Support older API versions if needed
- [ ] **Provider-specific quirks**: Handle unique provider behaviors
- [ ] **Fallback providers**: Implement graceful degradation

## Testing & Validation
- [ ] **Unit tests**: Test each conversion component independently
- [ ] **Integration tests**: Test full conversion pipelines
- [ ] **Provider compatibility**: Test against all target provider APIs
- [ ] **Edge case coverage**: Test with unusual inputs and scenarios
- [ ] **Performance testing**: Validate conversion speed and memory usage
- [ ] **Error scenario testing**: Test all error handling paths
- [ ] **Regression testing**: Ensure changes don't break existing functionality

## Documentation & Monitoring
- [ ] **Conversion logs**: Log all transformations for debugging
- [ ] **Provider metrics**: Track success/failure rates per provider
- [ ] **Compatibility matrix**: Document feature support across providers
- [ ] **Migration guides**: Provide guidance for switching providers
- [ ] **API differences**: Document key differences between providers
- [ ] **Troubleshooting guides**: Common issues and solutions