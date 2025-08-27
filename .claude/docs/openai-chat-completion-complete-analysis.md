# OpenAI Create Chat Completion API - Complete Schema Analysis

*Comprehensive field reference, conversion guide, and implementation notes*

## Table of Contents

1. [Overview](#overview)
2. [Complete Field Reference](#complete-field-reference)
3. [Message Structure Deep Dive](#message-structure-deep-dive)
4. [Tool Calling Comprehensive Guide](#tool-calling-comprehensive-guide)
5. [Schema Conversion Mapping](#schema-conversion-mapping)
6. [Critical Edge Cases](#critical-edge-cases)
7. [Validation Rules](#validation-rules)
8. [Request Examples](#request-examples)
9. [Best Practices](#best-practices)

## Overview

The OpenAI Create Chat Completion API (`POST /v1/chat/completions`) is the primary endpoint for generating conversational AI responses. This analysis covers all 26+ request parameters, their importance for schema conversion, and critical implementation considerations.

## Complete Field Reference

### Core Required Fields

| Field | Type | Required | Default | Description | Conversion Priority |
|-------|------|----------|---------|-------------|-------------------|
| `model` | string | ✅ | - | Model ID to use (e.g., gpt-4o, gpt-3.5-turbo) | **CRITICAL** |
| `messages` | array | ✅ | - | Conversation messages array (min: 1) | **CRITICAL** |

### Generation Control Fields

| Field | Type | Required | Default | Range | Description | Conversion Priority |
|-------|------|----------|---------|-------|-------------|-------------------|
| `temperature` | number | ❌ | 1 | 0-2 | Sampling temperature | **HIGH** |
| `top_p` | number | ❌ | 1 | 0-1 | Nucleus sampling | **HIGH** |
| `max_tokens` | integer\|null | ❌ | null | ≥1 | Maximum response tokens (deprecated, use `max_completion_tokens`) | **HIGH** |
| `max_completion_tokens` | integer\|null | ❌ | null | ≥1 | Maximum response tokens (preferred over `max_tokens`) | **HIGH** |
| `n` | integer | ❌ | 1 | 1-128 | Number of completions | **MEDIUM** |

### Behavior Modification Fields

| Field | Type | Required | Default | Range | Description | Conversion Priority |
|-------|------|----------|---------|-------|-------------|-------------------|
| `frequency_penalty` | number | ❌ | 0 | -2.0 to 2.0 | Penalize frequent tokens | **MEDIUM** |
| `presence_penalty` | number | ❌ | 0 | -2.0 to 2.0 | Penalize present tokens | **MEDIUM** |
| `logit_bias` | object | ❌ | {} | Token→bias map | Token probability biasing | **LOW** |
| `stop` | string\|array | ❌ | null | Max 4 items | Stop sequences | **HIGH** |

### Tool Calling Fields

| Field | Type | Required | Default | Description | Conversion Priority |
|-------|------|----------|---------|-------------|-------------------|
| `tools` | array | ❌ | [] | Function definitions | **CRITICAL** |
| `tool_choice` | string\|object | ❌ | "auto" | Tool selection strategy | **CRITICAL** |

### Response Control Fields

| Field | Type | Required | Default | Description | Conversion Priority |
|-------|------|----------|---------|-------------|-------------------|
| `response_format` | object | ❌ | {"type":"text"} | Output format control | **MEDIUM** |
| `stream` | boolean | ❌ | false | Enable streaming | **HIGH** |
| `stream_options` | object | ❌ | null | Streaming configuration options | **MEDIUM** |
| `reasoning_effort` | string | ❌ | null | Reasoning effort level for o-series and gpt-5 models | **HIGH** |
| `user` | string | ❌ | - | End-user identifier | **LOW** |
| `seed` | integer\|null | ❌ | null | Deterministic sampling | **LOW** |

### Advanced Fields

| Field | Type | Required | Default | Range | Description | Conversion Priority |
|-------|------|----------|---------|-------|-------------|-------------------|
| `logprobs` | boolean | ❌ | false | - | Return log probabilities | **LOW** |
| `top_logprobs` | integer\|null | ❌ | null | 0-5 | Top token probabilities | **LOW** |
| `service_tier` | string | ❌ | "auto" | auto/default/scale | Service tier (gpt-4o/gpt-4-turbo only) | **LOW** |

### Deprecated Fields (Legacy Support)

| Field | Type | Description | Modern Replacement |
|-------|------|-------------|-------------------|
| `function_call` | string\|object | Legacy function calling | Use `tool_choice` |
| `functions` | array | Legacy function definitions | Use `tools` |

## Message Structure Deep Dive

### Message Types Overview

The `messages` array supports four distinct message role types, each with specific schema requirements:

```json
{
  "messages": [
    {"role": "system", "content": "..."},     // SystemMessage
    {"role": "user", "content": "..."},       // UserMessage  
    {"role": "assistant", "content": "..."},  // AssistantMessage
    {"role": "tool", "content": "..."}        // ToolMessage
  ]
}
```

### 1. SystemMessage Schema

**Purpose**: Guide model behavior and provide context

```json
{
  "role": "system",                    // Required: must be "system"
  "content": "You are a helpful AI...", // Required: string
  "name": "assistant_persona"         // Optional: participant name
}
```

**Validation Rules**:
- `role`: Must be exactly "system"
- `content`: Required string, cannot be null
- `name`: Optional, pattern `^[a-zA-Z0-9_]{1,64}$`

### 2. UserMessage Schema

**Purpose**: User input, queries, and multimodal content including images

```json
{
  "role": "user",                      // Required: must be "user"
  "content": [                         // Can be string or array for multimodal
    {
      "type": "text",
      "text": "What is in this image?"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/jpeg;base64,..." // Base64 encoded image or URL
      }
    }
  ],
  "name": "user_123"                   // Optional: user identifier
}
```

**Validation Rules**:
- `role`: Must be exactly "user" 
- `content`: Can be string (text-only) or array (multimodal with text and images)
- For images: `image_url.url` supports base64 data URIs or HTTPS URLs
- `name`: Optional, pattern `^[a-zA-Z0-9_]{1,64}$`

### 3. AssistantMessage Schema

**Purpose**: AI responses and function calls

```json
{
  "role": "assistant",                 // Required: must be "assistant"
  "content": "The weather is...",      // Optional: can be null if tool_calls present
  "name": "my_assistant",              // Optional: assistant name
  "tool_calls": [...],                 // Optional: function calls (see Tool Calling section)
  "function_call": {...}               // Deprecated: use tool_calls instead
}
```

**Critical Rules**:
- `content` can be `null` when `tool_calls` is present
- Either `content` OR `tool_calls` must be present (not both null)
- `tool_calls` is the modern approach, `function_call` is deprecated

### 4. ToolMessage Schema  

**Purpose**: Results from tool/function execution

```json
{
  "role": "tool",                      // Required: must be "tool"
  "content": "Temperature: 72°F",      // Required: tool output as string
  "tool_call_id": "call_abc123"        // Required: references specific tool call
}
```

**Critical Rules**:
- `tool_call_id` must reference an existing tool call ID from assistant message
- `content` contains the actual tool execution result
- Must follow assistant message with `tool_calls`

### Message Flow Patterns

```
User → Assistant → Tool → Assistant → User
 ↓         ↓        ↓         ↓        ↓
user → assistant → tool → assistant → user
      (tool_calls) (result) (response)
```

## Tool Calling Comprehensive Guide

### Tools Array Structure

The `tools` field defines available functions:

```json
{
  "tools": [
    {
      "type": "function",              // Required: currently only "function" supported
      "function": {
        "name": "get_weather",         // Required: function name
        "description": "Get weather",  // Optional but recommended
        "parameters": {                // Required: JSON Schema
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name"
            },
            "unit": {
              "type": "string", 
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

### Tool Choice Strategies

**String Values**:
```json
{"tool_choice": "none"}     // Never call functions
{"tool_choice": "auto"}     // Model decides (default when tools provided)
{"tool_choice": "required"} // Must call at least one function
```

**Specific Function Forcing**:
```json
{
  "tool_choice": {
    "type": "function",
    "function": {
      "name": "get_weather"
    }
  }
}
```

### Tool Calls in Assistant Messages

When model decides to call functions:

```json
{
  "role": "assistant",
  "content": null,                     // Often null when tool_calls present
  "tool_calls": [
    {
      "id": "call_abc123",             // Unique identifier
      "type": "function",              // Currently always "function"
      "function": {
        "name": "get_weather",         // Function to call
        "arguments": "{\"location\": \"Boston\"}" // JSON string (not object!)
      }
    }
  ]
}
```

### Tool Response Pattern

Complete tool calling conversation:

```json
[
  {"role": "user", "content": "What's the weather in Boston?"},
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function", 
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Boston\"}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "content": "Temperature: 72°F, Sunny",
    "tool_call_id": "call_abc123"
  },
  {
    "role": "assistant",
    "content": "The weather in Boston is currently 72°F and sunny!"
  }
]
```

## Schema Conversion Mapping

### Claude → OpenAI Field Mappings

| Claude Field | OpenAI Field | Conversion Logic | Notes |
|--------------|--------------|------------------|--------|
| `model` | `model` | Map model names | See model mapping table |
| `messages` | `messages` | Complex conversion | Handle content blocks |
| `system` | `messages[0]` | Inject as system message | Prepend to messages array |
| `max_tokens` | `max_completion_tokens` | Direct copy | **DEPRECATED**: Use `max_completion_tokens` |
| `temperature` | `temperature` | Direct copy | Same range (0-2) |
| `stop_sequences` | `stop` | Array truncation | Max 4 sequences in OpenAI |
| `stream` | `stream` | Direct copy | Boolean flag |
| `tools` | `tools` | Convert structure | Different schema format |
| `tool_choice` | `tool_choice` | Map values | "any" → "required" |
| N/A | `reasoning_effort` | Claude doesn't support | o1 models only, controls reasoning depth |
| N/A | `stream_options` | Claude doesn't support | Streaming configuration |

### Model Name Mapping

```javascript
const modelMapping = {
  'claude-3-sonnet-20240229': 'gpt-4',
  'claude-3-haiku-20240307': 'gpt-3.5-turbo',
  'claude-3-opus-20240229': 'gpt-4-turbo',
  'claude-3-5-sonnet-20241022': 'gpt-4',
  'claude-3-5-haiku-20241022': 'gpt-3.5-turbo',
  'claude-sonnet-4-20250514': 'gpt-4o',
  'claude-3-5-sonnet-20250121': 'gpt-4o'
};
```

### Content Block Conversion

**Claude Content Blocks → OpenAI String**:

```javascript
// Claude format
{
  "content": [
    {"type": "text", "text": "Hello"},
    {"type": "tool_use", "name": "calculator", "input": {"a": 5}}
  ]
}

// OpenAI format  
{
  "content": "Hello\n[Tool: calculator with input: {\"a\": 5}]"
}
```

### Tool Choice Conversion

| Claude Value | OpenAI Value | Notes |
|-------------|--------------|--------|
| `"auto"` | `"auto"` | Direct mapping |
| `"any"` | `"required"` | Semantic difference |
| `"none"` | `"none"` | Direct mapping |
| `{"type": "tool", "name": "X"}` | `{"type": "function", "function": {"name": "X"}}` | Structure change |

## Critical Edge Cases

### 1. Message Content Edge Cases

**Empty Messages**:
```json
{"role": "user", "content": ""}  // ❌ Invalid - empty content
{"role": "assistant", "content": null, "tool_calls": []} // ❌ Invalid - no content or tool_calls
```

**Assistant Message Variations**:
```json
// ✅ Valid - content only
{"role": "assistant", "content": "Hello"}

// ✅ Valid - tool_calls only  
{"role": "assistant", "content": null, "tool_calls": [...]}

// ✅ Valid - both content and tool_calls
{"role": "assistant", "content": "I'll check that", "tool_calls": [...]}
```

### 2. Tool Call Edge Cases

**Invalid Tool Call IDs**:
```json
// ❌ Tool message without matching tool_call_id
{
  "role": "tool",
  "content": "Result",
  "tool_call_id": "nonexistent_id"  // No matching tool call
}
```

**Malformed Arguments**:
```json
{
  "function": {
    "name": "get_weather", 
    "arguments": "invalid json"  // ❌ Invalid JSON string
  }
}
```

**Model Hallucinated Parameters**:
```json
{
  "function": {
    "name": "get_weather",
    "arguments": "{\"nonexistent_param\": \"value\"}" // Model invented parameter
  }
}
```

### 3. Token Limit Edge Cases

```javascript
// ❌ Exceeds context window
const promptTokens = 8000;
const maxTokens = 4000;
const modelContextLimit = 8192; // GPT-3.5-turbo
// promptTokens + maxTokens > modelContextLimit
```

### 4. Stop Sequence Edge Cases

```json
// ❌ Too many stop sequences
{"stop": ["END", "STOP", "FINISH", "DONE", "TERMINATE"]} // OpenAI max is 4

// ✅ Truncated to 4
{"stop": ["END", "STOP", "FINISH", "DONE"]}
```

### 5. Name Field Validation Edge Cases

```json
// ❌ Invalid name patterns
{"name": "user-123"}      // Hyphens not allowed
{"name": "very_long_name_that_exceeds_the_64_character_limit_for_names"} // Too long
{"name": "123user"}       // Can start with numbers

// ✅ Valid names
{"name": "user_123"}      // Underscores OK
{"name": "User123"}       // Mixed case OK
```

### 6. Streaming Edge Cases

**Mixed Streaming States**:
```json
// ❌ Cannot change streaming mid-conversation
// First request: {"stream": false}  
// Second request: {"stream": true}  // Different connection required
```

**Incomplete Streaming Data**:
```
data: {"choices": [{"delta": {"content": "Hel
// ❌ Incomplete JSON in stream chunk
```

**Stream Options Configuration**:
```json
{
  "stream": true,
  "stream_options": {
    "include_usage": true  // Include token usage in streaming response
  }
}
```

**Stream Options Edge Cases**:
```json
// ❌ stream_options without stream enabled
{
  "stream": false,
  "stream_options": {"include_usage": true}  // Ignored when stream=false
}

// ✅ Proper streaming with usage tracking
{
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

### 7. Reasoning Effort Edge Cases

**Reasoning Effort Compatibility**:
```json
// ✅ Valid for o1 models
{
  "model": "o1-preview",
  "reasoning_effort": "medium",
  "messages": [...]
}

// ❌ Invalid for non-o1 models  
{
  "model": "gpt-4o",
  "reasoning_effort": "high",  // Not supported on this model
  "messages": [...]
}
```

**Reasoning Effort Values**:
```json
// ✅ Valid reasoning_effort values
{
  "model": "o1-preview", 
  "reasoning_effort": "low",     // Faster, less thorough
  "messages": [...]
}

{
  "model": "gpt-5",
  "reasoning_effort": "medium",  // Balanced (default)
  "messages": [...]
}

{
  "model": "gpt-5", 
  "reasoning_effort": "high",    // Slower, more thorough
  "messages": [...]
}

// ❌ Invalid reasoning_effort value
{
  "model": "o1-preview",
  "reasoning_effort": "extreme", // Invalid value
  "messages": [...]
}
```

**Reasoning Effort Parameter Restrictions**:
```json
// ❌ Reasoning effort has parameter restrictions
{
  "model": "o1-preview", 
  "reasoning_effort": "high",
  "temperature": 0.8,  // ❌ Temperature not allowed with o1 models
}

// ✅ Proper reasoning effort usage
{
  "model": "o1-preview",
  "reasoning_effort": "medium",
  "max_completion_tokens": 1000,  // ✅ max_completion_tokens allowed
  "messages": [...]
}
```

### 8. JSON Mode Edge Cases

```json
// Request with JSON mode
{
  "response_format": {"type": "json_object"},
  "messages": [
    {"role": "user", "content": "Generate a number"}  // ❌ No JSON instruction
  ]
}
```

**Best Practice**: Always instruct model to output JSON when using `json_object` mode.

### 9. Function Schema Edge Cases

**Missing Required Properties**:
```json
{
  "parameters": {
    "type": "object",
    "properties": {"location": {"type": "string"}},
    // ❌ Missing "required" array - all properties become optional
  }
}
```

**Schema Validation Issues**:
```json
{
  "parameters": {
    "type": "object",
    "properties": {
      "date": {
        "type": "string",
        "format": "date"  // ❌ OpenAI doesn't support all JSON Schema formats
      }
    }
  }
}
```

## Validation Rules

### Field Validation Patterns

1. **Name Fields** (system.name, user.name, assistant.name):
   - Pattern: `^[a-zA-Z0-9_]{1,64}$`
   - Max length: 64 characters
   - Allowed: letters, numbers, underscores

2. **Function Names** (tools[].function.name):
   - Pattern: `^[a-zA-Z0-9_]{1,64}$`
   - Max length: 64 characters
   - Must be unique within tools array

3. **Tool Call IDs**:
   - Usually format: `call_<alphanumeric>`
   - Must be unique within single response
   - Referenced by tool messages

### Range Validations

```javascript
const validations = {
  temperature: { min: 0, max: 2, type: 'number' },
  top_p: { min: 0, max: 1, type: 'number' },
  frequency_penalty: { min: -2, max: 2, type: 'number' },
  presence_penalty: { min: -2, max: 2, type: 'number' },
  max_tokens: { min: 1, type: 'integer|null', deprecated: 'Use max_completion_tokens instead' },
  max_completion_tokens: { min: 1, type: 'integer|null' },
  n: { min: 1, max: 128, type: 'integer' },
  top_logprobs: { min: 0, max: 5, type: 'integer|null' },
  stop: { maxItems: 4, type: 'string|array' },
  reasoning_effort: { 
    type: 'string', 
    enum: ['low', 'medium', 'high'], 
    models: ['o1-preview', 'o1-mini'],
    default: 'medium'
  },
  stream_options: { 
    type: 'object|null',
    properties: {
      include_usage: { type: 'boolean' }
    },
    requiresStream: true
  }
};
```

### Schema Validation Implementation

```javascript
function validateOpenAIRequest(request) {
  const errors = [];
  
  // Required fields
  if (!request.model) errors.push('model is required');
  if (!request.messages || !Array.isArray(request.messages) || request.messages.length === 0) {
    errors.push('messages array is required and must not be empty');
  }
  
  // Message validation
  request.messages?.forEach((msg, i) => {
    if (!['system', 'user', 'assistant', 'tool'].includes(msg.role)) {
      errors.push(`messages[${i}].role must be one of: system, user, assistant, tool`);
    }
    
    if (msg.role !== 'assistant' && !msg.content) {
      errors.push(`messages[${i}].content is required for ${msg.role} messages`);
    }
    
    if (msg.role === 'tool' && !msg.tool_call_id) {
      errors.push(`messages[${i}].tool_call_id is required for tool messages`);
    }
  });
  
  // Reasoning effort validation
  if (request.reasoning_effort !== undefined) {
    const reasoningModels = ['o1-preview', 'o1-mini'];
    if (!reasoningModels.includes(request.model)) {
      errors.push(`reasoning_effort is only supported for models: ${reasoningModels.join(', ')}`);
    }
    
    const validEfforts = ['low', 'medium', 'high'];
    if (!validEfforts.includes(request.reasoning_effort)) {
      errors.push(`reasoning_effort must be one of: ${validEfforts.join(', ')}`);
    }
    
    // Check incompatible parameters for o1 models
    const incompatibleFields = ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'stream', 'tools', 'logprobs'];
    incompatibleFields.forEach(field => {
      if (request[field] !== undefined) {
        errors.push(`${field} cannot be used with o1 models`);
      }
    });
  }
  
  // Stream options validation
  if (request.stream_options && !request.stream) {
    errors.push('stream_options can only be used when stream=true');
  }
  
  return errors;
}
```

## Request Examples

### Basic Chat Request

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system", 
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "temperature": 0.7,
  "max_completion_tokens": 150
}
```

### Function Calling Request

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": "What's the weather like in Tokyo?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
              "description": "Temperature unit"
            }
          },
          "required": ["location"],
          "additionalProperties": false
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

### Streaming Request

```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "Write a short story"}
  ],
  "stream": true,
  "temperature": 0.8,
  "max_completion_tokens": 500
}
```

### JSON Mode Request

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant designed to output JSON."
    },
    {
      "role": "user", 
      "content": "Generate a JSON object with information about the planet Mars."
    }
  ],
  "response_format": {"type": "json_object"},
  "temperature": 0.3
}
```

### Complex Multi-turn with Tools

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant that can access weather data and perform calculations."
    },
    {
      "role": "user",
      "content": "What's the weather in Boston and New York, and what's the average temperature?"
    },
    {
      "role": "assistant",
      "content": "I'll get the weather for both cities and calculate the average temperature.",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"Boston, MA\"}"
          }
        },
        {
          "id": "call_2", 
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"New York, NY\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "Temperature: 68°F, Condition: Partly Cloudy",
      "tool_call_id": "call_1"
    },
    {
      "role": "tool",
      "content": "Temperature: 72°F, Condition: Sunny", 
      "tool_call_id": "call_2"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City and state"
            }
          },
          "required": ["location"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

### Streaming with Usage Tracking

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": "Write a detailed explanation of quantum computing."
    }
  ],
  "stream": true,
  "stream_options": {
    "include_usage": true
  },
  "temperature": 0.7,
  "max_completion_tokens": 1000
}
```

### Multimodal Request with Image

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What do you see in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
          }
        }
      ]
    }
  ],
  "max_completion_tokens": 300
}
```

### Reasoning Effort Request (o1 models)

```json
{
  "model": "o1-preview",
  "messages": [
    {
      "role": "user",
      "content": "Solve this complex math problem step by step: If a train travels 120 km in 1.5 hours, then speeds up by 20 km/h for the next 2 hours, what is the total distance traveled?"
    }
  ],
  "reasoning_effort": "high",
  "max_completion_tokens": 2000
}
```

**Note**: Reasoning effort is only available for o-series and gpt-5 models (o1-preview, o1-mini) and has restrictions:
- Values: `"low"`, `"medium"` (default), `"high"`
- Cannot use `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`
- Cannot use `logprobs`

## Best Practices

### 1. Schema Conversion Best Practices

**Message Handling**:
```javascript
// ✅ Good: Preserve message order and handle all content types
function convertMessages(claudeMessages, system) {
  const openaiMessages = [];
  
  // Add system message first if present
  if (system) {
    openaiMessages.push({
      role: 'system',
      content: extractTextFromBlocks(system)
    });
  }
  
  // Convert all messages, handling content blocks
  claudeMessages.forEach(msg => {
    const converted = {
      role: msg.role,
      content: extractTextFromBlocks(msg.content)
    };
    
    if (converted.content || msg.tool_calls) {
      openaiMessages.push(converted);
    }
  });
  
  return openaiMessages;
}
```

**Tool Conversion**:
```javascript
// ✅ Good: Preserve all tool metadata and validate schemas
function convertTools(claudeTools) {
  if (!claudeTools?.length) return null;
  
  return claudeTools.map(tool => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description || '',
      parameters: tool.input_schema || { type: 'object', properties: {} }
    }
  }));
}
```

### 2. Error Handling Best Practices

```javascript
// ✅ Good: Comprehensive error handling
async function callOpenAI(request) {
  try {
    // Validate request before sending
    const validationErrors = validateRequest(request);
    if (validationErrors.length > 0) {
      throw new Error(`Validation failed: ${validationErrors.join(', ')}`);
    }
    
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`OpenAI API error: ${errorData.error?.message || response.statusText}`);
    }
    
    return await response.json();
    
  } catch (error) {
    // Log error details for debugging
    console.error('OpenAI API call failed:', {
      error: error.message,
      request: JSON.stringify(request, null, 2)
    });
    throw error;
  }
}
```

### 3. Performance Best Practices

**Token Management**:
```javascript
// ✅ Good: Estimate and manage token usage
function estimateTokens(text) {
  // Rough estimation: 1 token ≈ 4 characters for English
  return Math.ceil(text.length / 4);
}

function validateTokenLimits(request, modelContextLimit = 8192) {
  const promptTokens = request.messages.reduce((total, msg) => {
    return total + estimateTokens(msg.content || '');
  }, 0);
  
  const maxTokens = request.max_tokens || 0;
  
  if (promptTokens + maxTokens > modelContextLimit) {
    throw new Error(`Token limit exceeded: ${promptTokens + maxTokens} > ${modelContextLimit}`);
  }
}
```

**Caching Strategy**:
```javascript
// ✅ Good: Cache responses when appropriate
const responseCache = new Map();

function getCacheKey(request) {
  // Only cache deterministic requests
  if (request.temperature === 0 && request.seed) {
    return JSON.stringify({
      model: request.model,
      messages: request.messages,
      seed: request.seed
    });
  }
  return null;
}
```

### 4. Security Best Practices

```javascript
// ✅ Good: Sanitize and validate all inputs
function sanitizeRequest(request) {
  return {
    ...request,
    // Remove any potentially sensitive fields
    user: request.user ? sanitizeUserId(request.user) : undefined,
    // Validate all string inputs
    messages: request.messages.map(msg => ({
      ...msg,
      content: sanitizeContent(msg.content),
      name: msg.name ? sanitizeName(msg.name) : undefined
    }))
  };
}

function sanitizeName(name) {
  // Ensure name matches allowed pattern
  return name.replace(/[^a-zA-Z0-9_]/g, '').slice(0, 64);
}
```

### 5. Monitoring and Observability

```javascript
// ✅ Good: Track key metrics and errors
function trackAPICall(request, response, duration) {
  const metrics = {
    model: request.model,
    promptTokens: response.usage?.prompt_tokens || 0,
    completionTokens: response.usage?.completion_tokens || 0,
    totalTokens: response.usage?.total_tokens || 0,
    duration: duration,
    hasTools: Boolean(request.tools?.length),
    toolCallsCount: response.choices?.[0]?.message?.tool_calls?.length || 0
  };
  
  // Send to monitoring system
  logMetrics('openai_api_call', metrics);
}
```

---

## Summary

This comprehensive analysis covers all aspects of the OpenAI Create Chat Completion API schema, with particular focus on:

- **Complete field coverage**: All 24+ request parameters documented
- **Schema conversion guidance**: Claude ↔ OpenAI mapping strategies  
- **Tool calling mastery**: Complete function calling implementation guide
- **Edge case awareness**: 15+ critical edge cases for robust handling
- **Production readiness**: Validation, error handling, and best practices

Use this reference for implementing robust, production-ready OpenAI API integrations with proper schema conversion, validation, and error handling.