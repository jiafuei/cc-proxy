# cc-proxy 🚀

**A friendly proxy that makes Claude Code work with any AI provider**

Transform Claude Code into a universal AI client by routing requests to OpenAI, custom providers, or even multiple models simultaneously. Think of it as a smart translator that speaks every AI API dialect.

## What is cc-proxy?

cc-proxy sits between Claude Code and AI providers, translating requests and responses on-the-fly. Instead of being locked into one provider, you can:

- **Route to different models** for different tasks (fast models for quick work, powerful models for complex problems)
- **Use any provider** - OpenAI, custom endpoints, local models, or even multiple providers simultaneously  
- **Transform requests** - add authentication, modify prompts, or inject custom behavior
- **Stay in Claude Code** - no need to switch tools or learn new interfaces

Perfect for developers who want flexibility without sacrificing the Claude Code experience they love.

## Setup & Installation

### Prerequisites
- **Python 3.11+** (Python 3.12 recommended)
- **uv** for package management (or pip if you prefer)

### Quick Setup

1. **Clone and install**
   ```bash
   git clone https://github.com/your-username/cc-proxy.git
   cd cc-proxy
   uv sync
   ```

2. **Set up configuration directories**
   ```bash
   mkdir -p ~/.cc-proxy
   cp config.example.yaml ~/.cc-proxy/config.yaml
   cp user.example.yaml ~/.cc-proxy/user.yaml
   ```

3. **Configure your API keys**
   ```bash
   # Add to your shell profile (.bashrc, .zshrc, etc.)
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export OPENAI_API_KEY="your-openai-key"  # if using OpenAI
   ```

4. **Start the proxy**
   ```bash
   uv run python -m app.main
   ```

   You should see: `✅ Server running at http://127.0.0.1:8000`

5. **Test it works**
   ```bash
   curl -X POST "http://127.0.0.1:8000/v1/messages" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer dummy-key" \
     -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 100}'
   ```

## Key Features

### 🔄 **Universal Provider Support**
Connect to any AI provider with built-in transforms for OpenAI, Anthropic, and custom endpoints. No more vendor lock-in.

### 🧠 **Smart Model Routing** 
Automatically route different types of requests to different models:
- Planning tasks → Your most capable model
- Quick responses → Fast, cost-effective models  
- Background tasks → Specialized or cheaper models

### 🔧 **Flexible Transformations**
- **Request transforms**: Add authentication, modify prompts, inject context, merge tool calls
- **Response transforms**: Normalize outputs, add metadata, apply filters, clean system messages
- **Smart routing**: Subagent routing with intelligent message handling
- **Custom plugins**: Write your own transformers in Python
- **Software Engineering Mode**: Specialized system message for coding tasks

### 📦 **Drop-in Replacement**
Point Claude Code at `http://localhost:8000` and everything just works. No configuration changes needed in Claude Code itself.

### 🎯 **Composable Pipelines** 
Chain multiple transformers together for complex workflows. Each step is configurable and can be enabled/disabled per provider.

## Configuration

cc-proxy uses two configuration files:

### Server Configuration (`~/.cc-proxy/config.yaml`)
Basic server settings - host, port, logging, CORS, etc. Most users can use the defaults.

### User Configuration (`~/.cc-proxy/user.yaml`)
Your providers, models, and routing rules. This is where the magic happens:

```yaml
providers:
  - name: 'anthropic-provider'
    api_key: !env ANTHROPIC_API_KEY
    url: 'https://api.anthropic.com/v1/messages'
    # ... transformers configuration

models:
  - alias: 'sonnet'
    id: 'claude-3-5-sonnet-20241022'
    provider: 'anthropic-provider'
  - alias: 'gpt4'
    id: 'gpt-4o'
    provider: 'openai-provider'

routing:
  default: 'sonnet'        # Default model for most requests
  builtin_tools: 'sonnet'  # Built-in tools (WebSearch, WebFetch, etc.) - highest priority
  planning: 'gpt4'         # Use GPT-4 for planning tasks
  background: 'haiku'      # Use Haiku for quick tasks
```

See the example files for complete configuration options with detailed comments.

## Usage with Claude Code

1. **Start cc-proxy** (if not already running)
   ```bash
   uv run fastapi run
   ```

2. **Configure Claude Code** to use the proxy
   - Set API endpoint to: `http://localhost:8000`
   - Use any dummy API key (cc-proxy handles real authentication)

3. **Use Claude Code normally**
   - Your requests will be automatically routed based on your configuration
   - Different request types can go to different models
   - All responses come back in Claude format

That's it! Claude Code works exactly the same, but now with all the flexibility of cc-proxy.

## Advanced Configuration

### Custom Providers
```yaml
providers:
  - name: 'my-custom-api'
    api_key: !env CUSTOM_API_KEY
    url: 'https://my-api.example.com/v1/completions'
    transformers:
      request:
        - class: 'my_transformers.CustomAuthTransformer'
          params: {special_header: 'custom-value'}
```

### Multiple Model Routing
```yaml
routing:
  default: 'sonnet'           # Most requests
  builtin_tools: 'sonnet'     # Built-in tools (WebSearch, WebFetch, etc.) - highest priority
  thinking: 'o1'              # Complex reasoning
  planning: 'gpt4'            # Planning mode
  background: 'haiku'         # Quick tasks
  plan_and_think: 'sonnet'    # Planning + thinking
```

### Custom Transformers
Place Python files in directories listed in `transformer_paths` and reference them by `module.ClassName`.

#### Built-in Tools Support
CC-Proxy automatically converts Anthropic's built-in tools (WebSearch, WebFetch) to provider-specific formats:

```yaml
# OpenAI Provider with built-in tools support
- name: 'openai-provider'
  transformers:
    request:
      - class: 'app.services.transformers.openai.OpenAIRequestTransformer'
    response:
      - class: 'app.services.transformers.openai.OpenAIResponseTransformer'
```

**WebSearch Conversion**:
- Anthropic `web_search` tool → OpenAI `web_search_options` parameter
- Domain filters (`allowed_domains`, `blocked_domains`) mapped correctly
- User location parameters converted to OpenAI format
- Model automatically upgraded to search-preview variants
- Response annotations converted back to Anthropic format

## Technical Details

### Architecture
- **FastAPI server** with async request handling
- **Pluggable transformer system** for request/response modification
- **Provider abstraction** supporting HTTP and streaming protocols
- **Configuration-driven routing** with environment variable support
- **Comprehensive logging** and debugging tools

### API Compatibility
- **OpenAI Chat Completions** - Automatic format conversions
- **Streaming support** - Real-time response streaming
- **Subagent routing** - Advanced routing capabilities for complex workflows

### Development Commands
```bash
# Run tests
python -m pytest app/ -v

# Lint and format
uvx ruff check --fix && uvx ruff format .

# Start development server
uv run fastapi dev
```

## Contributing & Support

### Getting Help
- **Issues**: Report bugs or request features on GitHub
- **Discussions**: Ask questions or share configurations
- **Documentation**: Check example configs and inline comments

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Run linting: `uvx ruff check --fix && uvx ruff format .`
5. Submit a pull request

### Acknowledgments
Inspired by [ccflare](https://github.com/snipeship/ccflare) and [claude-code-router](https://github.com/musistudio/claude-code-router). Built with FastAPI, httpx, and lots of ☕.

---

**Ready to supercharge your Claude Code experience?** Start with the Quick Setup above and explore the possibilities! 🚀