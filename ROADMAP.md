# cc-proxy Development Roadmap

This document outlines the planned development phases for cc-proxy. Each phase builds upon the previous ones to create a comprehensive, production-ready proxy system.

## Phase 0: Local MVP ✅
**Status: Complete**

Basic functionality for local development and testing.

- [x] POST /v1/messages (stream/non-stream)
- [x] Built-in Anthropic ↔ OpenAI/Gemini transforms
- [x] Static YAML configuration
- [x] Minimal ModelRouter
- [x] Direct HTTP client
- [x] Basic timeouts
- [x] Console logging

## Phase 1: Config Foundations ✅
**Status: Complete**

Robust configuration system with validation.

- [x] Config loader/validator with Pydantic
- [x] Providers/Models schema (name, url, api_key, models[])
- [x] Environment variable support
- [x] Configuration file validation
- [x] Error handling and user-friendly messages

## Phase 2: Pipelines by Configuration 🚧
**Status: In Progress**

Dynamic pipeline configuration and built-in transformers.

- [x] Request/Response Pipeline engine
- [x] Pipelines declared by name with ordered transformer refs
- [x] Wire built-in transformers (Anthropic, OpenAI)
- [x] Pipeline validation and error recovery
- [ ] Gemini transformers 
- [ ] Pipeline debugging and introspection tools

## Phase 3: Routing Stages ✅
**Status: Complete**

Advanced routing based on request context and content.

- [x] ModelRouter with stages: default, thinking, planning, background, plan_and_think
- [x] Context-aware routing (detect request type from content)
- [x] Per-stage provider+model selection
- [x] Routing policy configuration
- [x] Request classification system
- [x] Direct routing with '!' suffix

## Phase 4: Extensibility (Plugin SDK) 🚧
**Status: In Progress**

User-defined transformers and plugin system (without sandboxing).

- [x] User-defined transformers referenced by name/id
- [x] Load transformers from local disk paths
- [x] Plugin discovery and registration
- [x] Fail-safe fallback mechanisms
- [ ] Plugin API documentation
- [ ] Example plugins and templates

## Phase 5: Reliability & Streaming Polish 📋
**Status: Planned**

Production-grade reliability features.

- [ ] Retries with exponential backoff
- [ ] Circuit breaker pattern
- [ ] Request deadlines and timeouts
- [ ] Backpressure handling
- [ ] Request/response cancellation
- [ ] Schema validation hardening
- [ ] Connection pooling optimization

## Phase 6: Observability 🚧
**Status: In Progress**

Monitoring, metrics, and debugging capabilities.

- [x] Structured logging with correlation IDs
- [x] Health check endpoints (/health)
- [x] Request/response debugging tools (dumping system)
- [x] Request/response correlation tracking
- [ ] Metrics collection (requests, latency, errors)
- [ ] Distributed tracing support
- [ ] Performance monitoring

## Phase 7: Tooling & Documentation 🚧
**Status: In Progress**

Developer experience improvements.

- [x] Configuration validation API (/api/config/validate)
- [x] YAML validation endpoint (/api/config/validate-yaml)
- [x] Configuration reload API (/api/reload)
- [x] Configuration status endpoint (/api/config/status)
- [ ] Config lint CLI tool
- [ ] Dry-run validation mode
- [ ] Configuration examples and templates
- [ ] Built-in transformer catalog (read-only)
- [ ] Interactive configuration generator
- [ ] Comprehensive API documentation

## Phase 8: Security & Policy 📋
**Status: Future**

Advanced security and multi-user support.

- [ ] Secure secrets handling
- [ ] Provider URL allowlist
- [ ] TLS certificate hardening
- [ ] Rate limiting per user/key
- [ ] Audit logging
- [ ] Multi-tenant support (if needed)

## Phase 9: Sandbox Capabilities 📋
**Status: Future**

Secure plugin execution environment.

- [ ] Run plugins in subprocess with CPU/mem/time caps
- [ ] Process isolation and security boundaries
- [ ] Graceful fallback when plugins fail
- [ ] Resource monitoring and limits
- [ ] Plugin health checks

## Additional Implemented Features

**Features that exist but were not in the original roadmap:**

### Message Processing Extensions
- **Token Counting Endpoint**: `/v1/messages/count_tokens` for request token analysis
- **Streaming Response Conversion**: Non-streaming responses converted to SSE format for consistency
- **Subagent Routing**: Advanced routing capabilities for complex AI workflows

### Generic Transformers (Phase 4 extras)
- **UrlPathTransformer**: Modify provider URLs dynamically
- **AddHeaderTransformer**: Generic header injection
- **AnthropicHeadersTransformer**: Configurable auth header support
- **ClaudeSystemMessageCleanerTransformer**: System message processing
- **RequestBodyTransformer**: JSONPath-based request modification

### Advanced Caching (Phase 2 extras)
- **AnthropicCacheTransformer**: Intelligent cache breakpoint optimization
- **Tool reordering**: Default tools first, MCP tools second
- **Cache breakpoint management**: Up to 4 breakpoints strategically placed

### Configuration Management APIs (Phase 7 extras)
- **Real-time config reload**: `/api/reload` without service restart
- **Config validation**: `/api/config/validate` and `/api/config/validate-yaml`
- **System status**: `/api/config/status` with provider/model counts

## Implementation Notes

### Phase Priorities (Updated)
- **Production Readiness** (Phases 4-6): Extensibility → Reliability → Observability
  - **Phase 4**: 🚧 **IN PROGRESS** - Plugin SDK and extensibility foundation
  - **Phase 5**: 📋 **PLANNED** - Production reliability and streaming polish
  - **Phase 6**: 🚧 **IN PROGRESS** - Monitoring and debugging capabilities
- **Developer Experience** (Phase 7): 🚧 **IN PROGRESS** - Tooling and documentation
- **Security & Enterprise** (Phases 8-9): Advanced security and isolation
  - **Phase 8**: 📋 **FUTURE** - Security policies and multi-user support
  - **Phase 9**: 📋 **FUTURE** - Secure plugin sandbox environment

### Breaking Changes
- Configuration format changes will be avoided where possible
- When necessary, migration tools and documentation will be provided
- Major version bumps will indicate breaking changes

### Community Input
- Feature requests and feedback will influence phase priorities
- Community contributions are welcome for any phase
- Plugin API design will involve community review

---

**Want to contribute to a specific phase?** Check our [Contributing Guidelines](README.md#contributing--support) and open an issue to discuss your ideas!