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
- [x] Wire built-in transformers (Anthropic, OpenAI, Gemini)
- [ ] Pipeline validation and error recovery
- [ ] Pipeline debugging and introspection tools

## Phase 3: Routing Stages 📋
**Status: Planned**

Advanced routing based on request context and content.

- [ ] ModelRouter with stages: default, thinking, planning, background
- [ ] Context-aware routing (detect request type from content)
- [ ] Per-stage provider+model selection
- [ ] Routing policy configuration
- [ ] Request classification system

## Phase 4: Extensibility (Plugin SDK) 📋
**Status: Planned**

User-defined transformers and plugin system (without sandboxing).

- [ ] User-defined transformers referenced by name/id
- [ ] Load transformers from local disk paths
- [ ] Plugin discovery and registration
- [ ] Fail-safe fallback mechanisms
- [ ] Plugin API documentation
- [ ] Example plugins and templates

## Phase 5: Sandbox Capabilities 📋
**Status: Future**

Secure plugin execution environment.

- [ ] Run plugins in subprocess with CPU/mem/time caps
- [ ] Process isolation and security boundaries
- [ ] Graceful fallback when plugins fail
- [ ] Resource monitoring and limits
- [ ] Plugin health checks

## Phase 6: Reliability & Streaming Polish 📋
**Status: Planned**

Production-grade reliability features.

- [ ] Retries with exponential backoff
- [ ] Circuit breaker pattern
- [ ] Request deadlines and timeouts
- [ ] Backpressure handling
- [ ] Request/response cancellation
- [ ] Schema validation hardening
- [ ] Connection pooling optimization

## Phase 7: Tooling & Documentation 📋
**Status: Planned**

Developer experience improvements.

- [ ] Config lint CLI tool
- [ ] Dry-run validation mode
- [ ] Configuration examples and templates
- [ ] Built-in transformer catalog (read-only)
- [ ] Interactive configuration generator
- [ ] Comprehensive API documentation

## Phase 8: Observability 📋
**Status: Planned**

Monitoring, metrics, and debugging capabilities.

- [ ] Structured logging with correlation IDs
- [ ] Metrics collection (requests, latency, errors)
- [ ] Distributed tracing support
- [ ] Health check endpoints
- [ ] Performance monitoring
- [ ] Request/response debugging tools

## Phase 9: Security & Policy 📋
**Status: Future**

Advanced security and multi-user support.

- [ ] Secure secrets handling
- [ ] Provider URL allowlist
- [ ] TLS certificate hardening
- [ ] Rate limiting per user/key
- [ ] Audit logging
- [ ] Multi-tenant support (if needed)

## Implementation Notes

### Phase Priorities
- **Phase 0-2**: Core functionality for single-user local development
- **Phase 3-4**: Advanced features for power users
- **Phase 5-6**: Production readiness and reliability  
- **Phase 7-8**: Developer experience and operations
- **Phase 9**: Enterprise and multi-user scenarios

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