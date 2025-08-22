cc-proxy
---

Overview
- Proxy for Anthropic Messages API (v1/messages)
- On-the-fly transforms to provider chat formats (OpenAI, Gemini, or user-defined)
- Forwards to user-configured endpoints, then maps responses back to Anthropic format
- Supports streaming and non-streaming
- Composable, middleware-style pipelines with per-step configurability

Key features
- Pluggable transformation pipelines (request and response)
- Model routing for planning, thinking, background tasks, and more
- Custom providers: define base URLs and models
- Streaming passthrough with back-translation to Anthropic
- User-defined transforms at each stage

Configuration
- Custom providers: define URLs and models
- Model routing: map tasks to provider/model
- Request transformation: convert Anthropic requests to provider formats or custom shapes
- Response transformation: normalize provider responses to Anthropic v1/messages

Roadmap
- Phase 0: Local MVP
  - POST /v1/messages (stream/non-stream); built-in Anthropic↔OpenAI/Gemini transforms
  - Static YAML; minimal ModelRouter; direct HTTP client; basic timeouts; console logs
- Phase 1: Config foundations
  - Config loader/validator; Providers/Models schema (name, url, api_key, models[])
- Phase 2: Pipelines by configuration
  - Request/Response Pipeline engine; pipelines declared by name with ordered transformer refs; wire built-in transformers
- Phase 3: Routing stages
  - ModelRouter with stages: default, thinking, planning, background (background tasks); select provider+model per stage
- Phase 4: Extensibility (Plugin SDK, no sandbox yet)
  - User-defined transformers referenced by name/id; load by local disk path; fail-safe fallback
- Phase 5: Sandbox caps (moved out of Phase 4)
  - Run plugins in subprocess with CPU/mem/time caps; isolation and graceful fallback
- Phase 6: Reliability & streaming polish
  - Retries/backoff, circuit breaker, deadlines; backpressure/cancellation; schema validation hardening
- Phase 7: Tooling & docs
  - Config lint CLI, dry-run validate, examples; built-in transformer catalog (read-only)
- Phase 8: Observability
  - Structured logs, metrics, tracing
- Phase 9: Security & policy (later)
  - Secrets handling; provider URL allowlist/TLS hardening; quotas/rate limiting/audit (if multi-user later)

Architecture
- app/routers/messages.py
  - MessagesRouter: POST /v1/messages, stream/non-stream
- app/services/messages_service.py
  - MessageRouterService: orchestrates validate → route → pipelines → provider → pipelines → validate
- app/services/config/
  - ConfigLoader; ConfigValidator; Pydantic config models: ProviderConfig, PipelineConfig, TransformerRef, RoutingStagesConfig
- app/services/pipeline/
  - RequestPipeline; ResponsePipeline; interfaces: RequestTransformer, ResponseTransformer; built-ins: AnthropicToOpenAI, OpenAIToAnthropic, AnthropicToGemini, GeminiToAnthropic, ToolCallMappers
- app/services/routing/
  - ModelRouter; RouteDecision; RoutePolicy (stub)
- app/services/providers/
  - ProviderClient; ProviderRegistry; OpenAIClient; GeminiClient; CustomHTTPClient
- app/services/streaming/
  - StreamBridge
- app/services/validation/
  - RequestValidator; ResponseValidator
- app/common/
  - models.py; errors.py
- app/dependencies/
  - container.py

Config schema outline
- providers: [{name, url, api_key, models[], request_pipeline?, response_pipeline?}]
- model_routing:
  - stages:
    - default: {provider, model}
    - thinking: {provider, model}
    - planning: {provider, model}
    - background: {provider, model}
- pipelines: [{name, transformers: [string]}]  # ordered refs by id or name
- transformers:
  - builtins: referenced by well-known ids (not defined/edited in config)
  - user: [{name, path, kind: request|response, config?}]