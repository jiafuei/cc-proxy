# Architecture Overview

## Purpose
- `cc-proxy` turns Claude Code into a provider-agnostic client by speaking Anthropic’s API on the front end and translating requests to other providers behind the scenes.

## High-Level Layout
- The FastAPI application factory lives in `app/main.py`. It bootstraps configuration, structured logging, middlewares, and mounts the channel-specific routers:
  - `app/api/claude.py` exposes `/claude/v1/messages` and `/claude/v1/messages/count_tokens`.
  - `app/api/codex.py` scaffolds `/codex/v1/responses` for Codex/OpenAI Responses integration.
  - Legacy `/v1/messages` forwarding has been removed; clients must target the channel-prefixed routes.
- Runtime behavior is driven by two configuration layers:
  - Server defaults (`ConfigModel`) cover host/port, logging, dump settings in `app/config/models.py`.
  - User routing/provider config (`UserConfig`) is hot-reloadable and stored in `~/.cc-proxy/user.yaml` via models in `app/config/user_models.py`.
- Dependency injection keeps global state out. `app/dependencies/container.py` builds the `ServiceContainer`, wiring the transformer loader, provider manager, and router before stashing the instance on `app.state`.

## Request Lifecycle
1. **Ingress & Context**
   - `RequestContextMiddleware` (`app/middlewares/request_context.py`) creates a per-request correlation ID and stashes metadata on a `ContextVar` for logs/dumps.
   - Security and CORS headers are applied by `SecurityHeadersMiddleware` and FastAPI’s native middleware stack.
2. **Routing Decision**
   - Routers translate HTTP payloads into `ExchangeRequest` objects (`app/routing/exchange.py`) capturing the channel (`claude` or `codex`), original stream intent, and metadata.
   - `SimpleRouter` (`app/routing/router.py`) classifies Anthropic requests (built-in tools, plan mode, `/model` overrides, `!` suffix) and returns a `RoutingResult`. Non-Claude channels bypass heuristics and delegate directly by alias. A default Anthropic provider seeded from environment variables handles fallbacks.
3. **Provider Execution**
   - `ProviderManager` (`app/providers/provider.py`) maps model aliases to concrete `ProviderClient` instances. Provider behavior is described by `ProviderDescriptor`s in `app/providers/registry.py`, which define operations, URL suffixes, default transformers per channel, and capability flags.
   - `ProviderClient.execute()` orchestrates request transformers, issues the HTTP call via `httpx.AsyncClient`, runs response transformers, and emits an `ExchangeResponse` for downstream formatting. Channel-specific transformer pipelines are loaded via `app/transformers/loader.py`.
4. **Response Handling**
   - Non-streamed calls return JSON via `ORJSONResponse`; streamed calls are converted to Server-Sent Events using `app/api/sse.py` operating on `ExchangeResponse` objects. The dumper captures raw/normalized payloads for debugging.
5. **Observability**
   - `Dumper` (`app/observability/dumper.py`) optionally writes sanitized headers, transformed payloads, and SSE traces to disk based on config flags.
   - Structlog configuration (`app/config/log.py`) merges request context into console + rotating JSON logs.

## Configuration & Reloading
- `ConfigurationService` (`app/config/__init__.py`) loads server config and exposes `reload_config`.
- `SimpleUserConfigManager` (`app/config/user_manager.py`) owns user config I/O, validation, and hot reload callbacks. It watches for API-triggered reloads and notifies the service container, which rebuilds providers/router safely.
- API endpoints in `app/api/config.py` allow status queries, YAML validation, and manual reloads. Validation uses Pydantic plus custom reference checks to ensure models reference existing providers and routing aliases.

## Transformers & Extensibility
- `TransformerLoader` (`app/transformers/loader.py`) dynamically imports request/response/stream transformers, caches instances, and honours user-specified search paths.
- Built-in transformers now live under `app/transformers/providers/<channel>/` with shared utilities in `app/transformers/shared/` and common ABCs in `app/transformers/interfaces.py`. Claude channel classes are prefixed `Claude*`; Codex placeholders ship ready for future implementations.
- Users can define custom transformers in directories listed under `transformer_paths` in `user.yaml` and reference them with fully-qualified class paths.

## Exchange Layer & Channels
- `ExchangeRequest`, `ExchangeResponse`, and `ExchangeStreamChunk` (`app/routing/exchange.py`) provide provider-neutral contracts that keep routers, providers, and response formatters decoupled.
- The router notes the logical channel on every request, allowing multiple API families (Claude, Codex) to share a single service container while maintaining independent transformer pipelines.

## Testing & Samples
- Automated tests have been removed for this migration cycle; rely on manual smoke tests (cURL or Claude Code) plus the dumper outputs to validate behavior after configuration changes.
- `examples/` contains recorded Anthropic/OpenAI/Gemini JSON and SSE transcripts for ad-hoc verification.

## Key Runtime Assets
- `config.yaml` (repo root) – sample server config; copy to `~/.cc-proxy/config.yaml`.
- `user.example.yaml` – rich user config template covering providers, models, routing, and transformer overrides.
- `exports/` - design docs and system prompts captured during development.

## Operational Tips
- Start the proxy with `uv run fastapi dev` for autoreload while iterating on providers or routers.
- Inspect `/api/config/status` to confirm providers/models are loaded; call `/api/config/reload` after editing `user.yaml`.
- Tail logs in `~/.cc-proxy/logs/` and the dump directory to inspect routing information, headers, and transformed payloads per request.
