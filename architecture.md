# Architecture Overview

## Purpose
- `cc-proxy` turns Claude Code into a provider-agnostic client by speaking Anthropic’s API on the front end and translating requests to other providers behind the scenes.

## High-Level Layout
- FastAPI application factory lives at `app/main.py`, wiring configuration, structured logging, middlewares, and routers before exporting `app`.
- Runtime behavior is driven by two config layers:
  - Server defaults (`ConfigModel`) cover host/port, logging, dump settings in `app/config/models.py`.
  - User routing/provider config (`UserConfig`) is hot-reloadable and stored in `~/.cc-proxy/user.yaml` via models in `app/config/user_models.py`.
- Dependency injection keeps global state out. `ServiceContainer` in `app/dependencies/service_container.py` assembles core services and rides in FastAPI `app.state`.

## Request Lifecycle
1. **Ingress & Context**
   - `RequestContextMiddleware` (`app/middlewares/request_context.py`) creates a per-request correlation ID and stashes metadata on a `ContextVar` for logs/dumps.
   - Security and CORS headers are applied by `SecurityHeadersMiddleware` and FastAPI’s native middleware stack.
2. **Routing Decision**
   - `/v1/messages` endpoint (`app/routers/messages.py`) validates the Anthropic-style payload, normalizes thinking budgets, and asks the router which provider/model to use.
   - `SimpleRouter` (`app/services/router.py`) inspects the request (built-in tools, plan mode, `/model` directives, `!` suffix) and returns a `RoutingResult` describing provider, resolved model id, and routing flags. Fallback: default Anthropic provider seeded from environment.
3. **Provider Execution**
   - `ProviderManager` (`app/services/provider.py`) maps model aliases to provider instances. Each `Provider` uses specs from `app/services/providers/specs.py` to know operations, URL suffixes, and default transformer stacks.
   - Outbound request flows through request transformers, hits the target HTTP endpoint via `httpx.AsyncClient`, and response transformers normalize the payload back to Anthropic format.
4. **Response Handling**
   - Non-streamed calls return JSON via `ORJSONResponse`; streamed calls are converted to Server-Sent Events with `app/common/sse_converter.py`, which emits `message_start`, `content_block_*`, `message_delta`, and `message_stop` frames.
5. **Observability**
   - `Dumper` (`app/common/dumper.py`) optionally writes sanitized headers, transformed payloads, and SSE traces to disk based on config flags.
   - Structlog configuration (`app/config/log.py`) merges request context into console + rotating JSON logs.

## Configuration & Reloading
- `ConfigurationService` (`app/config/__init__.py`) loads server config and exposes `reload_config`.
- `SimpleUserConfigManager` (`app/services/config/simple_user_config_manager.py`) owns user config I/O, validation, and hot reload callbacks. It watches for API-triggered reloads and notifies the service container, which rebuilds providers/router safely.
- API endpoints in `app/routers/config.py` allow status queries, YAML validation, and manual reloads. Validation uses Pydantic plus custom reference checks to ensure models reference existing providers and routing aliases.

## Transformers & Extensibility
- `TransformerLoader` (`app/services/transformer_loader.py`) dynamically imports request/response transformers, caches instances, and honours user-specified search paths.
- Built-in transformers live under `app/services/transformers/`, handling Anthropic cleanup, OpenAI/Gemini shape conversions, tool translation, authentication headers, and specialized Claude Code prompts.
- Users can define custom transformers in directories listed under `transformer_paths` in `user.yaml` and reference them with fully-qualified class paths.

## Testing & Samples
- `app/tests/test_integration.py` exercises end-to-end message handling, routing decisions, configuration validation, and reload flows using mocked providers/service containers supplied by helpers in `app/tests/utils.py`.
- `examples/` contains recorded Anthropic/OpenAI/Gemini JSON and SSE transcripts for manual testing or regression comparison.

## Key Runtime Assets
- `config.yaml` (repo root) – sample server config; copy to `~/.cc-proxy/config.yaml`.
- `user.example.yaml` – rich user config template covering providers, models, routing, and transformer overrides.
- `exports/` – design docs and system prompts captured during development.

## Operational Tips
- Start the proxy with `uv run python -m app.main` (or `uv run fastapi dev` for autoreload).
- Inspect `/api/config/status` to confirm providers/models are loaded; call `/api/config/reload` after editing `user.yaml`.
- Tail logs in `~/.cc-proxy/logs/` and the dump directory to diagnose routing, headers, and transformed payloads.
