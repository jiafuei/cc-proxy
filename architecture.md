# Architecture Overview

## Purpose
`cc-proxy` is a protocol adapter that makes Claude Code (or any Anthropic-compatible client) work with multiple LLM providers by accepting Anthropic API requests and transparently translating them to/from provider-specific formats (OpenAI, Gemini, etc.).

## Core Architecture

**Exchange Layer** (`app/routing/exchange.py`): Provider-neutral contracts (`ExchangeRequest`, `ExchangeResponse`) decouple routers from providers. Transformers bridge specific API dialects.

**Provider Descriptors** (`app/providers/descriptors.py`): Templates for backends (Anthropic, OpenAI, Gemini) defining operations, URLs, and default transformers. Extensible: add descriptor + transformers for new providers.

**Transformer Pipelines** (`app/transformers/`): Channel/stage-specific logic for API translation. Merging: `pre_*` + (override OR defaults) + `post_*`. Interfaces in `interfaces.py`; implementations under `providers/<channel>/`.

**Service Container** (`app/dependencies/container.py`): DI for providers, router, loader. Hot-reload via rebuild on config change.

**Context & Observability**:
- `RequestContext` (`app/context.py`): Thread-safe per-request state (ContextVar).
- Two-layer config: server (`config.yaml`) + user (`user.yaml`).
- Dumper traces payloads/headers to disk; structlog with context injection.

## Request Flow

1. **Ingress**: Middleware creates correlation ID/ContextVar; CORS/security headers.

2. **Routing** (`app/api/*`, `app/routing/router.py`):
   - Validate payload → `ExchangeRequest`.
   - Claude: Heuristics (tools, plan mode, `!` suffix) → routing key → model alias.
   - Others: Direct alias lookup.
   - `ProviderManager` resolves alias → provider/model ID; fallback to env-based Anthropic.

3. **Execution** (`app/providers/provider.py`):
   - `ProviderClient`: Run request transformers → HTTP via httpx (stream=false) → response transformers → `ExchangeResponse`.

4. **Response**: JSON (non-stream) or SSE conversion; dumper logs all stages.

## Configuration & Reloading

- **Server**: Host/port/logging/dumps (`config.yaml`).
- **User**: Providers (type/url/key/overrides), models (alias→provider/ID), routing (Claude keys), transformer paths (`user.yaml`).
- Reload: `/api/config/reload` rebuilds container (closes old clients, no downtime).

## Operational Tips
- Dev: `uv run fastapi dev`.
- Status: `/api/config/status`.
- Reload: `/api/config/reload`.
- Debug: Logs (`~/.cc-proxy/logs/`), dumps (`~/.cc-proxy/dumps/`).
- Lint: `uvx ruff check --fix && uvx ruff format .`.
