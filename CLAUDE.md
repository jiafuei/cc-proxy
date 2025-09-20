# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code organization
- `app/main.py` (FastAPI factory & router mounting)
- `app/api/` (channel-specific routers: `claude`, `codex`; legacy `/v1` shim has been removed)
- `app/routing/` (request inspector, router, exchange dataclasses)
- `app/providers/` (provider descriptors, registry, manager, HTTP clients)
- `app/transformers/` (interfaces, shared utilities, channelled transformers)
- `app/dependencies/` (service container wiring and FastAPI dependencies)
- `app/context.py` (request-scoped context helpers for logging/dumping)
- `app/observability/` (dumper implementation and related helpers)
- `app/models/` (Pydantic request/response models shared across routers)
- Tests have been removed for this migration cycle; rely on manual smoke checks.

## Commands
- `uvx ruff check --fix && uvx ruff format .` — lint and format
- `uv run fastapi dev` — launch the API with autoreload for local manual testing
- `curl -X POST http://127.0.0.1:8000/claude/v1/messages ...` — verify Claude channel manually

## Configuration Files
- `config.example.yaml`: The example static server config
- `user.example.yaml` The example dynamic user config

### Transformer Configuration
Providers support flexible transformer configuration:
- **Full override**: Specify `request`, `response`, or `stream` lists to completely replace provider defaults
- **Additive configuration**: Use `pre_request`, `post_request`, `pre_response`, `post_response`, `pre_stream`, `post_stream` to add transformers before/after any stage configuration
- **Merging behavior**: Pre-transformers + (full override OR provider defaults) + post-transformers
- **Always applied**: Pre/post transformers are always applied when specified, regardless of whether full stage overrides exist
- **Backward compatibility**: Existing configurations using full stage lists continue to work unchanged


- Embrace dependency injection (`app/dependencies/container.py`)—never instantiate providers/routers ad hoc.
- Transformers live under `app/transformers/providers/<channel>/`; prefer channel-specific naming.
- Manual verification matters: use curl/Claude Code plus the dumper outputs until automated coverage returns.
- Keep responses concise; avoid over-engineering core routing/exchange logic.
- Combine shell commands with `&&` to conserve tool budget.
