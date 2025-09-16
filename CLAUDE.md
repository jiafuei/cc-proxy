# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code organization
- `app/main.py` (FastAPI factory & router mounting)
- `app/api/` (channel-specific routers: `legacy`, `claude`, `codex`)
- `app/routing/` (request inspector, router, exchange dataclasses)
- `app/providers/` (provider descriptors, registry, manager, HTTP clients)
- `app/transformers/` (interfaces, shared utilities, channelled transformers)
- `app/di/` (service container wiring)
- `app/common/` (shared utilities: dumper, SSE converter, models, middleware helpers)
- Tests have been removed for this migration cycle; rely on manual smoke checks.

## Commands
- `uvx ruff check --fix && uvx ruff format .` — lint and format
- `uv run fastapi dev` — launch the API with autoreload for local manual testing
- `curl -X POST http://127.0.0.1:8000/claude/v1/messages ...` — verify Claude channel manually

## Configuration Files
- `config.example.yaml`: The example static server config
- `user.example.yaml` The example dynamic user config


- Embrace dependency injection (`app/di/container.py`)—never instantiate providers/routers ad hoc.
- Transformers live under `app/transformers/providers/<channel>/`; prefer channel-specific naming.
- Manual verification matters: use curl/Claude Code plus the dumper outputs until automated coverage returns.
- Keep responses concise; avoid over-engineering core routing/exchange logic.
- Combine shell commands with `&&` to conserve tool budget.
