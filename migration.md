# Migration Completion: Preparing for Codex Integration

## Goals (Achieved)
- Decouple Claude-specific logic from routing/provider/transformer layers to unlock new provider types.
- Support user-defined providers with explicit `base_url` and constrained `type` enums (anthropic, openai, openai-responses, gemini).
- Rename and regroup transformers into channel-aware modules while keeping the Transformer suffix.
- Provide channel-prefixed routers (/claude, /codex) backed by a shared service container and exchange layer.
- Remove the temporary `/v1/messages` compatibility shim now that prefixed endpoints are standard.

## Phase Summary

| Phase | Focus | Outcome |
| --- | --- | --- |
| 0 | Baseline assessment | Documented legacy behavior prior to the refactor. |
| 1 | Package layout reshuffle | Introduced app/api, app/routing, app/providers, app/transformers, and app/dependencies, migrating modules with shims for backward compatibility. |
| 2 | Service container & dependencies | Added `build_service_container` in app/dependencies/container.py and rewired FastAPI dependencies to build/refresh services through the container. |
| 3 | Exchange layer | Added ExchangeRequest, ExchangeResponse, and ExchangeStreamChunk to standardize provider interactions across channels. |
| 4 | Router prefixes | Implemented Claude and Codex routers plus a temporary /v1 shim (retired September 2025). |
| 5 | Provider registry | Replaced static specs with typed ProviderDescriptors, updated ProviderConfig to use `base_url`, and resolved operation URLs via descriptors. |
| 6 | Transformer reorganization | Consolidated interfaces in app/transformers/interfaces.py, renamed Claude transformers, introduced Codex placeholders, and moved shared utilities. |
| 7 | Provider execution flow | ProviderClient.execute() now drives channel-specific pipelines, preserves original stream intent, and returns ExchangeResponse objects. |
| 8 | API response handling | SSE converter consumes ExchangeResponse, Claude router streams or returns JSON based on the original request, Codex responds JSON-only for now. |
| 9 | Documentation & samples | Updated README, ARCHITECTURE, CLAUDE, config templates, and this migration log to reflect the new architecture and endpoint strategy. |
| 10 | Verification | Removed the obsolete pytest suite, performed manual smoke tests (curl, Claude Code), and confirmed structured logging/dumper outputs under the new flow. |
| 11 | Legacy cleanup | Removed the /v1 shim, compatibility exports, and legacy config helpers to finalize the migration. |

## Developer Notes
- **Service container** lives in app/dependencies/container.py and is the single source of truth for shared services.
- **Routers** are channelized: use /claude/v1/... for Anthropic-compatible flows and /codex/v1/... for future OpenAI Responses support. Legacy /v1/... endpoints are no longer available.
- **Provider registry** (app/providers/registry.py) is the extension point for adding new provider types or tweaking default transformer stacks.
- **Transformers** are organized by channel (app/transformers/providers/claude/*, app/transformers/providers/codex/*) with shared helpers in app/transformers/shared/.
- **Testing**: automated suites were intentionally removed for this migration. Rely on manual smoke tests plus dumper artifacts until fresh coverage is authored on top of the new architecture.

## Next Steps
- Flesh out Codex channel transformers once the upstream API stabilizes.
- Reintroduce automated coverage targeting the exchange layer and multi-channel routing once implementation is settled.
- Communicate the /v1 removal to integrators and ensure all docs reference the channel-prefixed endpoints.
