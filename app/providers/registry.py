"""Static registry of built-in provider descriptors."""

from __future__ import annotations

from typing import Dict

from app.providers.descriptors import ProviderDescriptor
from app.providers.types import ProviderType

PROVIDER_REGISTRY: Dict[ProviderType, ProviderDescriptor] = {
    ProviderType.ANTHROPIC: ProviderDescriptor(
        type=ProviderType.ANTHROPIC,
        base_url_suffixes={
            'messages': '/v1/messages',
            'count_tokens': '/v1/messages/count_tokens',
        },
        default_transformers={
            'claude': {
                'request': [],
                'response': [],
                'stream': [],
            },
            'codex': {
                'request': [
                    {'class': 'app.transformers.providers.codex.placeholders.CodexAnthropicBridgeRequestTransformer', 'params': {}},
                ],
                'response': [
                    {'class': 'app.transformers.providers.codex.placeholders.CodexAnthropicBridgeResponseTransformer', 'params': {}},
                ],
                'stream': [],
            },
        },
        supports_streaming=True,
        supports_count_tokens=True,
        supports_responses=False,
    ),
    ProviderType.OPENAI: ProviderDescriptor(
        type=ProviderType.OPENAI,
        base_url_suffixes={'messages': '/v1/chat/completions'},
        default_transformers={
            'claude': {
                'request': [
                    {'class': 'app.transformers.providers.claude.openai.ClaudeOpenAIRequestTransformer', 'params': {}},
                ],
                'response': [
                    {'class': 'app.transformers.providers.claude.openai.ClaudeOpenAIResponseTransformer', 'params': {}},
                ],
                'stream': [],
            },
            'codex': {
                'request': [],
                'response': [],
                'stream': [],
            },
        },
        supports_streaming=True,
        supports_count_tokens=False,
        supports_responses=False,
    ),
    ProviderType.OPENAI_RESPONSES: ProviderDescriptor(
        type=ProviderType.OPENAI_RESPONSES,
        base_url_suffixes={'responses': '/v1/responses'},
        default_transformers={
            'claude': {
                'request': [
                    {'class': 'app.transformers.providers.claude.openai.ClaudeOpenAIRequestTransformer', 'params': {}},
                ],
                'response': [
                    {'class': 'app.transformers.providers.claude.openai.ClaudeOpenAIResponseTransformer', 'params': {}},
                ],
                'stream': [],
            },
            'codex': {
                'request': [],
                'response': [],
                'stream': [],
            },
        },
        supports_streaming=True,
        supports_count_tokens=False,
        supports_responses=True,
    ),
    ProviderType.GEMINI: ProviderDescriptor(
        type=ProviderType.GEMINI,
        base_url_suffixes={
            'messages': '/v1beta/models/{model}:generateContent',
            'count_tokens': '/v1beta/models/{model}:countTokens',
        },
        default_transformers={
            'claude': {
                'request': [
                    {'class': 'app.transformers.shared.utils.GeminiApiKeyTransformer', 'params': {}},
                    {'class': 'app.transformers.providers.claude.gemini.ClaudeGeminiRequestTransformer', 'params': {}},
                ],
                'response': [
                    {'class': 'app.transformers.providers.claude.gemini.ClaudeGeminiResponseTransformer', 'params': {}},
                ],
                'stream': [],
            },
            'codex': {
                'request': [
                    {'class': 'app.transformers.providers.codex.placeholders.CodexGeminiBridgeRequestTransformer', 'params': {}},
                ],
                'response': [
                    {'class': 'app.transformers.providers.codex.placeholders.CodexGeminiBridgeResponseTransformer', 'params': {}},
                ],
                'stream': [],
            },
        },
        supports_streaming=True,
        supports_count_tokens=True,
        supports_responses=False,
    ),
}


def get_descriptor(provider_type: ProviderType) -> ProviderDescriptor:
    """Retrieve the descriptor for a provider type."""

    return PROVIDER_REGISTRY[provider_type]
