"""Hardcoded provider specifications for simplified architecture.

This module contains all provider specifications as Python constants, eliminating
the need for external configuration files and complex class hierarchies.
"""

# Provider specifications with all configuration data
PROVIDER_SPECS = {
    'anthropic': {
        'supported_operations': ['messages', 'count_tokens'],
        'url_suffixes': {'messages': '/v1/messages', 'count_tokens': '/v1/messages/count_tokens'},
        'default_transformers': {
            'request': [
                {'class': 'app.services.transformers.anthropic.AnthropicHeadersTransformer'},
                {'class': 'app.services.transformers.anthropic.AnthropicCacheTransformer'},
            ],
            'response': [{'class': 'app.services.transformers.anthropic.AnthropicResponseTransformer'}],
        },
    },
    'openai': {
        'supported_operations': ['messages'],
        'url_suffixes': {'messages': '/v1/chat/completions'},
        'default_transformers': {
            'request': [
                {'class': 'app.services.transformers.utils.HeaderTransformer', 'params': {'operations': [{'key': 'authorization', 'prefix': 'Bearer ', 'value': ''}]}},
                {'class': 'app.services.transformers.openai.OpenAIRequestTransformer'},
            ],
            'response': [{'class': 'app.services.transformers.openai.OpenAIResponseTransformer'}],
        },
    },
    'gemini': {
        'supported_operations': ['messages', 'count_tokens'],
        'url_suffixes': {'messages': '/v1beta/models/{model}:generateContent', 'count_tokens': '/v1beta/models/{model}:countTokens'},
        'default_transformers': {
            'request': [
                {'class': 'app.services.transformers.gemini.GeminiApiKeyTransformer'},
                {'class': 'app.services.transformers.gemini.GeminiRequestTransformer'},
            ],
            'response': [{'class': 'app.services.transformers.gemini.GeminiResponseTransformer'}],
        },
    },
}
