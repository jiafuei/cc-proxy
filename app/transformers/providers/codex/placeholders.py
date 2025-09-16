"""Placeholder transformers for the Codex channel integration."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from app.transformers.interfaces import ProviderRequestTransformer, ProviderResponseTransformer


class CodexAnthropicBridgeRequestTransformer(ProviderRequestTransformer):
    """Temporary passthrough transformer until Codex Anthropic bridge is implemented."""

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        return params['request'], params['headers']


class CodexAnthropicBridgeResponseTransformer(ProviderResponseTransformer):
    """Temporary passthrough for Anthropic bridge responses."""

    async def transform_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params['response']


class CodexOpenAIRequestTransformer(ProviderRequestTransformer):
    """Placeholder transformer for Codex OpenAI requests."""

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        return params['request'], params['headers']


class CodexOpenAIResponseTransformer(ProviderResponseTransformer):
    """Placeholder transformer for Codex OpenAI responses."""

    async def transform_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params['response']


class CodexOpenAIResponsesRequestTransformer(ProviderRequestTransformer):
    """Passthrough request transformer for OpenAI Responses API."""

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        return params['request'], params['headers']


class CodexOpenAIResponsesResponseTransformer(ProviderResponseTransformer):
    """Passthrough response transformer for OpenAI Responses API."""

    async def transform_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params['response']


class CodexGeminiBridgeRequestTransformer(ProviderRequestTransformer):
    """Placeholder transformer for Codex Gemini bridge requests."""

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        return params['request'], params['headers']


class CodexGeminiBridgeResponseTransformer(ProviderResponseTransformer):
    """Placeholder transformer for Codex Gemini bridge responses."""

    async def transform_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params['response']

