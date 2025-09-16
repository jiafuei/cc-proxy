"""Routing heuristics mapping exchange requests to providers and models."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from app.config.log import get_logger
from app.config.user_models import ProviderConfig, RoutingConfig
from app.context import get_request_context
from app.models import AnthropicRequest
from app.providers.provider import ProviderClient, ProviderManager
from app.providers.types import ProviderType
from app.routing.exchange import ExchangeRequest
from app.transformers.loader import TransformerLoader

logger = get_logger(__name__)


@dataclass
class RoutingResult:
    """Complete routing information for a request."""

    provider: ProviderClient
    routing_key: str
    model_alias: str
    resolved_model_id: str
    channel: str
    is_direct_routing: bool = False
    is_agent_routing: bool = False
    used_fallback: bool = False


def _normalize_base_url(raw_url: str) -> str:
    suffixes = ['/v1/messages', '/v1/messages/']
    for suffix in suffixes:
        if raw_url.endswith(suffix):
            return raw_url[: -len(suffix)]
    return raw_url


def _create_default_anthropic_config() -> ProviderConfig:
    """Create a default Anthropic provider configuration from environment variables."""

    base_url = _normalize_base_url(os.getenv('CCPROXY_FALLBACK_URL', 'https://api.anthropic.com'))
    api_key = os.getenv('CCPROXY_FALLBACK_API_KEY', '')

    if not api_key:
        logger.warning('CCPROXY_FALLBACK_API_KEY not set - default provider will not work without authentication')

    return ProviderConfig(
        name='default-anthropic (fallback)',
        base_url=base_url,
        api_key=api_key,
        type=ProviderType.ANTHROPIC,
        capabilities=['messages', 'count_tokens'],
    )


class RequestInspector:
    """Analyzes Anthropic requests to determine routing key."""

    AGENT_PATTERN = re.compile(r'^/model\s+([^\s]+)$')

    def determine_routing_key(self, request: AnthropicRequest) -> str:
        if self._has_builtin_tools(request):
            return 'builtin_tools'

        if request.max_tokens and request.max_tokens < 768:
            return 'background'

        has_plan_mode = self._has_plan_mode_activation(request)
        has_thinking = self._has_thinking_config(request)

        if has_plan_mode and has_thinking:
            return 'plan_and_think'
        if has_thinking:
            return 'thinking'
        if has_plan_mode:
            return 'planning'
        return 'default'

    def _has_thinking_config(self, request: AnthropicRequest) -> bool:
        return request.thinking is not None and request.thinking.budget_tokens > 0

    def _has_plan_mode_activation(self, request: AnthropicRequest) -> bool:
        plan_mode_text = '<system-reminder>\nPlan mode is active.'
        last_user_message = next((msg for msg in reversed(request.messages) if msg.role == 'user'), None)
        if not last_user_message:
            return False

        content = last_user_message.content
        if isinstance(content, str):
            return plan_mode_text in content
        if isinstance(content, list):
            for block in content:
                if hasattr(block, 'text') and block.text and plan_mode_text in block.text:
                    return True
                if getattr(block, 'type', None) == 'tool_result' and plan_mode_text in getattr(block, 'content', ''):
                    return True
        return False

    def _has_builtin_tools(self, request: AnthropicRequest) -> bool:
        if not request.tools:
            return False
        for tool in request.tools:
            if isinstance(tool, dict) and 'type' in tool and 'input_schema' not in tool:
                return True
            if hasattr(tool, 'type') and not hasattr(tool, 'input_schema'):
                return True
        return False

    def scan_for_agent_routing(self, request: AnthropicRequest) -> Optional[str]:
        if not request.system:
            return None

        if isinstance(request.system, str):
            content = request.system
        elif isinstance(request.system, list) and request.system:
            content = request.system[-1].text
        else:
            return None

        first_line = content.strip().split('\n', 1)[0].strip()
        match = self.AGENT_PATTERN.match(first_line)
        return match.group(1) if match else None


class SimpleRouter:
    """Routes exchange requests to providers based on routing configuration."""

    def __init__(self, provider_manager: ProviderManager, routing_config: RoutingConfig, transformer_loader: TransformerLoader):
        self.provider_manager = provider_manager
        self.routing_config = routing_config
        self.transformer_loader = transformer_loader
        self.inspector = RequestInspector()
        self.default_provider: ProviderClient = self._load_default_provider()

    def _load_default_provider(self) -> ProviderClient:
        default_config = _create_default_anthropic_config()
        provider = ProviderClient(default_config, self.transformer_loader)
        logger.info("Loaded fallback provider '%s'", default_config.name)
        return provider

    def route(self, exchange_request: ExchangeRequest) -> RoutingResult:
        if exchange_request.channel != 'claude':
            return self._route_non_claude(exchange_request)

        payload = exchange_request.payload
        if not isinstance(payload, AnthropicRequest):
            raise TypeError('SimpleRouter currently supports AnthropicRequest payloads')

        is_direct_routing = False
        is_agent_routing = False
        used_fallback = False
        original_model = payload.model

        agent_model_alias = self.inspector.scan_for_agent_routing(payload)
        if self.inspector._has_builtin_tools(payload):
            routing_key = 'builtin_tools'
            model_alias = self._get_model_for_key(routing_key)
        elif agent_model_alias:
            model_alias = agent_model_alias
            routing_key = 'agent_direct'
            is_agent_routing = True
        elif payload.model.endswith('!'):
            model_alias = payload.model[:-1]
            routing_key = 'direct'
            is_direct_routing = True
        else:
            routing_key = self.inspector.determine_routing_key(payload)
            model_alias = self._get_model_for_key(routing_key)

        provider_binding = self.provider_manager.get_provider_for_model(model_alias)
        if provider_binding:
            provider, resolved_model_id = provider_binding
        else:
            provider = self.default_provider
            resolved_model_id = original_model
            used_fallback = True

        payload.model = resolved_model_id

        ctx = get_request_context()
        ctx.original_model = original_model
        ctx.update_routing_info(
            model_alias=model_alias,
            resolved_model_id=resolved_model_id,
            provider_name=provider.config.name,
            routing_key=routing_key,
            is_direct_routing=is_direct_routing,
            is_agent_routing=is_agent_routing,
            used_fallback=used_fallback,
        )
        ctx.extra['channel'] = exchange_request.channel

        exchange_request.metadata['routing_key'] = routing_key

        if used_fallback:
            logger.debug('Routed request to fallback provider for alias %s', model_alias)
        else:
            logger.debug('Routed request to provider %s alias %s', provider.config.name, model_alias)

        return RoutingResult(
            provider=provider,
            routing_key=routing_key,
            model_alias=model_alias,
            resolved_model_id=resolved_model_id,
            channel=exchange_request.channel,
            is_direct_routing=is_direct_routing,
            is_agent_routing=is_agent_routing,
            used_fallback=used_fallback,
        )

    def _route_non_claude(self, exchange_request: ExchangeRequest) -> RoutingResult:
        model_alias = exchange_request.model
        provider_binding = self.provider_manager.get_provider_for_model(model_alias)
        if not provider_binding:
            raise ValueError(f'Unknown model alias "{model_alias}" for channel {exchange_request.channel}')

        provider, resolved_model_id = provider_binding

        payload = exchange_request.payload
        if isinstance(payload, dict):
            payload['model'] = resolved_model_id

        exchange_request.metadata['routing_key'] = 'direct'

        ctx = get_request_context()
        ctx.update_routing_info(
            model_alias=model_alias,
            resolved_model_id=resolved_model_id,
            provider_name=provider.config.name,
            routing_key='direct',
            is_direct_routing=True,
            is_agent_routing=False,
            used_fallback=False,
        )
        ctx.extra['channel'] = exchange_request.channel

        return RoutingResult(
            provider=provider,
            routing_key='direct',
            model_alias=model_alias,
            resolved_model_id=resolved_model_id,
            channel=exchange_request.channel,
            is_direct_routing=True,
            is_agent_routing=False,
            used_fallback=False,
        )

    def _get_model_for_key(self, routing_key: str) -> str:
        if routing_key == 'builtin_tools':
            return self.routing_config.builtin_tools or self.routing_config.default
        if routing_key == 'planning':
            return self.routing_config.planning or self.routing_config.default
        if routing_key == 'background':
            return self.routing_config.background or self.routing_config.default
        if routing_key == 'thinking':
            return self.routing_config.thinking or self.routing_config.default
        if routing_key == 'plan_and_think':
            return self.routing_config.plan_and_think or self.routing_config.default
        return self.routing_config.default

    def list_available_models(self) -> list[str]:
        return self.provider_manager.list_models()

    def get_routing_info(self) -> dict:
        return {
            'default_model': self.routing_config.default,
            'planning_model': self.routing_config.planning,
            'background_model': self.routing_config.background,
            'thinking_model': self.routing_config.thinking,
            'plan_and_think_model': self.routing_config.plan_and_think,
            'builtin_tools_model': self.routing_config.builtin_tools,
            'available_models': self.list_available_models(),
            'providers': self.provider_manager.list_providers(),
        }

    async def close(self) -> None:
        await self.default_provider.close()
