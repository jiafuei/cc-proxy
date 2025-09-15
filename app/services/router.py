"""Simple router system for the simplified architecture."""

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from app.common.models import AnthropicRequest
from app.common.vars import get_request_context
from app.config.log import get_logger
from app.config.user_models import ProviderConfig, RoutingConfig
from app.services.provider import Provider, ProviderManager
from app.services.transformer_loader import TransformerLoader

logger = get_logger(__name__)


@dataclass
class RoutingResult:
    """Complete routing information for a request."""

    provider: Provider
    routing_key: str
    model_alias: str
    resolved_model_id: str
    is_direct_routing: bool = False
    is_agent_routing: bool = False
    used_fallback: bool = False


def _create_default_anthropic_config() -> ProviderConfig:
    """Create a default Anthropic provider configuration from environment variables."""
    base_url = os.getenv('CCPROXY_FALLBACK_URL', 'https://api.anthropic.com/v1/messages')
    api_key = os.getenv('CCPROXY_FALLBACK_API_KEY', '')

    if not api_key:
        logger.warning('CCPROXY_FALLBACK_API_KEY not set - default provider will not work without authentication')

    return ProviderConfig(
        name='default-anthropic (fallback)',
        url=base_url,
        api_key=api_key,
        type='anthropic',
        capabilities=['messages', 'count_tokens'],
        transformers={
            'request': [
                {'class': 'app.services.transformers.anthropic.ClaudeSystemMessageCleanerTransformer', 'params': {}},
                {'class': 'app.services.transformers.anthropic.AnthropicCacheTransformer', 'params': {}},
                {'class': 'app.services.transformers.anthropic.AnthropicHeadersTransformer', 'params': {'auth_header': 'x-api-key'}},
                {'class': 'app.services.transformers.anthropic.ClaudeSoftwareEngineeringSystemMessageTransformer', 'params': {}},
                {'class': 'app.services.transformers.utils.ToolDescriptionOptimizerTransformer', 'params': {}},
            ],
            'response': [{'class': 'app.services.transformers.anthropic.AnthropicResponseTransformer', 'params': {}}],
        },
    )


class RequestInspector:
    """Analyzes requests to determine routing key."""

    # Compiled regex for efficient agent routing pattern matching
    AGENT_PATTERN = re.compile(r'^/model\s+([^\s]+)$')

    def __init__(self):
        """Initialize the request inspector."""
        pass

    def determine_routing_key(self, request: AnthropicRequest) -> str:
        """Determine routing key based on request content.

        Args:
            request: Anthropic API request

        Returns:
            Routing key ('builtin_tools', 'default', 'planning', 'background', 'thinking', 'plan_and_think')
        """
        # Check for built-in tools first (highest priority)
        if self._has_builtin_tools(request):
            return 'builtin_tools'

        if request.max_tokens and request.max_tokens < 768:
            return 'background'

        # Check for combined plan mode + thinking
        has_plan_mode = self._has_plan_mode_activation(request)
        has_thinking = self._has_thinking_config(request)

        if has_plan_mode and has_thinking:
            return 'plan_and_think'

        # Check for thinking only
        if has_thinking:
            return 'thinking'

        # Check for plan mode only
        if has_plan_mode:
            return 'planning'

        # Default routing
        return 'default'

    def _has_thinking_config(self, request: AnthropicRequest) -> bool:
        """Check if the request has thinking configuration with budget tokens > 0.

        Args:
            request: Anthropic API request

        Returns:
            True if thinking config exists and budget_tokens > 0
        """
        return request.thinking is not None and request.thinking.budget_tokens > 0

    def _has_plan_mode_activation(self, request: AnthropicRequest) -> bool:
        """Check if the last user message contains plan mode activation text.

        Args:
            request: Anthropic API request

        Returns:
            True if plan mode activation text is found in the last user message
        """
        plan_mode_text = '<system-reminder>\nPlan mode is active.'

        # Find the last user message
        last_user_message = None
        for message in reversed(request.messages):
            if message.role == 'user':
                last_user_message = message
                break

        if not last_user_message:
            return False

        # Check content blocks in the last user message
        content = last_user_message.content
        if isinstance(content, str):
            return plan_mode_text in content
        elif isinstance(content, list):
            for block in content:
                if hasattr(block, 'text') and block.text and plan_mode_text in block.text:
                    return True
                elif block.type == 'tool_result' and plan_mode_text in block.content:
                    return True

        return False

    def _has_builtin_tools(self, request: AnthropicRequest) -> bool:
        """Check if the request contains built-in tool definitions.

        Built-in tools are identified by having a 'type' field but no 'input_schema',
        which distinguishes them from regular tool definitions.

        Args:
            request: Anthropic API request

        Returns:
            True if the request contains built-in tools, False otherwise
        """
        if not request.tools:
            return False

        # Check if any tool has 'type' field but no 'input_schema' (built-in tool indicator)
        for tool in request.tools:
            # Handle both dict and Pydantic object cases
            if isinstance(tool, dict):
                if 'type' in tool and 'input_schema' not in tool:
                    return True
            else:
                # Pydantic object - check if it has 'type' attribute but no 'input_schema'
                if hasattr(tool, 'type') and not hasattr(tool, 'input_schema'):
                    return True

        return False

    def _scan_for_agent_routing(self, request: AnthropicRequest) -> Optional[str]:
        """Scan last system message for agent routing pattern.

        Args:
            request: Anthropic API request

        Returns:
            Model alias if /model <model-alias> found as first line, None otherwise
        """
        if not request.system:
            return None

        # Get last system message content
        if isinstance(request.system, str):
            content = request.system
        elif isinstance(request.system, list) and request.system:
            content = request.system[-1].text
        else:
            return None

        # Check first line after trimming
        trimmed = content.strip()
        if not trimmed:
            return None

        first_line = trimmed.split('\n', 1)[0].strip()
        match = self.AGENT_PATTERN.match(first_line)
        return match.group(1) if match else None


class SimpleRouter:
    """Simple router that maps requests to providers based on routing configuration."""

    def __init__(self, provider_manager: ProviderManager, routing_config: RoutingConfig, transformer_loader: TransformerLoader):
        self.provider_manager = provider_manager
        self.routing_config = routing_config
        self.transformer_loader = transformer_loader
        self.inspector = RequestInspector()
        self.default_provider: Provider = None  # Will be set by _load_default_provider
        self._load_default_provider()

    def _load_default_provider(self):
        """Load the default Anthropic provider as fallback."""
        default_config = _create_default_anthropic_config()
        self.default_provider = Provider(default_config, self.transformer_loader)
        logger.info(f"Loaded default provider '{default_config.name}'")

    def get_provider_for_request(self, request: AnthropicRequest) -> RoutingResult:
        """Get the appropriate provider for a request.

        Args:
            request: Anthropic API request

        Returns:
            RoutingResult with complete routing information
        """
        # Initialize routing flags
        is_direct_routing = False
        is_agent_routing = False
        used_fallback = False
        original_model = request.model

        # Check for built-in tools first (highest priority)
        if self.inspector._has_builtin_tools(request):
            routing_key = 'builtin_tools'
            model_alias = self._get_model_for_key(routing_key)
            logger.debug(f'Built-in tools detected, routing to: {model_alias}')
        # Check for agent routing in system message
        elif agent_model_alias := self.inspector._scan_for_agent_routing(request):
            model_alias = agent_model_alias
            routing_key = 'agent_direct'
            is_agent_routing = True
            logger.debug(f'Agent routing detected in system message: /model {model_alias}')
        # Check for direct routing with '!' suffix
        elif request.model.endswith('!'):
            model_alias = request.model[:-1]  # Strip '!'
            routing_key = 'direct'
            is_direct_routing = True
            logger.debug(f'Direct routing detected: {request.model} -> {model_alias}')
        else:
            # 1. Determine routing key based on request content
            routing_key = self.inspector.determine_routing_key(request)
            logger.debug(f'Determined routing key: {routing_key}')

            # 2. Get model alias for routing key (guaranteed to return a value)
            model_alias = self._get_model_for_key(routing_key)

        # 3. Try to get configured provider for model alias
        provider_result = self.provider_manager.get_provider_for_model(model_alias)
        if provider_result:
            provider, resolved_model_id = provider_result
            # Update request.model to the resolved model ID
            request.model = resolved_model_id
        else:
            # 4. Use default provider as fallback (guaranteed to exist)
            provider = self.default_provider
            resolved_model_id = original_model  # Keep original model for fallback
            used_fallback = True

        # Build result
        result = RoutingResult(
            provider=provider,
            routing_key=routing_key,
            model_alias=model_alias,
            resolved_model_id=resolved_model_id,
            is_direct_routing=is_direct_routing,
            is_agent_routing=is_agent_routing,
            used_fallback=used_fallback,
        )

        # Update request context
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

        # Structured logging with context (context fields will be auto-added)
        if routing_key == 'agent_direct':
            if used_fallback:
                logger.debug(f'Agent routing to fallback: /model {model_alias} -> {model_alias} -> {resolved_model_id} (unchanged) -> {provider.config.name}')
            else:
                logger.debug(f'Agent routing: /model {model_alias} -> {model_alias} -> {resolved_model_id} -> {provider.config.name}')
        elif routing_key == 'direct':
            if used_fallback:
                logger.debug(f'Direct routing to fallback: {model_alias}! -> {model_alias} -> {resolved_model_id} (unchanged) -> {provider.config.name}')
            else:
                logger.debug(f'Direct routing: {model_alias}! -> {model_alias} -> {resolved_model_id} -> {provider.config.name}')
        else:
            if used_fallback:
                logger.debug(f'Routed request to fallback: {routing_key} -> {model_alias} -> {resolved_model_id} (unchanged) -> {provider.config.name}')
            else:
                logger.debug(f'Routed request: {routing_key} -> {model_alias} -> {resolved_model_id} -> {provider.config.name}')

        return result

    def _get_model_for_key(self, routing_key: str) -> str:
        """Get model alias for a routing key."""
        if routing_key == 'builtin_tools':
            return self.routing_config.builtin_tools
        elif routing_key == 'planning':
            return self.routing_config.planning
        elif routing_key == 'background':
            return self.routing_config.background
        elif routing_key == 'thinking':
            return self.routing_config.thinking
        elif routing_key == 'plan_and_think':
            return self.routing_config.plan_and_think
        else:
            return self.routing_config.default

    def get_provider_for_model(self, alias: str) -> Optional[Tuple[Provider, str]]:
        """Get provider that supports a specific model alias.

        Args:
            alias: Model alias

        Returns:
            Tuple (Provider, resolved_model_id) or None if not found
        """
        return self.provider_manager.get_provider_for_model(alias)

    def list_available_models(self) -> list[str]:
        """List all available models across all providers."""
        return self.provider_manager.list_models()

    def get_routing_info(self) -> dict:
        """Get information about current routing configuration."""
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

    async def close(self):
        """Clean up resources."""
        await self.default_provider.close()
