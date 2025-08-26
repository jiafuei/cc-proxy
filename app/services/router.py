"""Simple router system for the simplified architecture."""

import os
from typing import Optional, Tuple

from app.common.models import AnthropicRequest
from app.config.log import get_logger
from app.config.user_models import ProviderConfig, RoutingConfig
from app.services.provider import Provider, ProviderManager
from app.services.transformer_loader import TransformerLoader

logger = get_logger(__name__)

OPUS_MODEL_ID = 'claude-opus-4-1-20250805'
SONNET_MODEL_ID = 'claude-sonnet-4-20250514'


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
        models=[
            OPUS_MODEL_ID,
            SONNET_MODEL_ID,
            'claude-3-5-haiku-20241022',
        ],
        transformers={
            'request': [{'class': 'app.services.transformers.anthropic.AnthropicAuthTransformer', 'params': {'api_key': api_key, 'base_url': base_url}}] if api_key else [],
            'response': [],
        },
        timeout=300,
    )


class RequestInspector:
    """Analyzes requests to determine routing key."""

    def __init__(self):
        """Initialize with routing keyword configuration."""
        self.routing_keywords = {
            'planning': ['plan', 'strategy', 'approach', 'steps', 'methodology', 'design', 'architecture', 'roadmap', 'timeline'],
            'background': ['analyze', 'review', 'summarize', 'extract', 'process', 'batch', 'bulk', 'generate report', 'data analysis'],
        }

    def determine_routing_key(self, request: AnthropicRequest) -> str:
        """Determine routing key based on request content.

        Args:
            request: Anthropic API request

        Returns:
            Routing key ('default', 'planning', 'background')
        """
        # Extract all text content from the request
        request_text = self._extract_request_text(request)

        # Check for routing keywords in priority order
        for routing_type, keywords in self.routing_keywords.items():
            if self._contains_keywords(request_text, keywords):
                return routing_type

        # Default routing
        return 'default'

    def _extract_request_text(self, request: AnthropicRequest) -> str:
        """Extract all text content from request for analysis.

        Args:
            request: Anthropic API request

        Returns:
            Concatenated lowercase text from all messages
        """
        texts = []

        # Extract system message text
        if request.system:
            for msg in request.system:
                texts.append(msg.text.lower())

        # Extract user message text
        for message in request.messages:
            if message.role == 'user':
                content = message.content
                if isinstance(content, str):
                    texts.append(content.lower())
                elif isinstance(content, list):
                    for block in content:
                        if hasattr(block, 'text') and block.text:
                            texts.append(block.text.lower())

        return ' '.join(texts)

    def _contains_keywords(self, text: str, keywords: list[str]) -> bool:
        """Check if text contains any of the specified keywords.

        Args:
            text: Text to search in (should be lowercase)
            keywords: List of keywords to search for

        Returns:
            True if any keyword is found
        """
        return any(keyword in text for keyword in keywords)


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
        logger.info(f"Loaded default provider '{default_config.name}' with {len(default_config.models)} models")

    def get_provider_for_request(self, request: AnthropicRequest) -> Tuple[Provider, str]:
        """Get the appropriate provider for a request.

        Args:
            request: Anthropic API request

        Returns:
            Tuple (Provider, routing_key)
        """
        # 1. Determine routing key based on request content
        routing_key = self.inspector.determine_routing_key(request)
        logger.debug(f'Determined routing key: {routing_key}')

        # 2. Get model for routing key (guaranteed to return a value)
        model_id = self._get_model_for_key(routing_key)

        # 3. Try to get configured provider for model
        provider = self.provider_manager.get_provider_for_model(model_id)
        if provider:
            logger.info(f'Routed request: {routing_key} -> {model_id} -> {provider.config.name}')
            return provider

        # 4. Use default provider as fallback (guaranteed to exist)
        logger.info(f'Routed request to fallback: {routing_key} -> {model_id} -> {self.default_provider.config.name}')
        return self.default_provider, routing_key

    def _get_model_for_key(self, routing_key: str) -> str:
        """Get model ID for a routing key."""
        if routing_key == 'planning':
            return self.routing_config.planning or SONNET_MODEL_ID
        elif routing_key == 'background':
            return self.routing_config.background or SONNET_MODEL_ID
        else:
            return self.routing_config.default or SONNET_MODEL_ID

    def get_provider_for_model(self, model_id: str) -> Optional[Provider]:
        """Get provider that supports a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Provider instance or None if not found
        """
        return self.provider_manager.get_provider_for_model(model_id)

    def list_available_models(self) -> list[str]:
        """List all available models across all providers."""
        return self.provider_manager.list_models()

    def get_routing_info(self) -> dict:
        """Get information about current routing configuration."""
        return {
            'default_model': self.routing_config.default,
            'planning_model': self.routing_config.planning,
            'background_model': self.routing_config.background,
            'available_models': self.list_available_models(),
            'providers': self.provider_manager.list_providers(),
        }

    async def close(self):
        """Clean up resources."""
        await self.default_provider.close()
