"""Simple router system for the simplified architecture."""

from typing import Optional

from app.common.models import ClaudeRequest
from app.config.log import get_logger
from app.config.user_models import RoutingConfig
from app.services.provider import Provider, ProviderManager

logger = get_logger(__name__)


class RequestInspector:
    """Analyzes requests to determine routing key."""

    def __init__(self):
        """Initialize with routing keyword configuration."""
        self.routing_keywords = {
            'planning': ['plan', 'strategy', 'approach', 'steps', 'methodology', 'design', 'architecture', 'roadmap', 'timeline'],
            'background': ['analyze', 'review', 'summarize', 'extract', 'process', 'batch', 'bulk', 'generate report', 'data analysis']
        }

    def determine_routing_key(self, request: ClaudeRequest) -> str:
        """Determine routing key based on request content.

        Args:
            request: Claude API request

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

    def _extract_request_text(self, request: ClaudeRequest) -> str:
        """Extract all text content from request for analysis.
        
        Args:
            request: Claude API request
            
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

    def __init__(self, provider_manager: ProviderManager, routing_config: RoutingConfig):
        self.provider_manager = provider_manager
        self.routing_config = routing_config
        self.inspector = RequestInspector()

    def get_provider_for_request(self, request: ClaudeRequest) -> Optional[Provider]:
        """Get the appropriate provider for a request.

        Args:
            request: Claude API request

        Returns:
            Provider instance or None if no suitable provider found
        """
        try:
            # 1. Determine routing key based on request content
            routing_key = self.inspector.determine_routing_key(request)
            logger.debug(f'Determined routing key: {routing_key}')

            # 2. Get model for routing key
            model_id = self._get_model_for_key(routing_key)
            if not model_id:
                logger.warning(f"No model configured for routing key '{routing_key}'")
                return None

            # 3. Get provider for model
            provider = self.provider_manager.get_provider_for_model(model_id)
            if not provider:
                logger.warning(f"No provider found for model '{model_id}' - no default provider available")
                return None

            logger.info(f'Routed request: {routing_key} -> {model_id} -> {provider.config.name}')
            return provider

        except Exception as e:
            logger.error(f'Error in request routing: {e}', exc_info=True)
            return None

    def _get_model_for_key(self, routing_key: str) -> str:
        """Get model ID for a routing key."""
        if routing_key == 'planning':
            return self.routing_config.planning
        elif routing_key == 'background':
            return self.routing_config.background
        else:
            return self.routing_config.default

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
