"""Simple router system for the simplified architecture."""

from typing import Optional

from app.common.models import ClaudeRequest
from app.config.log import get_logger
from app.services.provider import Provider, ProviderManager

logger = get_logger(__name__)


class RoutingConfig:
    """Configuration for routing requests to different models."""

    def __init__(self, config: dict):
        self.default = config.get('default', '')
        self.planning = config.get('planning', self.default)
        self.background = config.get('background', self.default)

    def get_model_for_key(self, routing_key: str) -> str:
        """Get model ID for a routing key."""
        if routing_key == 'planning':
            return self.planning
        elif routing_key == 'background':
            return self.background
        else:
            return self.default


class RequestInspector:
    """Analyzes requests to determine routing key."""

    def determine_routing_key(self, request: ClaudeRequest) -> str:
        """Determine routing key based on request content.

        Args:
            request: Claude API request

        Returns:
            Routing key ('default', 'planning', 'background')
        """
        # Simple routing logic based on request content
        # Users can extend this by creating custom inspectors

        # Check if request contains planning-related keywords
        if self._is_planning_request(request):
            return 'planning'

        # Check if request is suitable for background processing
        if self._is_background_request(request):
            return 'background'

        # Default routing
        return 'default'

    def _is_planning_request(self, request: ClaudeRequest) -> bool:
        """Check if request appears to be planning-related."""
        planning_keywords = ['plan', 'strategy', 'approach', 'steps', 'methodology', 'design', 'architecture', 'roadmap', 'timeline']

        # Check system messages
        if request.system:
            for msg in request.system:
                if any(keyword in msg.text.lower() for keyword in planning_keywords):
                    return True

        # Check user messages
        for message in request.messages:
            if message.role == 'user':
                content = message.content
                if isinstance(content, str):
                    if any(keyword in content.lower() for keyword in planning_keywords):
                        return True
                elif isinstance(content, list):
                    for block in content:
                        if hasattr(block, 'text') and block.text:
                            if any(keyword in block.text.lower() for keyword in planning_keywords):
                                return True

        return False

    def _is_background_request(self, request: ClaudeRequest) -> bool:
        """Check if request is suitable for background processing."""
        background_keywords = ['analyze', 'review', 'summarize', 'extract', 'process', 'batch', 'bulk', 'generate report', 'data analysis']

        # Check system messages
        if request.system:
            for msg in request.system:
                if any(keyword in msg.text.lower() for keyword in background_keywords):
                    return True

        # Check user messages
        for message in request.messages:
            if message.role == 'user':
                content = message.content
                if isinstance(content, str):
                    if any(keyword in content.lower() for keyword in background_keywords):
                        return True
                elif isinstance(content, list):
                    for block in content:
                        if hasattr(block, 'text') and block.text:
                            if any(keyword in block.text.lower() for keyword in background_keywords):
                                return True

        return False


class SimpleRouter:
    """Simple router that maps requests to providers based on routing configuration."""

    def __init__(self, provider_manager: ProviderManager, routing_config: dict):
        self.provider_manager = provider_manager
        self.routing_config = RoutingConfig(routing_config)
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
            model_id = self.routing_config.get_model_for_key(routing_key)
            if not model_id:
                logger.warning(f"No model configured for routing key '{routing_key}'")
                return None

            # 3. Get provider for model
            provider = self.provider_manager.get_provider_for_model(model_id)
            if not provider:
                logger.warning(f"No provider found for model '{model_id}'")
                return None

            logger.info(f'Routed request: {routing_key} -> {model_id} -> {provider.config.name}')
            return provider

        except Exception as e:
            logger.error(f'Error in request routing: {e}', exc_info=True)
            return None

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
