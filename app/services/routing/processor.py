"""Request processor for routing requests to appropriate models and providers."""

from typing import Optional, Tuple

from app.common.models import ClaudeRequest
from app.config.log import get_logger
from app.config.user_models import UserConfig
from app.services.pipeline.messages_service import MessagesPipelineService
from app.services.registry.models import ModelRegistry
from app.services.registry.providers import ProviderRegistry
from app.services.routing.inspector import RequestInspector

logger = get_logger(__name__)


class RequestProcessor:
    """Processes requests and routes them to appropriate models and providers."""

    def __init__(self, user_config: UserConfig, model_registry: ModelRegistry, provider_registry: ProviderRegistry, inspector: Optional[RequestInspector] = None):
        """Initialize request processor.

        Args:
            user_config: User configuration with routing settings
            model_registry: Registry of available models
            provider_registry: Registry of available providers
            inspector: Request inspector for routing analysis
        """
        self.user_config = user_config
        self.model_registry = model_registry
        self.provider_registry = provider_registry
        self.inspector = inspector or RequestInspector()

        logger.debug(f'Initialized request processor with {model_registry.size()} models and {provider_registry.size()} providers')

    def process_request(self, request: ClaudeRequest) -> Tuple[str, str, Optional[MessagesPipelineService]]:
        """Process a request and determine routing.

        Args:
            request: Claude request to process

        Returns:
            Tuple of (routing_key, model_id, pipeline_service)
            pipeline_service may be None if provider not available or no configuration
        """
        try:
            # If no routing configuration, do passthrough
            if not self.user_config.routing:
                logger.debug('No routing configuration available, doing passthrough')
                return 'default', '', None

            # Determine routing key based on request content
            routing_key = self.inspector.determine_routing_key(request)

            # Get target model for routing key
            model_id = self.get_target_model(routing_key)
            if not model_id:
                logger.warning(f"No model configured for routing key '{routing_key}', falling back to default")
                routing_key = 'default'
                model_id = self.get_target_model('default')

            if not model_id:
                logger.error('No default model configured, doing passthrough')
                return routing_key, '', None

            # Get pipeline service for the model
            pipeline_service = self.get_pipeline_for_model(model_id)

            logger.info(f'Request routed: {routing_key} -> {model_id} -> {pipeline_service is not None}')
            return routing_key, model_id, pipeline_service

        except Exception as e:
            logger.error(f'Error processing request: {e}', exc_info=True)
            return 'default', '', None

    def get_target_model(self, routing_key: str) -> Optional[str]:
        """Get the target model for a routing key.

        Args:
            routing_key: Routing key ('default', 'planning', 'background')

        Returns:
            Model ID or None if not configured
        """
        if not self.user_config.routing:
            logger.debug('No routing configuration available')
            return None

        routing_config = self.user_config.routing

        if routing_key == 'default':
            return routing_config.default
        elif routing_key == 'planning':
            return routing_config.planning
        elif routing_key == 'background':
            return routing_config.background
        else:
            logger.warning(f"Unknown routing key '{routing_key}', falling back to default")
            return routing_config.default

    def get_pipeline_for_model(self, model_id: str) -> Optional[MessagesPipelineService]:
        """Get pipeline service for a specific model.

        Args:
            model_id: ID of the model

        Returns:
            Pipeline service or None if not available
        """
        # Get provider for the model
        provider_name = self.model_registry.get_provider_for_model(model_id)
        if not provider_name:
            logger.warning(f"No provider found for model '{model_id}'")
            return None

        # Get provider instance
        provider = self.provider_registry.get_provider_by_name(provider_name)
        if not provider:
            logger.warning(f"Provider '{provider_name}' not found in registry")
            return None

        # Create pipeline service for the provider
        return MessagesPipelineService(
            request_pipeline=provider.request_pipeline, response_pipeline=provider.response_pipeline, http_client=provider.http_client, sse_formatter=provider.sse_formatter
        )

    def get_provider_for_model(self, model_id: str) -> Optional[str]:
        """Get the provider name for a model.

        Args:
            model_id: ID of the model

        Returns:
            Provider name or None if not found
        """
        return self.model_registry.get_provider_for_model(model_id)

    def validate_routing_configuration(self) -> list[str]:
        """Validate that routing configuration references valid models.

        Returns:
            List of validation error messages
        """
        errors = []

        if not self.user_config.routing:
            errors.append('No routing configuration defined')
            return errors

        routing_config = self.user_config.routing

        # Check each routing type
        for routing_type, model_id in [('default', routing_config.default), ('planning', routing_config.planning), ('background', routing_config.background)]:
            if not self.model_registry.model_exists(model_id):
                errors.append(f"Routing '{routing_type}' references unknown model '{model_id}'")
            else:
                # Check that model has a valid provider
                provider_name = self.model_registry.get_provider_for_model(model_id)
                if not provider_name:
                    errors.append(f"Model '{model_id}' (used by {routing_type}) has no provider")
                elif not self.provider_registry.get_provider_by_name(provider_name):
                    errors.append(f"Model '{model_id}' references unknown provider '{provider_name}'")

        return errors

    def get_routing_summary(self) -> dict[str, any]:
        """Get a summary of the current routing configuration.

        Returns:
            Dictionary with routing information
        """
        summary = {
            'routing_configured': self.user_config.routing is not None,
            'models_available': self.model_registry.size(),
            'providers_available': self.provider_registry.size(),
            'inspector_stats': self.inspector.get_stats(),
            'routing_mapping': {},
        }

        if self.user_config.routing:
            routing_config = self.user_config.routing
            summary['routing_mapping'] = {
                'default': {'model': routing_config.default, 'provider': self.get_provider_for_model(routing_config.default)},
                'planning': {'model': routing_config.planning, 'provider': self.get_provider_for_model(routing_config.planning)},
                'background': {'model': routing_config.background, 'provider': self.get_provider_for_model(routing_config.background)},
            }

        return summary
