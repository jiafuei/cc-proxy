"""Service builder for creating services from user configuration."""

import logging
from typing import Dict, List, Optional, Tuple

import httpx

from app.common.dumper import Dumper
from app.config.models import ConfigModel
from app.config.user_models import UserConfig
from app.services.config.interfaces import ServiceBuilder
from app.services.error_handling.error_formatter import ApiErrorFormatter
from app.services.error_handling.exception_mapper import HttpExceptionMapper
from app.services.pipeline.http_client import HttpClientService
from app.services.pipeline.messages_service import MessagesPipelineService
from app.services.registry.models import ModelRegistry
from app.services.registry.providers import ProviderRegistry
from app.services.registry.transformers import TransformerRegistry
from app.services.sse_formatter.anthropic_formatter import AnthropicSseFormatter
from app.services.transformers.anthropic.transformers import (
    AnthropicRequestTransformer,
    AnthropicResponseTransformer,
    AnthropicStreamTransformer,
)

logger = logging.getLogger(__name__)


class DynamicServiceBuilder(ServiceBuilder):
    """Builder for creating services from user configuration with dynamic components."""

    def __init__(self, app_config: ConfigModel):
        """Initialize with application configuration.

        Args:
            app_config: Static application configuration
        """
        self.app_config = app_config
        self.httpx_client = httpx.AsyncClient(timeout=60 * 5, http2=True)

    def build_services(self, user_config: UserConfig) -> 'DynamicServices':
        """Build services from user configuration.

        Args:
            user_config: User configuration to build from

        Returns:
            Built services instance
        """
        logger.info('Building services from user configuration')

        # Create registries and load components
        transformer_registry = self._build_transformer_registry(user_config)
        provider_registry = self._build_provider_registry(user_config, transformer_registry)
        model_registry = self._build_model_registry(user_config)

        # Create services instance with registries
        services = DynamicServices(
            app_config=self.app_config,
            user_config=user_config,
            httpx_client=self.httpx_client,
            transformer_registry=transformer_registry,
            provider_registry=provider_registry,
            model_registry=model_registry,
        )

        logger.info(f'Successfully built services with {transformer_registry.size()} transformers, {provider_registry.size()} providers, {model_registry.size()} models')
        return services

    def _build_transformer_registry(self, user_config: UserConfig) -> TransformerRegistry:
        """Build transformer registry from configuration."""
        registry = TransformerRegistry()

        # Load custom transformers
        success_count = registry.load_transformers_from_config(user_config.transformers)
        logger.debug(f'Loaded {success_count} custom transformers')

        return registry

    def _build_provider_registry(self, user_config: UserConfig, transformer_registry: TransformerRegistry) -> ProviderRegistry:
        """Build provider registry from configuration."""
        from app.services.registry.providers import ProviderFactory

        # Create built-in transformers
        builtin_transformers = self._create_builtin_transformers()

        # Create provider factory
        provider_factory = ProviderFactory(transformer_registry, builtin_transformers)

        # Create registry
        registry = ProviderRegistry(provider_factory)

        # Register each provider with its own HTTP client and SSE formatter
        success_count = 0
        for provider_config in user_config.providers:
            try:
                # Create dedicated HTTP client for this provider
                # This allows per-provider settings like timeouts, retries, etc.
                http_client = self._create_http_client_for_provider(provider_config)

                # Create SSE formatter for this provider
                # This could be made configurable per provider in the future
                sse_formatter = self._create_sse_formatter_for_provider(provider_config)

                # Register the provider
                if registry.register_provider_from_config(provider_config, http_client, sse_formatter):
                    success_count += 1
                    logger.info(f"Successfully registered provider '{provider_config.name}' with custom client")
                else:
                    logger.error(f"Failed to register provider '{provider_config.name}'")

            except Exception as e:
                logger.error(f"Error setting up provider '{provider_config.name}': {e}", exc_info=True)

        logger.info(f'Loaded {success_count}/{len(user_config.providers)} providers successfully')
        return registry

    def _build_model_registry(self, user_config: UserConfig) -> ModelRegistry:
        """Build model registry from configuration."""
        registry = ModelRegistry()

        success_count = registry.register_models_from_config(user_config.models)
        logger.debug(f'Loaded {success_count} models')

        return registry

    def _create_builtin_transformers(self) -> Dict[str, any]:
        """Create built-in transformers."""
        return {
            'anthropic-request-transformer': AnthropicRequestTransformer(self.app_config),
            'anthropic-response-transformer': AnthropicResponseTransformer(),
            'anthropic-stream-response-transformer': AnthropicStreamTransformer(),
        }

    def _create_http_client_for_provider(self, provider_config) -> HttpClientService:
        """Create HTTP client for a specific provider.

        Args:
            provider_config: Provider configuration

        Returns:
            HTTP client service configured for the provider
        """
        # For now, use the shared HTTP client, but this could be enhanced to:
        # - Set provider-specific timeouts
        # - Configure retries per provider
        # - Set custom headers or authentication
        # - Configure connection pooling per provider

        return HttpClientService(self.httpx_client)

    def _create_sse_formatter_for_provider(self, provider_config) -> AnthropicSseFormatter:
        """Create SSE formatter for a specific provider.

        Args:
            provider_config: Provider configuration

        Returns:
            SSE formatter for the provider
        """
        # For now, use Anthropic formatter for all providers
        # This could be enhanced to:
        # - Detect provider type and use appropriate formatter
        # - Support custom formatters per provider
        # - Configure formatting options per provider

        return AnthropicSseFormatter()


class DynamicServices:
    """Extended Services class that supports dynamic user configuration."""

    def __init__(
        self,
        app_config: ConfigModel,
        user_config: UserConfig,
        httpx_client: httpx.AsyncClient,
        transformer_registry: TransformerRegistry,
        provider_registry: ProviderRegistry,
        model_registry: ModelRegistry,
    ):
        """Initialize dynamic services.

        Args:
            app_config: Static application configuration
            user_config: User configuration
            httpx_client: HTTP client instance
            transformer_registry: Registry of transformers
            provider_registry: Registry of providers
            model_registry: Registry of models
        """
        # Don't call super().__init__() to avoid creating hardcoded services
        self.config = app_config
        self.user_config = user_config
        self.httpx_client = httpx_client

        # Store registries
        self.transformer_registry = transformer_registry
        self.provider_registry = provider_registry
        self.model_registry = model_registry

        # Create services
        self._create_dynamic_services()

        # Create request processor for routing
        self._create_request_processor()

    def _create_request_processor(self):
        """Create request processor for handling routing."""
        from app.services.routing.processor import RequestProcessor

        self.request_processor = RequestProcessor(user_config=self.user_config, model_registry=self.model_registry, provider_registry=self.provider_registry)

    def _create_dynamic_services(self):
        """Create services with dynamic configuration support."""
        # Create default pipeline service (fallback to built-in Anthropic)
        self._create_default_pipeline_service()

        # Initialize error handling services (these remain static)
        self.exception_mapper = HttpExceptionMapper()
        self.error_formatter = ApiErrorFormatter()

        # Other services
        self.dumper = Dumper(self.config)

    def _create_default_pipeline_service(self):
        """Create default pipeline service as fallback."""
        # Use built-in Anthropic transformers as default
        anthropic_request_transformer = AnthropicRequestTransformer(self.config)
        anthropic_response_transformer = AnthropicResponseTransformer()
        anthropic_stream_transformer = AnthropicStreamTransformer()

        # Create pipelines
        from app.services.pipeline.request_pipeline import RequestPipeline
        from app.services.pipeline.response_pipeline import ResponsePipeline

        request_pipeline = RequestPipeline([anthropic_request_transformer])
        response_pipeline = ResponsePipeline([anthropic_response_transformer], [anthropic_stream_transformer])

        # Create HTTP client
        http_client = HttpClientService(self.httpx_client)

        # Create SSE formatter
        sse_formatter = AnthropicSseFormatter()

        # Create default pipeline service
        self.messages_pipeline = MessagesPipelineService(request_pipeline, response_pipeline, http_client, sse_formatter)

    def get_provider_pipeline(self, provider_name: str) -> Optional[MessagesPipelineService]:
        """Get pipeline service for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Pipeline service for the provider or None if not found
        """
        provider = self.provider_registry.get_provider_by_name(provider_name)
        if not provider:
            return None

        # Create pipeline service for the provider
        return MessagesPipelineService(
            request_pipeline=provider.request_pipeline, response_pipeline=provider.response_pipeline, http_client=provider.http_client, sse_formatter=provider.sse_formatter
        )

    def get_pipeline_for_model(self, model_id: str) -> Optional[MessagesPipelineService]:
        """Get pipeline service for a specific model.

        Args:
            model_id: ID of the model

        Returns:
            Pipeline service for the model's provider or None if not found
        """
        provider_name = self.model_registry.get_provider_for_model(model_id)
        if not provider_name:
            return None

        return self.get_provider_pipeline(provider_name)

    def process_request_with_routing(self, request) -> Tuple[str, str, Optional[MessagesPipelineService]]:
        """Process request using dynamic routing.

        Args:
            request: Claude request to process

        Returns:
            Tuple of (routing_key, model_id, pipeline_service)
        """
        return self.request_processor.process_request(request)

    def get_routing_summary(self) -> Dict[str, any]:
        """Get routing configuration summary."""
        return self.request_processor.get_routing_summary()

    def validate_configuration(self) -> List[str]:
        """Validate the current configuration.

        Returns:
            List of validation error messages
        """
        errors = []

        # Validate routing references
        routing_errors = self.request_processor.validate_routing_configuration()
        errors.extend(routing_errors)

        # Validate model-provider references
        model_errors = self.model_registry.validate_model_references(self.provider_registry.list_provider_names())
        errors.extend(model_errors)

        return errors
