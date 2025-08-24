"""Simplified dependency injection for FastAPI routes."""

from typing import Optional

import httpx

from app.common.dumper import Dumper
from app.config import get_config
from app.config.log import get_logger
from app.config.user_models import UserConfig
from app.services.config.simple_user_config_manager import get_user_config_manager
from app.services.error_handling.error_formatter import ApiErrorFormatter
from app.services.error_handling.exception_mapper import HttpExceptionMapper
from app.services.pipeline.http_client import HttpClientService
from app.services.pipeline.messages_service import MessagesPipelineService
from app.services.pipeline.request_pipeline import RequestPipeline
from app.services.pipeline.response_pipeline import ResponsePipeline
from app.services.registry.models import ModelRegistry
from app.services.registry.providers import ProviderFactory, ProviderRegistry
from app.services.registry.transformers import TransformerRegistry
from app.services.routing.processor import RequestProcessor
from app.services.sse_formatter.anthropic_formatter import AnthropicSseFormatter
from app.services.transformers.anthropic.transformers import (
    AnthropicRequestTransformer,
    AnthropicResponseTransformer,
    AnthropicStreamTransformer,
)

logger = get_logger(__name__)


class ServiceContainer:
    """Container for all application services built from user configuration."""

    def __init__(self):
        self.app_config = get_config()
        self.user_config = None

        # Core services (always available)
        self.dumper = Dumper(self.app_config)
        self.exception_mapper = HttpExceptionMapper()
        self.error_formatter = ApiErrorFormatter()

        # HTTP client (shared across providers)
        self.httpx_client = httpx.AsyncClient(timeout=60 * 5, http2=True)

        # Registries (built from user config)
        self.transformer_registry: Optional[TransformerRegistry] = None
        self.provider_registry: Optional[ProviderRegistry] = None
        self.model_registry: Optional[ModelRegistry] = None
        self.request_processor: Optional[RequestProcessor] = None

        # Build services from user configuration
        self._initialize_from_config()

        # Set up config change callback
        config_manager = get_user_config_manager()
        config_manager.on_config_change(self._on_config_change)

    def _initialize_from_config(self):
        """Initialize services from user configuration."""
        try:
            config_manager = get_user_config_manager()
            self.user_config = config_manager.load_config()
            self._build_services(self.user_config)
            logger.info('Services initialized from user configuration successfully')
        except Exception as e:
            logger.error(f'Failed to initialize from user config: {e}', exc_info=True)
            # Initialize with empty config as fallback
            from app.config.user_models import UserConfig

            self.user_config = UserConfig()
            self._build_services(self.user_config)

    def _build_services(self, user_config: UserConfig):
        """Build all services from user configuration."""
        logger.info('Building services from user configuration')

        # 1. Build transformer registry with defaults
        self.transformer_registry = self._build_transformer_registry(user_config)

        # 2. Build provider registry
        self.provider_registry = self._build_provider_registry(user_config)

        # 3. Build model registry
        self.model_registry = self._build_model_registry(user_config)

        # 4. Build request processor for routing
        self.request_processor = RequestProcessor(user_config=user_config, model_registry=self.model_registry, provider_registry=self.provider_registry)

        logger.info(f'Built services: {self.transformer_registry.size()} transformers, {self.provider_registry.size()} providers, {self.model_registry.size()} models')

    def _build_transformer_registry(self, user_config: UserConfig) -> TransformerRegistry:
        """Build transformer registry with default Anthropic transformers."""
        registry = TransformerRegistry()

        # Load custom transformers from user config
        success_count = registry.load_transformers_from_config(user_config.transformers)
        logger.debug(f'Loaded {success_count} custom transformers')

        return registry

    def _build_provider_registry(self, user_config: UserConfig) -> ProviderRegistry:
        """Build provider registry with configured providers."""
        # Create built-in default transformers
        builtin_transformers = {
            'anthropic-request-transformer': AnthropicRequestTransformer(self.app_config),
            'anthropic-response-transformer': AnthropicResponseTransformer(),
            'anthropic-stream-response-transformer': AnthropicStreamTransformer(),
        }

        # Create provider factory
        provider_factory = ProviderFactory(self.transformer_registry, builtin_transformers)
        registry = ProviderRegistry(provider_factory)

        # Register providers from user config
        success_count = 0
        for provider_config in user_config.providers:
            try:
                http_client = HttpClientService(self.httpx_client)
                sse_formatter = AnthropicSseFormatter()  # TODO: make configurable per provider

                if registry.register_provider_from_config(provider_config, http_client, sse_formatter):
                    success_count += 1
                    logger.debug(f"Registered provider '{provider_config.name}'")
            except Exception as e:
                logger.error(f"Failed to register provider '{provider_config.name}': {e}", exc_info=True)

        logger.info(f'Registered {success_count}/{len(user_config.providers)} providers successfully')
        return registry

    def _build_model_registry(self, user_config: UserConfig) -> ModelRegistry:
        """Build model registry with model-to-provider mappings."""
        registry = ModelRegistry()
        success_count = registry.register_models_from_config(user_config.models)
        logger.debug(f'Registered {success_count} models')
        return registry

    def _on_config_change(self, user_config: UserConfig):
        """Handle user configuration changes."""
        try:
            logger.info('User configuration changed, rebuilding services')
            self.user_config = user_config
            self._build_services(user_config)
            logger.info('Services rebuilt successfully after config change')
        except Exception as e:
            logger.error(f'Failed to rebuild services after config change: {e}', exc_info=True)

    def get_pipeline_for_model(self, model_id: str) -> Optional[MessagesPipelineService]:
        """Get pipeline service for a specific model."""
        if not self.model_registry or not self.provider_registry:
            return None

        provider_name = self.model_registry.get_provider_for_model(model_id)
        if not provider_name:
            return None

        provider = self.provider_registry.get_provider_by_name(provider_name)
        if not provider:
            return None

        return MessagesPipelineService(
            request_pipeline=provider.request_pipeline, response_pipeline=provider.response_pipeline, http_client=provider.http_client, sse_formatter=provider.sse_formatter
        )

    def create_default_pipeline(self) -> MessagesPipelineService:
        """Create a default Anthropic pipeline for fallback scenarios."""
        # Create default Anthropic transformers
        request_transformer = AnthropicRequestTransformer(self.app_config)
        response_transformer = AnthropicResponseTransformer()
        stream_transformer = AnthropicStreamTransformer()

        # Create pipelines
        request_pipeline = RequestPipeline([request_transformer])
        response_pipeline = ResponsePipeline([response_transformer], [stream_transformer])

        # Create HTTP client and SSE formatter
        http_client = HttpClientService(self.httpx_client)
        sse_formatter = AnthropicSseFormatter()

        return MessagesPipelineService(request_pipeline=request_pipeline, response_pipeline=response_pipeline, http_client=http_client, sse_formatter=sse_formatter)


# Global service container
_service_container: Optional[ServiceContainer] = None


def get_service_container() -> ServiceContainer:
    """Get the global service container."""
    global _service_container

    if _service_container is None:
        _service_container = ServiceContainer()

    return _service_container


# Dependency injection functions for FastAPI routes
def get_core_services():
    """Get core services (error handling, dumper, etc.)."""
    container = get_service_container()
    return type('CoreServices', (), {'dumper': container.dumper, 'exception_mapper': container.exception_mapper, 'error_formatter': container.error_formatter})()


def get_routing_service():
    """Get request processor for routing."""
    container = get_service_container()
    return container.request_processor


def get_default_pipeline_service():
    """Get default Anthropic pipeline service."""
    container = get_service_container()
    return container.create_default_pipeline()
