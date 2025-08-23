"""Provider registry and factory system for managing custom API providers."""

import logging
from typing import Dict, List, Optional

from app.config.user_models import PipelineTransformerConfig, ProviderConfig
from app.services.config.interfaces import ComponentRegistry
from app.services.pipeline.http_client import HttpClientService
from app.services.pipeline.request_pipeline import RequestPipeline
from app.services.pipeline.response_pipeline import ResponsePipeline
from app.services.sse_formatter.interfaces import SseFormatter

from .transformers import TransformerRegistry

logger = logging.getLogger(__name__)


class Provider:
    """Represents a configured API provider with its pipelines."""

    def __init__(
        self,
        config: ProviderConfig,
        request_pipeline: RequestPipeline,
        response_pipeline: ResponsePipeline,
        http_client: HttpClientService,
        sse_formatter: Optional[SseFormatter] = None,
    ):
        self.config = config
        self.request_pipeline = request_pipeline
        self.response_pipeline = response_pipeline
        self.http_client = http_client
        self.sse_formatter = sse_formatter

    @property
    def name(self) -> str:
        """Get provider name."""
        return self.config.name

    @property
    def api_url(self) -> str:
        """Get provider API URL."""
        return self.config.url

    @property
    def api_key(self) -> str:
        """Get provider API key."""
        return self.config.api_key


class ProviderFactory:
    """Factory for creating Provider instances from configuration."""

    def __init__(self, transformer_registry: TransformerRegistry, builtin_transformers: Optional[Dict[str, any]] = None):
        """Initialize provider factory.

        Args:
            transformer_registry: Registry containing custom transformers
            builtin_transformers: Dictionary of built-in transformers
        """
        self.transformer_registry = transformer_registry
        self.builtin_transformers = builtin_transformers or {}

    def create_provider(self, config: ProviderConfig, http_client: HttpClientService, sse_formatter: Optional[SseFormatter] = None) -> Provider:
        """Create a Provider instance from configuration.

        Args:
            config: Provider configuration
            http_client: HTTP client service to use
            sse_formatter: Optional SSE formatter

        Returns:
            Configured Provider instance

        Raises:
            ValueError: If required transformers are not found
        """
        # Build request pipeline
        request_transformers = self._build_transformers(config.request_pipeline, 'request')
        request_pipeline = RequestPipeline(request_transformers)

        # Build response pipeline
        response_transformers = self._build_transformers(config.response_pipeline, 'response')
        stream_transformers = self._build_transformers(config.response_pipeline, 'stream')
        response_pipeline = ResponsePipeline(response_transformers, stream_transformers)

        # Create provider
        provider = Provider(config=config, request_pipeline=request_pipeline, response_pipeline=response_pipeline, http_client=http_client, sse_formatter=sse_formatter)

        logger.info(f"Created provider '{config.name}' with {len(request_transformers)} request transformers, {len(response_transformers)} response transformers")
        return provider

    def _build_transformers(self, pipeline_config: List[PipelineTransformerConfig], transformer_type: str) -> List[any]:
        """Build transformers for a pipeline.

        Args:
            pipeline_config: Pipeline transformer configurations
            transformer_type: Type of transformers to build ('request', 'response', 'stream')

        Returns:
            List of transformer instances

        Raises:
            ValueError: If required transformer is not found
        """
        transformers = []

        for transformer_config in pipeline_config:
            transformer_name = transformer_config.name

            # Try to get transformer from registry
            transformer_factory = self.transformer_registry.get(transformer_name)

            if transformer_factory:
                # Create transformer using factory
                if transformer_type == 'request':
                    transformer = transformer_factory.create_request_transformer(transformer_config.args)
                elif transformer_type == 'response':
                    transformer = transformer_factory.create_response_transformer(transformer_config.args)
                elif transformer_type == 'stream':
                    transformer = transformer_factory.create_stream_transformer(transformer_config.args)
                else:
                    transformer = None

                if transformer:
                    transformers.append(transformer)
                    logger.debug(f'Added {transformer_type} transformer: {transformer_name}')
                else:
                    logger.warning(f"Transformer factory '{transformer_name}' did not provide a {transformer_type} transformer")

            elif transformer_name.startswith('builtin-'):
                # Handle built-in transformers
                builtin_name = transformer_name[8:]  # Remove 'builtin-' prefix
                builtin_transformer = self.builtin_transformers.get(builtin_name)

                if builtin_transformer:
                    transformers.append(builtin_transformer)
                    logger.debug(f'Added built-in {transformer_type} transformer: {builtin_name}')
                else:
                    logger.warning(f"Built-in transformer '{builtin_name}' not found")

            else:
                # Transformer not found - this could be an error or warning depending on requirements
                logger.error(f"Transformer '{transformer_name}' not found in registry or built-ins")
                # For now, we'll skip missing transformers rather than failing
                # This could be made configurable based on requirements

        return transformers


class ProviderRegistry(ComponentRegistry[Provider]):
    """Registry for managing API providers."""

    def __init__(self, provider_factory: ProviderFactory):
        super().__init__()
        self.provider_factory = provider_factory

    def register_provider_from_config(self, config: ProviderConfig, http_client: HttpClientService, sse_formatter: Optional[SseFormatter] = None) -> bool:
        """Register a provider from configuration.

        Args:
            config: Provider configuration
            http_client: HTTP client service
            sse_formatter: Optional SSE formatter

        Returns:
            True if registered successfully, False otherwise
        """
        try:
            provider = self.provider_factory.create_provider(config, http_client, sse_formatter)
            self.register(config.name, provider)
            logger.info(f"Registered provider '{config.name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to register provider '{config.name}': {e}", exc_info=True)
            return False

    def register_providers_from_config(self, providers: List[ProviderConfig], http_client: HttpClientService, sse_formatter: Optional[SseFormatter] = None) -> int:
        """Register multiple providers from configuration.

        Args:
            providers: List of provider configurations
            http_client: HTTP client service
            sse_formatter: Optional SSE formatter

        Returns:
            Number of providers registered successfully
        """
        success_count = 0

        for provider_config in providers:
            if self.register_provider_from_config(provider_config, http_client, sse_formatter):
                success_count += 1

        logger.info(f'Registered {success_count}/{len(providers)} providers successfully')
        return success_count

    def get_provider_by_name(self, name: str) -> Optional[Provider]:
        """Get a provider by name."""
        return self.get(name)

    def list_provider_names(self) -> List[str]:
        """List all registered provider names."""
        return self.list_names()

    def clear_all_providers(self) -> None:
        """Clear all registered providers."""
        self.clear()
        logger.info('Cleared all providers')
