import logging
from typing import Optional

import httpx
from fastapi import Request

from app.common.dumper import Dumper
from app.config import get_config
from app.services.config.simple_user_config_manager import get_user_config_manager
from app.services.error_handling.error_formatter import ApiErrorFormatter
from app.services.error_handling.exception_mapper import HttpExceptionMapper
from app.services.lifecycle.service_builder import DynamicServiceBuilder
from app.services.lifecycle.service_provider import DynamicServiceProvider
from app.services.pipeline.http_client import HttpClientService
from app.services.pipeline.messages_service import MessagesPipelineService
from app.services.pipeline.request_pipeline import RequestPipeline
from app.services.pipeline.response_pipeline import ResponsePipeline
from app.services.sse_formatter.anthropic_formatter import AnthropicSseFormatter
from app.services.transformers.anthropic.transformers import AnthropicRequestTransformer, AnthropicResponseTransformer, AnthropicStreamTransformer

logger = logging.getLogger(__name__)


class Services:
    """Legacy Services class for backward compatibility.

    This class maintains the same interface as the original Services class
    but delegates to dynamic services when available.
    """

    def __init__(self):
        self.config = get_config()
        self.httpx_client = httpx.AsyncClient(timeout=60 * 5, http2=True)
        self.create_services()

    def create_services(self):
        # Initialize transformers
        anthropic_request_transformer = AnthropicRequestTransformer(self.config)
        anthropic_response_transformer = AnthropicResponseTransformer()
        anthropic_stream_transformer = AnthropicStreamTransformer()

        # Initialize pipelines
        request_pipeline = RequestPipeline([anthropic_request_transformer])
        response_pipeline = ResponsePipeline([anthropic_response_transformer], [anthropic_stream_transformer])

        # Initialize HTTP client
        http_client = HttpClientService(self.httpx_client)

        # Initialize SSE formatter
        sse_formatter = AnthropicSseFormatter()

        # Initialize pipeline service with SSE formatter
        self.messages_pipeline = MessagesPipelineService(request_pipeline, response_pipeline, http_client, sse_formatter)

        # Initialize error handling services
        self.exception_mapper = HttpExceptionMapper()
        self.error_formatter = ApiErrorFormatter()

        # Other services
        self.dumper = Dumper(self.config)


# Global dynamic service provider
_dynamic_service_provider: Optional[DynamicServiceProvider] = None
_fallback_services: Optional[Services] = None


def get_dynamic_service_provider() -> DynamicServiceProvider:
    """Get the global dynamic service provider."""
    global _dynamic_service_provider

    if _dynamic_service_provider is None:
        logger.info('Initializing dynamic service provider')

        # Get configuration
        app_config = get_config()
        service_builder = DynamicServiceBuilder(app_config)

        # Create service provider
        _dynamic_service_provider = DynamicServiceProvider(app_config, service_builder)

        # Load initial user configuration and build services
        config_manager = get_user_config_manager()
        try:
            user_config = config_manager.load_config()
            _dynamic_service_provider.rebuild_services(user_config)

            # Register callback for manual config changes
            config_manager.on_config_change(_on_user_config_change)

            logger.info('Dynamic service provider initialized successfully')

        except Exception as e:
            logger.error(f'Failed to initialize user configuration: {e}', exc_info=True)
            # Service provider will fall back to empty config

    return _dynamic_service_provider


def _on_user_config_change(user_config):
    """Callback for user configuration changes."""
    global _dynamic_service_provider

    if _dynamic_service_provider:
        try:
            logger.info('User configuration changed, rebuilding services')
            _dynamic_service_provider.rebuild_services(user_config)
            logger.info('Services rebuilt successfully')
        except Exception as e:
            logger.error(f'Failed to rebuild services after config change: {e}', exc_info=True)


def get_fallback_services() -> Services:
    """Get fallback services for when dynamic services are not available."""
    global _fallback_services

    if _fallback_services is None:
        _fallback_services = Services()

    return _fallback_services


def get_services(request: Optional[Request] = None):
    """Get services for the current request.

    This function supports both the new dynamic service system and
    falls back to the legacy static services for backward compatibility.

    Args:
        request: Optional FastAPI request object

    Returns:
        Services instance (either dynamic or static)
    """
    # Try to get services from request state (set by middleware)
    if request and hasattr(request.state, 'services') and request.state.services:
        logger.debug('Using services from request state')
        return request.state.services

    # Try to get current services from dynamic provider
    try:
        service_provider = get_dynamic_service_provider()
        generation_id, services = service_provider.get_current_services()
        logger.debug(f'Using dynamic services (generation: {generation_id})')
        return services

    except Exception as e:
        logger.warning(f'Failed to get dynamic services, falling back to static: {e}')
        return get_fallback_services()
