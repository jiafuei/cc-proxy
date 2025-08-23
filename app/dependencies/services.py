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
from app.services.lifecycle.simple_service_provider import SimpleServiceProvider
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


# Global simple service provider
_simple_service_provider: Optional[SimpleServiceProvider] = None
_fallback_services: Optional[Services] = None


def get_dynamic_service_provider() -> SimpleServiceProvider:
    """Get the global simple service provider."""
    global _simple_service_provider

    if _simple_service_provider is None:
        logger.info('Initializing simple service provider')

        # Get configuration
        app_config = get_config()
        service_builder = DynamicServiceBuilder(app_config)

        # Create service provider
        _simple_service_provider = SimpleServiceProvider(app_config, service_builder)

        # Load initial user configuration and build services
        config_manager = get_user_config_manager()
        try:
            user_config = config_manager.load_config()
            _simple_service_provider.rebuild_services(user_config)

            # Register callback for manual config changes
            config_manager.on_config_change(_on_user_config_change)

            logger.info('Simple service provider initialized successfully')

        except Exception as e:
            logger.error(f'Failed to initialize user configuration: {e}', exc_info=True)
            # Service provider will fall back to empty config

    return _simple_service_provider


def _on_user_config_change(user_config):
    """Callback for user configuration changes."""
    global _simple_service_provider

    if _simple_service_provider:
        try:
            logger.info('User configuration changed, rebuilding services')
            _simple_service_provider.rebuild_services(user_config)
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

    This function supports both the new simple service system and
    falls back to the legacy static services for backward compatibility.

    Args:
        request: Optional FastAPI request object (no longer used)

    Returns:
        Services instance (either dynamic or static)
    """
    # Try to get current services from simple provider
    try:
        service_provider = get_dynamic_service_provider()
        _, services = service_provider.get_current_services()
        logger.debug('Using simple services')
        return services

    except Exception as e:
        logger.warning(f'Failed to get simple services, falling back to static: {e}')
        return get_fallback_services()
