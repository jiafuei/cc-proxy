"""Simplified services dependency with extracted pipeline."""

import logging
from typing import Optional

from app.common.dumper import Dumper
from app.config import get_config
from app.dependencies.pipeline import get_message_pipeline
from app.services.config.simple_user_config_manager import get_user_config_manager
from app.services.error_handling.error_formatter import ApiErrorFormatter
from app.services.error_handling.exception_mapper import HttpExceptionMapper
from app.services.lifecycle.service_builder import DynamicServiceBuilder
from app.services.lifecycle.simple_service_provider import SimpleServiceProvider
from app.services.pipeline.messages_service import MessagesPipelineService

logger = logging.getLogger(__name__)


class CoreServices:
    """Core services that don't depend on user configuration."""

    def __init__(self):
        self.config = get_config()
        self.dumper = Dumper(self.config)
        self.exception_mapper = HttpExceptionMapper()
        self.error_formatter = ApiErrorFormatter()


# Global service provider
_service_provider: Optional[SimpleServiceProvider] = None
_core_services: Optional[CoreServices] = None


def get_service_provider() -> SimpleServiceProvider:
    """Get the global service provider."""
    global _service_provider

    if _service_provider is None:
        logger.info('Initializing service provider')

        # Get configuration
        app_config = get_config()
        service_builder = DynamicServiceBuilder(app_config)

        # Create service provider
        _service_provider = SimpleServiceProvider(app_config, service_builder)

        # Load initial user configuration and build services
        config_manager = get_user_config_manager()
        try:
            user_config = config_manager.load_config()
            _service_provider.rebuild_services(user_config)

            # Register callback for config changes
            config_manager.on_config_change(_on_user_config_change)

            logger.info('Service provider initialized successfully')

        except Exception as e:
            logger.error(f'Failed to initialize user configuration: {e}', exc_info=True)
            # Service provider will fall back to empty config

    return _service_provider


def get_core_services() -> CoreServices:
    """Get core services that don't require user configuration."""
    global _core_services

    if _core_services is None:
        _core_services = CoreServices()

    return _core_services


def get_message_pipeline_service() -> MessagesPipelineService:
    """Get the default message pipeline service."""
    return get_message_pipeline()


def _on_user_config_change(user_config):
    """Callback for user configuration changes."""
    global _service_provider

    if _service_provider:
        try:
            logger.info('User configuration changed, rebuilding services')
            _service_provider.rebuild_services(user_config)
            logger.info('Services rebuilt successfully')
        except Exception as e:
            logger.error(f'Failed to rebuild services after config change: {e}', exc_info=True)


# Compatibility functions for existing code
def get_services():
    """Get services - returns service provider for compatibility."""
    try:
        return get_service_provider().get_services()
    except Exception as e:
        logger.warning(f'Failed to get dynamic services: {e}')
        return get_core_services()


def get_dynamic_service_provider() -> SimpleServiceProvider:
    """Get the dynamic service provider (alias for compatibility)."""
    return get_service_provider()
