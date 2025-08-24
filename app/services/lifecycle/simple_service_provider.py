"""Simple service provider without generation complexity."""

import logging
from typing import Optional

from app.config.log import get_logger
from app.config.models import ConfigModel
from app.config.user_models import UserConfig
from app.services.config.interfaces import ServiceBuilder, ServiceProvider

logger = get_logger(__name__)


class SimpleServiceProvider(ServiceProvider):
    """Simple service provider that rebuilds services on configuration changes."""

    def __init__(self, app_config: ConfigModel, service_builder: ServiceBuilder):
        """Initialize the service provider.

        Args:
            app_config: Static application configuration
            service_builder: Builder for creating services from configuration
        """
        self.app_config = app_config
        self.service_builder = service_builder
        self._current_services: Optional[any] = None

        logger.info('Initialized simple service provider')

    def get_services(self) -> any:
        """Get current services.

        Returns:
            Current services instance
        """
        if self._current_services is None:
            raise RuntimeError('No services available - call rebuild_services first')

        return self._current_services

    def rebuild_services(self, config: UserConfig) -> None:
        """Rebuild services from new configuration.

        Args:
            config: New user configuration
        """
        logger.info('Rebuilding services from new configuration')

        try:
            # Build new services
            new_services = self.service_builder.build_services(config)

            # Replace current services
            self._current_services = new_services

            logger.info('Services rebuilt successfully')

        except Exception as e:
            logger.error(f'Failed to rebuild services: {e}', exc_info=True)
            raise
