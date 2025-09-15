"""Service container that brings together the new architecture components."""

from typing import Dict, Optional

from app.config import ConfigurationService
from app.config.log import get_logger
from app.config.user_models import RoutingConfig, UserConfig
from app.services.config.simple_user_config_manager import get_user_config_manager
from app.services.provider import ProviderManager
from app.services.router import SimpleRouter
from app.services.transformer_loader import TransformerLoader

logger = get_logger(__name__)


class ServiceContainer:
    """Service container that manages providers, routing, and basic services."""

    def __init__(self, config_service: Optional[ConfigurationService] = None):
        self.config_service = config_service or ConfigurationService()
        self.app_config = self.config_service.get_config()

        # Core components
        self.transformer_loader: Optional[TransformerLoader] = None
        self.provider_manager: Optional[ProviderManager] = None
        self.router: Optional[SimpleRouter] = None

        # Initialize from user config
        self._initialize()

    def _initialize(self):
        """Initialize the service container from user configuration."""
        try:
            # Load user config
            config_manager = get_user_config_manager()
            user_config = config_manager.load_config()

            # Register callback for config changes
            config_manager.on_config_change(self.reinitialize_from_config)

            # Initialize transformer loader
            self.transformer_loader = TransformerLoader(user_config.transformer_paths)

            # Initialize provider manager
            self.provider_manager = ProviderManager(user_config.providers, user_config.models, self.transformer_loader)

            # Initialize router
            routing_config = user_config.routing if user_config.routing else self._get_default_routing()
            self.router = SimpleRouter(self.provider_manager, routing_config, self.transformer_loader)

            logger.info(f'Service container initialized: {len(self.provider_manager.list_providers())} providers, {len(self.provider_manager.list_models())} models')

        except Exception as e:
            logger.error(f'Failed to initialize service container: {e}', exc_info=True)
            # Initialize with empty configs as fallback
            self.transformer_loader = TransformerLoader([])
            self.provider_manager = ProviderManager([], [], self.transformer_loader)
            self.router = SimpleRouter(self.provider_manager, self._get_default_routing(), self.transformer_loader)

    def _get_default_routing(self) -> RoutingConfig:
        """Get default routing configuration."""
        return RoutingConfig(default='', planning='', background='')

    def get_system_info(self) -> Dict:
        """Get information about the current system state."""
        if not self.provider_manager or not self.router:
            return {'status': 'not_initialized'}

        return {
            'status': 'initialized',
            'providers': self.provider_manager.list_providers(),
            'models': self.provider_manager.list_models(),
            'routing': self.router.get_routing_info(),
            'transformer_cache': self.transformer_loader.get_cache_info() if self.transformer_loader else {},
        }

    async def reinitialize_from_config(self, new_config: UserConfig):
        """Reinitialize the service container with new configuration."""
        try:
            logger.info('Reinitializing service container with new config')

            # Clean up existing resources
            if self.provider_manager:
                try:
                    # Wait for async cleanup to complete
                    await self.provider_manager.close_all()
                except Exception as e:
                    logger.warning(f'Error during provider cleanup: {e}', exc_info=True)

            if self.router:
                try:
                    await self.router.close()
                except Exception as e:
                    logger.warning(f'Error during router cleanup: {e}', exc_info=True)

            # Reinitialize with new config
            self.transformer_loader = TransformerLoader(new_config.transformer_paths)
            self.provider_manager = ProviderManager(new_config.providers, new_config.models, self.transformer_loader)

            routing_config = new_config.routing if new_config.routing else self._get_default_routing()
            self.router = SimpleRouter(self.provider_manager, routing_config, self.transformer_loader)

            logger.info(f'Service container reinitialized: {len(self.provider_manager.list_providers())} providers, {len(self.provider_manager.list_models())} models')

        except Exception as e:
            logger.error(f'Failed to reinitialize service container: {e}', exc_info=True)

    async def close(self):
        """Clean up resources."""
        if self.provider_manager:
            await self.provider_manager.close_all()
        if self.router:
            await self.router.close()


