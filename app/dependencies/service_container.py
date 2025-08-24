"""Service container that brings together the new architecture components."""

from typing import Dict, Optional

from app.common.dumper import Dumper
from app.config import get_config
from app.config.log import get_logger
from app.config.user_models import UserConfig
from app.services.config.simple_user_config_manager import get_user_config_manager
from app.services.provider import ProviderManager
from app.services.router import SimpleRouter
from app.services.transformer_loader import TransformerLoader

logger = get_logger(__name__)


class ServiceContainer:
    """Service container that manages providers, routing, and basic services."""

    def __init__(self):
        self.app_config = get_config()
        self.dumper = Dumper(self.app_config)

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

            # Extract config sections
            providers_config = self._extract_providers_config(user_config)
            routing_config = self._extract_routing_config(user_config)
            transformer_paths = self._extract_transformer_paths(user_config)

            # Initialize transformer loader
            self.transformer_loader = TransformerLoader(transformer_paths)

            # Initialize provider manager
            self.provider_manager = ProviderManager(providers_config, self.transformer_loader)

            # Initialize router
            self.router = SimpleRouter(self.provider_manager, routing_config)

            logger.info(f'Service container initialized: {len(self.provider_manager.list_providers())} providers, {len(self.provider_manager.list_models())} models')

        except Exception as e:
            logger.error(f'Failed to initialize service container: {e}', exc_info=True)
            # Initialize with empty configs as fallback
            self.transformer_loader = TransformerLoader([])
            self.provider_manager = ProviderManager({}, self.transformer_loader)
            self.router = SimpleRouter(self.provider_manager, {'default': ''})

    def _extract_providers_config(self, user_config: UserConfig) -> Dict:
        """Extract provider configuration from user config."""
        if hasattr(user_config, 'providers') and user_config.providers:
            # Convert provider objects to dict format
            providers_dict = {}
            for provider in user_config.providers:
                if hasattr(provider, 'name'):
                    providers_dict[provider.name] = {
                        'url': getattr(provider, 'url', ''),
                        'api_key': getattr(provider, 'api_key', ''),
                        'models': getattr(provider, 'models', []),
                        'transformers': getattr(provider, 'transformers', {}),
                        'timeout': getattr(provider, 'timeout', 300),
                    }
            return providers_dict
        return {}

    def _extract_routing_config(self, user_config: UserConfig) -> Dict:
        """Extract routing configuration from user config."""
        if hasattr(user_config, 'routing') and user_config.routing:
            return {
                'default': getattr(user_config.routing, 'default', ''),
                'planning': getattr(user_config.routing, 'planning', ''),
                'background': getattr(user_config.routing, 'background', ''),
            }
        return {'default': '', 'planning': '', 'background': ''}

    def _extract_transformer_paths(self, user_config: UserConfig) -> list:
        """Extract transformer paths from user config."""
        if hasattr(user_config, 'transformer_paths'):
            return getattr(user_config, 'transformer_paths', [])
        return []

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

    async def close(self):
        """Clean up resources."""
        if self.provider_manager:
            await self.provider_manager.close_all()


# Global service container instance
_service_container: Optional[ServiceContainer] = None


def get_service_container() -> ServiceContainer:
    """Get the global service container instance."""
    global _service_container

    if _service_container is None:
        _service_container = ServiceContainer()

    return _service_container