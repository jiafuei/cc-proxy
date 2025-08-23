"""Simple user configuration manager with manual reload support."""

import logging
import threading
from pathlib import Path
from typing import Callable, List, Optional

from app.common.utils import get_app_dir
from app.config.user_models import UserConfig

from .interfaces import UserConfigManager

logger = logging.getLogger(__name__)


class SimpleUserConfigManager(UserConfigManager):
    """Implementation of UserConfigManager with manual reload support."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the config manager.

        Args:
            config_path: Optional path to config file (defaults to ~/.cc-proxy/user.yaml)
        """
        self._config_path = config_path or (get_app_dir() / 'user.yaml')
        self._current_config: Optional[UserConfig] = None
        self._callbacks: List[Callable[[UserConfig], None]] = []
        self._lock = threading.RLock()

        logger.info(f'Initialized user config manager for: {self._config_path}')

    def load_config(self) -> UserConfig:
        """Load the current user configuration."""
        with self._lock:
            try:
                config = UserConfig.load(self._config_path)

                # Validate configuration references
                config.validate_references()

                self._current_config = config
                logger.info('Successfully loaded user configuration')
                logger.debug(f'Config: {len(config.transformers)} transformers, {len(config.providers)} providers, {len(config.models)} models')

                return config

            except Exception as e:
                logger.error(f'Failed to load user configuration: {e}')
                # If we have a previous config, keep using it
                if self._current_config is not None:
                    logger.info('Keeping previous configuration due to load error')
                    return self._current_config
                # Otherwise return empty config
                logger.info('Using empty configuration due to load error')
                empty_config = UserConfig()
                self._current_config = empty_config
                return empty_config

    def get_current_config(self) -> Optional[UserConfig]:
        """Get the currently loaded configuration."""
        with self._lock:
            return self._current_config

    def reload_config(self) -> UserConfig:
        """Reload configuration from file."""
        logger.info('Manually reloading user configuration')

        with self._lock:
            old_config = self._current_config
            new_config = self.load_config()

            # Only notify callbacks if config actually changed
            if self._config_changed(old_config, new_config):
                logger.info('Configuration changed, notifying callbacks')
                self._notify_callbacks(new_config)
            else:
                logger.debug('Configuration unchanged after reload')

            return new_config

    def on_config_change(self, callback: Callable[[UserConfig], None]) -> None:
        """Register a callback for configuration changes."""
        with self._lock:
            self._callbacks.append(callback)
            logger.debug(f'Registered config change callback: {callback.__name__ if hasattr(callback, "__name__") else "anonymous"}')

    def trigger_reload(self) -> dict:
        """Manually trigger configuration reload via API.

        Returns:
            Dictionary with reload results
        """
        try:
            old_config = self.get_current_config()
            new_config = self.reload_config()

            return {
                'success': True,
                'message': 'Configuration reloaded successfully',
                'changes': {
                    'transformers': len(new_config.transformers),
                    'providers': len(new_config.providers),
                    'models': len(new_config.models),
                    'routing_configured': new_config.routing is not None,
                },
                'config_changed': self._config_changed(old_config, new_config),
            }

        except Exception as e:
            logger.error(f'Failed to reload configuration: {e}', exc_info=True)
            return {'success': False, 'message': f'Failed to reload configuration: {str(e)}', 'error': str(e)}

    def get_config_status(self) -> dict:
        """Get current configuration status.

        Returns:
            Dictionary with configuration status
        """
        config = self.get_current_config()

        if config is None:
            return {'loaded': False, 'config_file_exists': self._config_path.exists(), 'config_path': str(self._config_path)}

        return {
            'loaded': True,
            'config_file_exists': self._config_path.exists(),
            'config_path': str(self._config_path),
            'transformers': len(config.transformers),
            'providers': len(config.providers),
            'models': len(config.models),
            'routing_configured': config.routing is not None,
            'transformer_names': [t.name for t in config.transformers],
            'provider_names': [p.name for p in config.providers],
            'model_ids': [m.id for m in config.models],
        }

    def _notify_callbacks(self, config: UserConfig) -> None:
        """Notify all registered callbacks of config change."""
        for callback in self._callbacks:
            try:
                callback(config)
            except Exception as e:
                logger.error(f'Error in config change callback {callback}: {e}', exc_info=True)

    def _config_changed(self, old_config: Optional[UserConfig], new_config: UserConfig) -> bool:
        """Check if configuration actually changed."""
        if old_config is None:
            return True

        # Simple comparison - could be more sophisticated
        try:
            return old_config.model_dump() != new_config.model_dump()
        except Exception:
            # If comparison fails, assume it changed
            return True


# Global instance for easy access
_global_config_manager: Optional[SimpleUserConfigManager] = None
_global_lock = threading.Lock()


def get_user_config_manager() -> SimpleUserConfigManager:
    """Get the global user configuration manager instance."""
    global _global_config_manager

    with _global_lock:
        if _global_config_manager is None:
            _global_config_manager = SimpleUserConfigManager()

        return _global_config_manager


def set_user_config_manager(manager: SimpleUserConfigManager) -> None:
    """Set a custom user configuration manager (primarily for testing)."""
    global _global_config_manager

    with _global_lock:
        _global_config_manager = manager
