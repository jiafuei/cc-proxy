"""Simple user configuration manager with manual reload support."""

import asyncio
from pathlib import Path
from typing import Awaitable, Callable, Optional, Union

from app.common.utils import get_app_dir
from app.config.log import get_logger
from app.config.user_models import UserConfig

from .interfaces import UserConfigManager

logger = get_logger(__name__)


class SimpleUserConfigManager(UserConfigManager):
    """Implementation of UserConfigManager with manual reload support."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the config manager.

        Args:
            config_path: Optional path to config file (defaults to ~/.cc-proxy/user.yaml)
        """
        self._config_path = config_path or (get_app_dir() / 'user.yaml')
        self._current_config: Optional[UserConfig] = None
        self._callback: Optional[Callable[[UserConfig], None]] = None

        logger.info(f'Initialized user config manager for: {self._config_path}')

    def load_config(self) -> UserConfig:
        """Load the current user configuration."""
        try:
            config = UserConfig.load(self._config_path)
            logger.info('Successfully loaded user configuration from file')

        except FileNotFoundError:
            # Config file doesn't exist - this is expected for fresh installs
            logger.info(f'No user config file found at {self._config_path}, using empty configuration')
            empty_config = UserConfig()
            self._current_config = empty_config
            return empty_config

        except PermissionError as e:
            # File permission issues - fallback but log error
            logger.error(f'Permission denied reading config file {self._config_path}: {e}')
            if self._current_config is not None:
                logger.info('Keeping previous configuration due to permission error')
                return self._current_config
            logger.warning('Using empty configuration due to permission error')
            empty_config = UserConfig()
            self._current_config = empty_config
            return empty_config

        except ValueError as e:
            # This covers YAML parsing errors and pydantic validation errors from UserConfig.load()
            if 'Invalid YAML' in str(e):
                logger.error(f'YAML syntax error in config file: {e}')
            else:
                logger.error(f'Configuration validation error: {e}')

            # For syntax/validation errors, we should not silently fall back
            # These are user configuration mistakes that need to be fixed
            if self._current_config is not None:
                logger.warning('Keeping previous configuration due to validation error - please fix your config file')
                return self._current_config

            # If no previous config and validation fails, this is a critical error
            # The user needs to know their config is broken
            logger.error('No valid configuration available - system will run with empty config but may not function correctly')
            logger.error('Please check your config file syntax and structure using the /api/config/validate-yaml endpoint')
            empty_config = UserConfig()
            self._current_config = empty_config
            return empty_config

        # Validate configuration references
        try:
            config.validate_references()

        except ValueError as e:
            # Reference validation errors are serious - these indicate logical errors in config
            logger.error(f'Configuration reference validation failed: {e}')
            if self._current_config is not None:
                logger.warning('Keeping previous configuration due to reference validation error')
                return self._current_config

            # If no previous config, log detailed error but continue with loaded config
            # The config structure is valid but references are wrong
            logger.error('Configuration loaded but has invalid references - some functionality may not work')
            logger.error('Please use the /api/config/validate endpoint to check your configuration')
            # Still save the config so user can see what was loaded
            self._current_config = config
            return config

        # Configuration loaded and validated successfully
        self._current_config = config
        logger.info('Successfully loaded and validated user configuration')
        logger.debug(f'Config: {len(config.transformer_paths)} transformer paths, {len(config.providers)} providers, {len(config.models)} models')
        return config

    def get_current_config(self) -> Optional[UserConfig]:
        """Get the currently loaded configuration."""
        return self._current_config

    async def reload_config(self) -> UserConfig:
        """Reload configuration from file."""
        logger.info('Manually reloading user configuration')

        old_config = self._current_config
        new_config = self.load_config()

        # Only notify callback if config actually changed
        if self._config_changed(old_config, new_config):
            logger.info('Configuration changed, notifying callback')
            await self._notify_callback(new_config)
        else:
            logger.debug('Configuration unchanged after reload')

        return new_config

    def on_config_change(self, callback: Union[Callable[[UserConfig], None], Callable[[UserConfig], Awaitable[None]]]) -> None:
        """Register a callback for configuration changes."""
        self._callback = callback
        logger.debug(f'Registered config change callback: {callback.__name__ if hasattr(callback, "__name__") else "anonymous"}')

    async def trigger_reload(self) -> dict:
        """Manually trigger configuration reload via API.

        Returns:
            Dictionary with reload results
        """
        try:
            old_config = self.get_current_config()
            new_config = await self.reload_config()

            return {
                'success': True,
                'message': 'Configuration reloaded successfully',
                'changes': {
                    'transformer_paths': len(new_config.transformer_paths),
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
            'transformer_paths': len(config.transformer_paths),
            'providers': len(config.providers),
            'models': len(config.models),
            'routing_configured': config.routing is not None,
            'transformer_paths_list': config.transformer_paths,
            'provider_names': [p.name for p in config.providers],
        }

    async def _notify_callback(self, config: UserConfig) -> None:
        """Notify the registered callback of config change."""
        if self._callback:
            try:
                if asyncio.iscoroutinefunction(self._callback):
                    await self._callback(config)
                else:
                    self._callback(config)
            except Exception as e:
                logger.error(f'Error in config change callback {self._callback}: {e}', exc_info=True)

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


def get_user_config_manager() -> SimpleUserConfigManager:
    """Get the global user configuration manager instance."""
    global _global_config_manager

    if _global_config_manager is None:
        _global_config_manager = SimpleUserConfigManager()

    return _global_config_manager


def set_user_config_manager(manager: SimpleUserConfigManager) -> None:
    """Set a custom user configuration manager (primarily for testing)."""
    global _global_config_manager

    _global_config_manager = manager
