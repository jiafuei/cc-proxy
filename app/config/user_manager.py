"""User configuration loading and manual reload helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable, Optional, Union

from app.config.log import get_logger
from app.config.paths import get_app_dir
from app.config.user_models import UserConfig

logger = get_logger(__name__)


class UserConfigManager:
    """Interface for managing user configuration with manual reload support."""

    def load_config(self) -> UserConfig:
        raise NotImplementedError

    def get_current_config(self) -> Optional[UserConfig]:
        raise NotImplementedError

    async def reload_config(self) -> UserConfig:
        raise NotImplementedError

    async def trigger_reload(self) -> dict:
        raise NotImplementedError

    def on_config_change(
        self,
        callback: Union[Callable[[UserConfig], None], Callable[[UserConfig], Awaitable[None]]],
    ) -> None:
        raise NotImplementedError


class SimpleUserConfigManager(UserConfigManager):
    """Default implementation storing config in ~/.cc-proxy/user.yaml."""

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or (get_app_dir() / 'user.yaml')
        self._current_config: Optional[UserConfig] = None
        self._callback: Optional[Callable[[UserConfig], None]] = None
        logger.info('Initialized user config manager for: %s', self._config_path)

    def load_config(self) -> UserConfig:
        try:
            config = UserConfig.load(self._config_path)
            logger.info('Successfully loaded user configuration from file')
        except FileNotFoundError:
            logger.info('No user config file found at %s, using empty configuration', self._config_path)
            empty_config = UserConfig()
            self._current_config = empty_config
            return empty_config
        except PermissionError as exc:
            logger.error('Permission denied reading config file %s: %s', self._config_path, exc)
            if self._current_config is not None:
                logger.info('Keeping previous configuration due to permission error')
                return self._current_config
            empty_config = UserConfig()
            self._current_config = empty_config
            return empty_config
        except ValueError as exc:
            if 'Invalid YAML' in str(exc):
                logger.error('YAML syntax error in config file: %s', exc)
            else:
                logger.error('Configuration validation error: %s', exc)

            if self._current_config is not None:
                logger.warning('Keeping previous configuration due to validation error - please fix your config file')
                return self._current_config

            logger.error('No valid configuration available - system will run with empty config but may not function correctly')
            logger.error('Please check your config file syntax and structure using the /api/config/validate-yaml endpoint')
            empty_config = UserConfig()
            self._current_config = empty_config
            return empty_config

        try:
            config.validate_references()
        except ValueError as exc:
            logger.error('Configuration reference validation failed: %s', exc)
            if self._current_config is not None:
                logger.warning('Keeping previous configuration due to reference validation error')
                return self._current_config

            logger.error('Configuration loaded but has invalid references - some functionality may not work')
            logger.error('Please use the /api/config/validate endpoint to check your configuration')
            self._current_config = config
            return config

        self._current_config = config
        logger.info('Successfully loaded and validated user configuration')
        logger.debug(
            'Config: %s transformer paths, %s providers, %s models',
            len(config.transformer_paths),
            len(config.providers),
            len(config.models),
        )
        return config

    def get_current_config(self) -> Optional[UserConfig]:
        return self._current_config

    async def reload_config(self) -> UserConfig:
        logger.info('Manually reloading user configuration')

        old_config = self._current_config
        new_config = self.load_config()

        if self._config_changed(old_config, new_config):
            logger.info('Configuration changed, notifying callback')
            await self._notify_callback(new_config)
        else:
            logger.debug('Configuration unchanged after reload')

        return new_config

    def on_config_change(
        self,
        callback: Union[Callable[[UserConfig], None], Callable[[UserConfig], Awaitable[None]]],
    ) -> None:
        self._callback = callback
        logger.debug('Registered config change callback: %s', getattr(callback, '__name__', 'anonymous'))

    async def trigger_reload(self) -> dict:
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
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error('Failed to reload configuration: %s', exc, exc_info=True)
            return {'success': False, 'message': f'Failed to reload configuration: {exc}', 'error': str(exc)}

    def get_config_status(self) -> dict:
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
            'provider_names': [provider.name for provider in config.providers],
        }

    async def _notify_callback(self, config: UserConfig) -> None:
        if not self._callback:
            return
        try:
            if asyncio.iscoroutinefunction(self._callback):
                await self._callback(config)
            else:
                self._callback(config)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error('Error in config change callback %s: %s', self._callback, exc, exc_info=True)

    def _config_changed(self, old_config: Optional[UserConfig], new_config: UserConfig) -> bool:
        if old_config is None:
            return True
        try:
            return old_config.model_dump() != new_config.model_dump()
        except Exception:
            return True


_global_config_manager: Optional[SimpleUserConfigManager] = None


def get_user_config_manager() -> SimpleUserConfigManager:
    global _global_config_manager

    if _global_config_manager is None:
        _global_config_manager = SimpleUserConfigManager()

    return _global_config_manager


def set_user_config_manager(manager: SimpleUserConfigManager) -> None:
    global _global_config_manager
    _global_config_manager = manager


__all__ = ['SimpleUserConfigManager', 'UserConfigManager', 'get_user_config_manager', 'set_user_config_manager']
