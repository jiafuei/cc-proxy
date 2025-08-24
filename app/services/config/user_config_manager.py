"""Central user configuration manager with dynamic reloading."""

import logging
import threading
from pathlib import Path
from typing import Callable, List, Optional

from app.common.utils import get_app_dir
from app.config.log import get_logger
from app.config.user_models import UserConfig

from .interfaces import ConfigWatcher, UserConfigManager
from .watcher import WatchdogConfigWatcher

logger = get_logger(__name__)


class DynamicUserConfigManager(UserConfigManager):
    """Implementation of UserConfigManager with dynamic reloading support."""

    def __init__(self, config_path: Optional[Path] = None, watcher: Optional[ConfigWatcher] = None):
        """Initialize the config manager.

        Args:
            config_path: Optional path to config file (defaults to ~/.cc-proxy/user.yaml)
            watcher: Optional custom config watcher (defaults to WatchdogConfigWatcher)
        """
        self._config_path = config_path or (get_app_dir() / 'user.yaml')
        self._watcher = watcher or WatchdogConfigWatcher()
        self._current_config: Optional[UserConfig] = None
        self._callbacks: List[Callable[[UserConfig], None]] = []
        self._lock = threading.RLock()
        self._watching = False

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
        logger.info('Reloading user configuration')

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

    def start_watching(self) -> None:
        """Start watching for configuration file changes."""
        with self._lock:
            if self._watching:
                logger.warning('Already watching for config changes')
                return

            # Load initial configuration
            self.load_config()

            # Start watching for changes
            self._watcher.watch(self._config_path, self._on_file_change)
            self._watching = True

            logger.info('Started watching for configuration changes')

    def stop_watching(self) -> None:
        """Stop watching for configuration file changes."""
        with self._lock:
            if not self._watching:
                return

            self._watcher.stop_watching()
            self._watching = False

            logger.info('Stopped watching for configuration changes')

    def is_watching(self) -> bool:
        """Check if currently watching for changes."""
        with self._lock:
            return self._watching

    def _on_file_change(self, path: Path) -> None:
        """Internal callback for file system changes."""
        logger.info(f'Configuration file changed: {path}')
        try:
            # Small delay to ensure file write is complete
            import time

            time.sleep(0.1)

            self.reload_config()
        except Exception as e:
            logger.error(f'Error handling config file change: {e}', exc_info=True)

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

    def __del__(self):
        """Ensure watcher is stopped on cleanup."""
        try:
            self.stop_watching()
        except Exception as e:
            logger.error(f'Error stopping config manager during cleanup: {e}')


# Global instance for easy access
_global_config_manager: Optional[DynamicUserConfigManager] = None
_global_lock = threading.Lock()


def get_user_config_manager() -> DynamicUserConfigManager:
    """Get the global user configuration manager instance."""
    global _global_config_manager

    with _global_lock:
        if _global_config_manager is None:
            _global_config_manager = DynamicUserConfigManager()

        return _global_config_manager


def set_user_config_manager(manager: DynamicUserConfigManager) -> None:
    """Set a custom user configuration manager (primarily for testing)."""
    global _global_config_manager

    with _global_lock:
        # Stop the old manager if it exists
        if _global_config_manager is not None:
            _global_config_manager.stop_watching()

        _global_config_manager = manager
