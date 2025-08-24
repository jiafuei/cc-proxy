"""Core interfaces for simplified configuration management."""

from abc import ABC, abstractmethod
from typing import Callable, Optional

from app.config.user_models import UserConfig


class UserConfigManager(ABC):
    """Interface for managing user configuration with manual reloading."""

    @abstractmethod
    def load_config(self) -> UserConfig:
        """Load the current user configuration."""
        pass

    @abstractmethod
    def get_current_config(self) -> Optional[UserConfig]:
        """Get the currently loaded configuration."""
        pass

    @abstractmethod
    def reload_config(self) -> UserConfig:
        """Reload configuration from file."""
        pass

    @abstractmethod
    def on_config_change(self, callback: Callable[[UserConfig], None]) -> None:
        """Register a callback for configuration changes."""
        pass
