"""Core interfaces for simplified configuration management."""

from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional, Union

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
    async def reload_config(self) -> UserConfig:
        """Reload configuration from file."""
        pass
    
    @abstractmethod
    async def trigger_reload(self) -> dict:
        """Manually trigger configuration reload via API."""
        pass

    @abstractmethod
    def on_config_change(self, callback: Union[Callable[[UserConfig], None], Callable[[UserConfig], Awaitable[None]]]) -> None:
        """Register a callback for configuration changes."""
        pass
