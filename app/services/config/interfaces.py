"""Core interfaces for dynamic configuration management."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, Optional, TypeVar

from app.config.user_models import UserConfig

T = TypeVar('T')


class ComponentRegistry(Generic[T]):
    """Generic registry for managing components."""

    def __init__(self):
        self._components: Dict[str, T] = {}

    def register(self, name: str, component: T) -> None:
        """Register a component by name."""
        self._components[name] = component

    def unregister(self, name: str) -> None:
        """Unregister a component by name."""
        self._components.pop(name, None)

    def get(self, name: str) -> Optional[T]:
        """Get a component by name."""
        return self._components.get(name)

    def list_names(self) -> list[str]:
        """List all registered component names."""
        return list(self._components.keys())

    def clear(self) -> None:
        """Clear all registered components."""
        self._components.clear()

    def size(self) -> int:
        """Get number of registered components."""
        return len(self._components)


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
