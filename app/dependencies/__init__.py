"""Common dependency injection functions for FastAPI."""

from fastapi import Request

from app.config import ConfigurationService
from app.dependencies.container import ServiceContainer, build_service_container


def get_config_service_dependency(request: Request) -> ConfigurationService:
    """Get configuration service from app state for dependency injection."""
    return request.app.state.config_service


def get_service_container_dependency(request: Request) -> ServiceContainer:
    """Get service container from app state for dependency injection."""
    return ensure_service_container(request)


def ensure_service_container(request: Request) -> ServiceContainer:
    """Ensure the service container exists on the application state."""
    container = getattr(request.app.state, 'service_container', None)
    if container is None:
        container = build_service_container(request.app.state.config_service)
        request.app.state.service_container = container
    return container
