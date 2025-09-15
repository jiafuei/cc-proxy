"""Common dependency injection functions for FastAPI."""

from fastapi import Request

from app.config import ConfigurationService
from app.dependencies.service_container import ServiceContainer


def get_config_service_dependency(request: Request) -> ConfigurationService:
    """Get configuration service from app state for dependency injection."""
    return request.app.state.config_service


def get_service_container_dependency(request: Request) -> ServiceContainer:
    """Get service container from app state for dependency injection."""
    print(request.base_url)
    return request.app.state.service_container