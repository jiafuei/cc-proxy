"""Dumper dependency injection for FastAPI."""

from typing import Annotated

from fastapi import Depends

from app.config import ConfigurationService
from app.dependencies import get_config_service_dependency
from app.observability.dumper import Dumper


def get_dumper(config_service: Annotated[ConfigurationService, Depends(get_config_service_dependency)]) -> Dumper:
    """Get a configured dumper instance for dependency injection."""
    config = config_service.get_config()
    return Dumper(config)
