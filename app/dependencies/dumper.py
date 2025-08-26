"""Dumper dependency injection for FastAPI."""

from app.common.dumper import Dumper
from app.config import get_config


def get_dumper() -> Dumper:
    """Get a configured dumper instance for dependency injection."""
    config = get_config()
    return Dumper(config)
