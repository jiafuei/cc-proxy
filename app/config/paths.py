"""Shared filesystem helpers for configuration storage."""

from pathlib import Path


def get_app_dir() -> Path:
    """Return the cc-proxy configuration directory under the user's home."""

    return Path.home() / '.cc-proxy'


__all__ = ['get_app_dir']
