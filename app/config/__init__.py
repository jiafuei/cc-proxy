from dataclasses import dataclass
from typing import List
from functools import lru_cache

import yaml


@dataclass
class Config:
    """Configuration class that loads settings from YAML file."""

    cors_allow_origins: List[str]
    host: str = '127.0.0.1'
    port: int = 8000
    dev: bool = False

    @classmethod
    def load(cls, config_path: str = 'config.yaml') -> 'Config':
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            # Return default config if file doesn't exist
            data = {}

        # Use default values for missing keys
        return cls(
            cors_allow_origins=data.get('cors_allow_origins', []),
            host=data.get('host', '127.0.0.1'),
            port=data.get('port', 8000),
            dev=data.get('dev', False),
        )

# might need lock
_config : Config | None = None

def reload_config() -> Config:
    _config = Config.load()
    return _config

def get_config() -> Config:
    if not _config:
        return reload_config()
    return _config