from dataclasses import dataclass
from typing import List

import yaml


@dataclass
class Config:
    """Configuration class that loads settings from YAML file."""

    cors_allow_origins: List[str]
    host: str = '127.0.0.1'
    port: int = 8000
    dev: bool = False
    dump_requests: bool = False
    dump_responses: bool = False
    dump_headers: bool = False
    dump_dir: str | None = None
    redact_headers: List[str] | None = None

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
            dump_requests=data.get('dump_requests', False),
            dump_responses=data.get('dump_responses', False),
            dump_headers=data.get('dump_headers', False),
            dump_dir=data.get('dump_dir', None),
            redact_headers=data.get('redact_headers', ['authorization', 'x-api-key', 'cookie', 'set-cookie']),
        )


# might need lock
_config: Config | None = None


def reload_config() -> Config:
    _config = Config.load()
    return _config


def get_config() -> Config:
    if not _config:
        return reload_config()
    return _config
