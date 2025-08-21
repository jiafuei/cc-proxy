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