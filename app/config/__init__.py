from typing import Optional

from app.common.utils import get_app_dir
from app.config.models import ConfigModel


class ConfigurationService:
    """Configuration service that manages config loading without global state."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration service with optional config path."""
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> ConfigModel:
        """Load configuration from file."""
        return ConfigModel.load(self.config_path)

    def get_config(self) -> ConfigModel:
        """Get the configuration instance."""
        return self._config

    def reload_config(self) -> ConfigModel:
        """Reload configuration from file."""
        self._config = self._load_config()
        return self._config




def setup_config() -> None:
    """Create .cc-proxy directory and config.yaml in user's home directory if they don't exist."""

    # Create .cc-proxy directory if it doesn't exist
    cc_proxy_dir = get_app_dir()
    cc_proxy_dir.mkdir(exist_ok=True)

    # Create config.yaml if it doesn't exist
    config_file = cc_proxy_dir / 'config.yaml'
    if not config_file.exists():
        # Create default config using ConfigModel
        config = ConfigModel()
        config.save(str(config_file))
