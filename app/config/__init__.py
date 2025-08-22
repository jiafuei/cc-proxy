# Thread-safe lock for config reloading
import threading
from pathlib import Path

from app.config.models import ConfigModel

_config_lock = threading.Lock()

# Global config instance
_config: ConfigModel | None = None


def reload_config() -> ConfigModel:
    """Reload the global configuration from file in a thread-safe manner."""
    global _config
    with _config_lock:
        _config = ConfigModel.load()
        return _config


def get_config() -> ConfigModel:
    """Get the global configuration instance, loading it if not already loaded."""
    global _config
    if not _config:
        return reload_config()
    return _config


def setup_user_config() -> None:
    """Create .cc-proxy directory and config.yaml in user's home directory if they don't exist."""
    # Get user's home directory
    home_dir = Path.home()

    # Create .cc-proxy directory if it doesn't exist
    cc_proxy_dir = home_dir / '.cc-proxy'
    cc_proxy_dir.mkdir(exist_ok=True)

    # Create config.yaml if it doesn't exist
    config_file = cc_proxy_dir / 'config.yaml'
    if not config_file.exists():
        # Create default config using ConfigModel
        config = ConfigModel()
        config.save(str(config_file))
