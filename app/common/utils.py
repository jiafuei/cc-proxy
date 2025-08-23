from pathlib import Path
import uuid

def generate_correlation_id() -> str:
    """Generate a new correlation ID for request tracing."""
    return uuid.uuid4().hex

def get_app_dir() -> Path:
    return Path.home() / '.cc-proxy'