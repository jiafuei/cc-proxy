import uuid

def generate_correlation_id() -> str:
    """Generate a new correlation ID for request tracing."""
    return uuid.uuid4().hex