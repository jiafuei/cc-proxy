import uuid
from contextvars import ContextVar

correlation_id = ContextVar('correlation_id', default='default-' + uuid.uuid4().hex[8:])
