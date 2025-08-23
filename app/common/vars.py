from contextvars import ContextVar

correlation_id = ContextVar('correlation_id', default='default-correlation-id')
