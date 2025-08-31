import logging
import logging.handlers
import re
from pathlib import Path

import orjson
import structlog
from structlog.types import FilteringBoundLogger

from app.common.utils import get_app_dir
from app.config import get_config


class _ConsoleFormatter(logging.Formatter):
    """Custom formatter for console output using structlog ConsoleRenderer."""

    def __init__(self):
        super().__init__()
        self.renderer = structlog.dev.ConsoleRenderer()

    def format(self, record):
        # Convert log record to structlog event dict
        event_dict = {
            'event': record.getMessage(),
            'level': record.levelname,
            'timestamp': record.created,
            'logger': record.name,
        }

        # Add any extra attributes
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in (
                    'name',
                    'msg',
                    'args',
                    'levelname',
                    'levelno',
                    'pathname',
                    'filename',
                    'module',
                    'lineno',
                    'funcName',
                    'created',
                    'msecs',
                    'relativeCreated',
                    'thread',
                    'threadName',
                    'processName',
                    'process',
                    'message',
                    'exc_info',
                    'exc_text',
                    'stack_info',
                ):
                    event_dict[key] = value

        # Use ConsoleRenderer to format
        return self.renderer(None, None, event_dict)


class _JSONFormatter(logging.Formatter):
    """Custom formatter for JSON file output using structlog JSONRenderer."""

    def __init__(self):
        super().__init__()
        self.renderer = structlog.processors.JSONRenderer(serializer=lambda *x, **y: orjson.dumps(*x, **y).decode('utf-8'))

    def format(self, record):
        # Convert log record to structlog event dict
        event_dict = {
            'event': record.getMessage(),
            'level': record.levelname,
            'timestamp': record.created,
            'logger': record.name,
        }

        # Add any extra attributes
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in (
                    'name',
                    'msg',
                    'args',
                    'levelname',
                    'levelno',
                    'pathname',
                    'filename',
                    'module',
                    'lineno',
                    'funcName',
                    'created',
                    'msecs',
                    'relativeCreated',
                    'thread',
                    'threadName',
                    'processName',
                    'process',
                    'message',
                    'exc_info',
                    'exc_text',
                    'stack_info',
                ):
                    event_dict[key] = value

        # Use JSONRenderer to format
        return self.renderer(None, None, event_dict)


def _create_log_handlers(log_config, log_dir: Path) -> list:
    """Create logging handlers based on configuration."""
    handlers = []

    if log_config.console_enabled:
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, structlog.processors.format_exc_info, structlog.dev.ConsoleRenderer()]
        )
        console_handler = logging.StreamHandler()
        # console_handler.setFormatter(_ConsoleFormatter())
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if log_config.file_enabled:
        log_file = log_dir / 'app.log'
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(serializer=lambda *x, **y: orjson.dumps(*x, **y).decode('utf-8')),
            ]
        )

        # Parse file size (e.g., "10MB" -> 10 * 1024 * 1024)
        size_match = re.match(r'(\d+)\s*([KMGT]?B?)', log_config.max_file_size.upper())
        if size_match:
            size_num = int(size_match.group(1))
            size_unit = size_match.group(2)
            multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
            max_bytes = size_num * multipliers.get(size_unit, multipliers['MB'])
        else:
            max_bytes = 10 * 1024 * 1024  # Default 10MB

        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=log_config.backup_count)
        # file_handler.setFormatter(_JSONFormatter())
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

        # Also create timed rotating handler if specified
        # if log_config.rotation_when and log_config.rotation_when != 'size':
        #     timed_log_file = log_dir / 'app-timed.log'
        #     timed_handler = logging.handlers.TimedRotatingFileHandler(timed_log_file, when=log_config.rotation_when, backupCount=log_config.backup_count)
        #     timed_handler.setFormatter(formatter)
        #     handlers.append(timed_handler)

    return handlers


def _correlation_id_processor(logger, method_name, event_dict):
    """Add correlation ID to log events if available in context."""
    from app.common.utils import get_correlation_id

    if 'correlation_id' not in event_dict:
        correlation_id = get_correlation_id()
        event_dict['correlation_id'] = correlation_id

    return event_dict


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a configured structlog logger."""
    return structlog.get_logger(name)


def configure_structlog() -> None:
    """Configure structlog with dual output (console + rotating files) and standard library integration."""
    config = get_config()
    log_config = config.logging

    # Setup log directory
    log_dir = Path(log_config.log_file_dir if log_config.log_file_dir else get_app_dir() / 'logs')
    if log_dir.exists() and not log_dir.is_dir(follow_symlinks=True):
        raise Exception(f'Log directory {log_dir} is not a directory')
    if not log_dir.exists():
        log_dir.mkdir(exist_ok=True, parents=True)

    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, log_config.level.upper()),
        handlers=_create_log_handlers(log_config, log_dir),
        format='%(message)s',  # structlog will handle formatting
    )

    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.processors.add_log_level,
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt='ISO', utc=True),
        _correlation_id_processor,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_config.level.upper())),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
