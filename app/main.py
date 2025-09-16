import logging
from pprint import pprint
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse

from app.api.claude import router as claude_router
from app.api.codex import router as codex_router
from app.api.config import router as config_router
from app.api.health import router as health_router
from app.config import ConfigurationService, setup_config
from app.config.log import configure_structlog, get_logger
from app.config.models import ConfigModel
from app.dependencies.container import build_service_container
from app.dependencies.dumper import get_dumper
from app.middlewares.request_context import RequestContextMiddleware
from app.middlewares.security_headers import SecurityHeadersMiddleware


def create_app(config: Optional[ConfigModel] = None) -> FastAPI:
    """Application factory for creating FastAPI instances.

    Args:
        config: Optional configuration. If None, loads default config.

    Returns:
        Configured FastAPI application instance.
    """
    # Set up user config directory and file on startup
    setup_config()

    # Use provided config or create default config service
    config_service = ConfigurationService()
    if config is None:
        config = config_service.get_config()

    # Configure structured logging
    configure_structlog(config_service)

    # Initialize service container with config service
    service_container = build_service_container(config_service)

    # Create FastAPI app
    app = FastAPI(title='cc-proxy', version='0.1.0')

    # Store dependencies in app state
    app.state.config = config
    app.state.config_service = config_service
    app.state.service_container = service_container

    # Configure logging levels
    for k in logging.root.manager.loggerDict.keys():
        if any(k.startswith(v) for v in {'fastapi', 'uvicorn', 'httpx', 'httpcore', 'hpack'}):
            logging.getLogger(k).setLevel('INFO')

    # Register routers
    app.include_router(config_router)
    app.include_router(health_router, prefix='/api', tags=['health'])
    app.include_router(claude_router)
    app.include_router(codex_router)

    # Add middlewares (executed LIFO)
    app.add_middleware(GZipMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_allow_origins,
        allow_credentials=True,
        allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        allow_headers=['*'],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)

    # Add exception handlers
    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, exc: RequestValidationError):
        req_body = await request.json()
        config_service = request.app.state.config_service
        dumper = get_dumper(config_service)
        handles = dumper.begin(request=request, payload=req_body)
        logger = get_logger(__name__)
        logger.debug('validation error', body=req_body)
        try:
            error_msg = f'request validation error: {str(exc.errors())}'
            dumper.write_response_chunk(handles, error_msg)
            return ORJSONResponse(status_code=400, content={'type': 'error', 'error': {'type': 'invalid_request_error', 'message': error_msg}})
        finally:
            dumper.close(handles)

    # Print config in dev mode
    if config.dev:
        pprint(config.model_dump())

    return app


# Create the application instance
app = create_app()


if __name__ == '__main__':
    import uvicorn

    # Get config from the app instance
    config = app.state.config

    uvicorn.run(
        'main:app',
        host=config.host,
        port=config.port,
        reload=config.dev,
    )
