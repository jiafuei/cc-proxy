import logging
from pprint import pprint

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse

from app.config import get_config, setup_config
from app.config.log import configure_structlog, get_logger
from app.dependencies.dumper import get_dumper
from app.dependencies.service_container import get_service_container
from app.middlewares.context import ContextMiddleware
from app.middlewares.correlation_id import CorrelationIdMiddleware
from app.middlewares.security_headers import SecurityHeadersMiddleware
from app.routers.config import router as config_router
from app.routers.health import router as health_router
from app.routers.messages import router as messages_router

app = FastAPI(title='cc-proxy', version='0.1.0')
for k in logging.root.manager.loggerDict.keys():
    if any(k.startswith(v) for v in {'fastapi', 'uvicorn', 'httpx', 'httpcore', 'hpack'}):
        logging.getLogger(k).setLevel('INFO')


app.include_router(config_router)
app.include_router(health_router)
app.include_router(messages_router)

# Set up user config directory and file on startup
setup_config()
config = get_config()

# Configure structured logging
configure_structlog()

# Initialize service container
service_container = get_service_container()

if config.dev:
    pprint(config.model_dump())

# Middlewares are executed LIFO
app.add_middleware(GZipMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_allow_origins,
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allow_headers=['*'],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(ContextMiddleware)

logger = get_logger(__name__)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return ORJSONResponse(status_code=exc.status_code, content={'type': 'error', 'error': {'type': 'api_error', 'message': 'http exception: ' + str(exc.detail)}})


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError):
    req_body = await request.json()
    dumper = get_dumper()
    handles = dumper.begin(request=request, payload=req_body)
    logger.debug('validation error', body=req_body)
    try:
        error_msg = f'request validation error: {str(exc.errors())}'
        dumper.write_response_chunk(handles, error_msg)
        return ORJSONResponse(status_code=400, content={'type': 'error', 'error': {'type': 'invalid_request_error', 'message': error_msg}})
    finally:
        dumper.close(handles)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'main:app',
        host=config.host,
        port=config.port,
        reload=config.dev,
    )
