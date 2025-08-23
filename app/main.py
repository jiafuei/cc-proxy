import traceback
from pprint import pprint

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse

from app.config import get_config, setup_user_config
from app.config.log import configure_structlog
from app.middlewares.context import ContextMiddleware
from app.middlewares.correlation_id import CorrelationIdMiddleware
from app.middlewares.security_headers import SecurityHeadersMiddleware
from app.routers.health import router as health_router
from app.routers.messages import router as messages_router

app = FastAPI(title='cc-proxy', version='0.1.0')


app.include_router(health_router)
app.include_router(messages_router)

# Set up user config directory and file on startup
setup_user_config()
config = get_config()

# Configure structured logging
configure_structlog()

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


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return ORJSONResponse(status_code=exc.status_code, content={'type': 'error', 'error': {'type': 'invalid_request_error', 'message': traceback.format_exception(exc)}})


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError):
    return ORJSONResponse(status_code=400, content={'type': 'error', 'error': {'type': 'invalid_request_error', 'message': str(exc.errors())}})


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'main:app',
        host=config.host,
        port=config.port,
        reload=config.dev,
    )
