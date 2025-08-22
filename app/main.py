from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.config import get_config
from app.middlewares.security_headers import SecurityHeadersMiddleware
from app.routers.health import router as health_router
from app.routers.messages import router as messages_router

app = FastAPI(title='cc-proxy', version='0.1.0')
app.include_router(health_router)
app.include_router(messages_router)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return ORJSONResponse(status_code=exc.status_code, content={'detail': exc.detail})

def main():
    import uvicorn
    config = get_config()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_allow_origins,
        allow_credentials=True,
        allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        allow_headers=['*'],
    )

    app.add_middleware(SecurityHeadersMiddleware)

    uvicorn.run(
        'main:app',
        host=config.host,
        port=config.port,
        reload=config.dev,
    )

if __name__ == '__main__':
    main()
