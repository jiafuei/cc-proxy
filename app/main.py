from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.middlewares.security_headers import SecurityHeadersMiddleware
from app.config import Config
from app.routers.health import router as health_router

config = Config.load()

app = FastAPI(title='cc-proxy', version='0.1.0')

app.include_router(health_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_allow_origins,
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allow_headers=['*'],
)

app.add_middleware(SecurityHeadersMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={'detail': exc.detail})


if __name__ == '__main__':
    import uvicorn

    config = Config.load()
    uvicorn.run(
        'main:app',
        host=config.host,
        port=config.port,
        reload=config.dev,
    )
