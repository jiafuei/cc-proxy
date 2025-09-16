from fastapi import APIRouter

from app.config.log import get_logger

router = APIRouter()
log = get_logger(__name__)


@router.get('/health')
async def health():
    log.debug('Health check ok')
    return {'status': 'ok'}
