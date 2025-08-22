from fastapi import APIRouter

router = APIRouter()


@router.post('/v1/messages')
async def messages():
    return []
