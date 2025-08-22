from typing import Annotated

from fastapi import APIRouter, Depends, Header
from fastapi.responses import StreamingResponse

from app.dependencies.services import Services, get_services
from app.services.anthropic.models import MessagesRequest

router = APIRouter()


@router.post('/v1/messages')
async def messages(payload: MessagesRequest, services: Annotated[Services, Depends(get_services)], headers: Annotated[dict | None, Header()] = None):
    service = services.anthropic

    async def generator():
        async for chunk in service.stream_response(payload):
            yield chunk

    return StreamingResponse(generator(), media_type='text/event-stream')
