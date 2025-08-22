from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.dependencies.services import Services, get_services
from app.services.anthropic.models import MessagesRequest

router = APIRouter()


@router.post('/v1/messages')
async def messages(payload: MessagesRequest, services: Annotated[Services, Depends(get_services)], request: Request):
    service = services.anthropic
    dumper = services.dumper

    handles = dumper.begin(request, payload.model_dump())

    async def generator():
        try:
            async for chunk in service.stream_response(payload, request.headers):
                dumper.write_chunk(handles, chunk)
                yield chunk
        finally:
            dumper.close(handles)

    return StreamingResponse(generator(), media_type='text/event-stream')
