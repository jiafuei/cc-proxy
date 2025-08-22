from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.services.anthropic.models import MessagesRequest
from app.dependencies.anthropic import get_anthropic_service
from app.services.anthropic.streaming import AnthropicStreamingService

router = APIRouter()


@router.post('/v1/messages')
async def messages(payload: MessagesRequest | None = None, service: AnthropicStreamingService = Depends(get_anthropic_service)):
    if payload is None:
        return []

    try:
        payload = MessagesRequest(**payload.dict())
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    async def generator():
        async for chunk in service.stream_response(payload):
            yield chunk

    return StreamingResponse(generator(), media_type="text/event-stream")
