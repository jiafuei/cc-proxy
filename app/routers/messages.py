from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status as Status
from fastapi.responses import ORJSONResponse, StreamingResponse
import httpx

from app.dependencies.services import Services, get_services
from app.services.anthropic.models import MessagesRequest

router = APIRouter()


@router.post('/v1/messages')
async def messages(services: Annotated[Services, Depends(get_services)], request: Request, beta: bool | None = None,):
    service = services.anthropic
    dumper = services.dumper

    payload = await request.json()

    # handles = dumper.begin(request, payload.model_dump())
    handles = dumper.begin(request, payload)

    async def generator():
        try:
            async for chunk in service.stream_response(payload, request):
                dumper.write_chunk(handles, chunk)
                yield chunk
        except httpx.HTTPError as e:
            err = str(e)
            dumper.write_chunk(handles, err)
            yield 'event: error\ndata: {"type": "error", "error": {"type": "invalid_request_error", "message":"' + err +'"}}'
        except httpx.HTTPStatusError as e:
            err = e.response.text
            err_type = "invalid_request_error"
            match e.response.status_code:
                case Status.HTTP_401_UNAUTHORIZED:
                    err_type = "authentication_error"
                case Status.HTTP_403_FORBIDDEN:
                    err_type = "permission_error"
                case Status.HTTP_404_NOT_FOUND:
                    err_type = "not_found_error"
                case Status.HTTP_413_REQUEST_ENTITY_TOO_LARGE:
                    err_type = "request_too_large"
                case Status.HTTP_429_TOO_MANY_REQUESTS:
                    err_type = "rate_limit_error"
                case Status.HTTP_500_INTERNAL_SERVER_ERROR:
                    err_type = "api_error"
                case 529:
                    err_type = "overloaded_error"

            dumper.write_chunk(handles, err)
            yield f'event: error\ndata: {{"type": "error", "error": {{"type": "{err_type}", "message":"{err}"}} }}'
        finally:
            dumper.close(handles)

    return StreamingResponse(generator(), media_type='application/x-ndjson')
