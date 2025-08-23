import traceback
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.common.models import ClaudeRequest
from app.dependencies.services import Services, get_services
from app.services.pipeline.exceptions import PipelineException
from app.services.pipeline.models import ClaudeError, ClaudeErrorDetail

router = APIRouter()


@router.post('/v1/messages')
async def messages(
    claude_request: ClaudeRequest,
    services: Annotated[Services, Depends(get_services)],
    request: Request,
):
    pipeline_service = services.messages_pipeline
    dumper = services.dumper
    error_handler = services.error_handler

    # Generate correlation ID for request tracing
    correlation_id = error_handler.generate_correlation_id()

    # Convert ClaudeRequest to dict for compatibility with current pipeline
    payload = claude_request.model_dump()

    handles = dumper.begin(request, payload, correlation_id)

    async def generator():
        try:
            if claude_request.stream:
                # Use streaming processing
                async for chunk in pipeline_service.process_stream(claude_request, request, correlation_id):
                    dumper.write_chunk(handles, chunk.data)
                    yield chunk.data
            else:
                # Use non-streaming processing
                response = await pipeline_service.process_request(claude_request, request, correlation_id)
                response_bytes = response.to_bytes()
                dumper.write_chunk(handles, response_bytes)
                yield response_bytes
        except httpx.HTTPStatusError as e:
            # Handle HTTP status errors (most specific first)
            print('http request err', e)
            domain_exception = error_handler.convert_httpx_exception(e, correlation_id)
            error_data, sse_event = error_handler.get_error_response_data(domain_exception)
            dumper.write_chunk(handles, sse_event.encode('utf-8'))
            yield sse_event
        except httpx.HTTPError as e:
            # Handle other HTTP errors (general case)
            print('proxy request err', e)
            domain_exception = error_handler.convert_httpx_exception(e, correlation_id)
            error_data, sse_event = error_handler.get_error_response_data(domain_exception)
            dumper.write_chunk(handles, sse_event.encode('utf-8'))
            yield sse_event
        except PipelineException as e:
            # Handle domain exceptions
            print('pipeline exception', e)
            error_data, sse_event = error_handler.get_error_response_data(e)
            dumper.write_chunk(handles, sse_event.encode('utf-8'))
            yield sse_event
        except Exception as e:
            # Handle unexpected errors
            err_message = '\n'.join(traceback.format_exception(e))
            print('unknown exception', err_message)
            err = ClaudeError(error=ClaudeErrorDetail(type='api_error', message=err_message))
            error = err.model_dump_json()
            # error_message = (
            #     f'event: error\ndata: {{"type": "error", "error": {{"type": "internal_error", "message": "Internal server error", "correlation_id": "{correlation_id}"}}}}'
            # )
            dumper.write_chunk(handles, error)
            yield error
        finally:
            dumper.close(handles)

    return StreamingResponse(generator(), media_type='application/x-ndjson')
