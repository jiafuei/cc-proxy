import traceback
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.common.utils import generate_correlation_id
from app.common.models import ClaudeRequest
from app.dependencies.services import Services, get_services
from app.services.error_handling.exceptions import PipelineException
from app.services.error_handling.models import ClaudeError, ClaudeErrorDetail

router = APIRouter()


@router.post('/v1/messages')
async def messages(
    claude_request: ClaudeRequest,
    services: Annotated[Services, Depends(get_services)],
    request: Request,
):
    pipeline_service = services.messages_pipeline
    dumper = services.dumper
    exception_mapper = services.exception_mapper
    error_formatter = services.error_formatter

    # Generate correlation ID for request tracing
    correlation_id = generate_correlation_id()

    # Convert ClaudeRequest to dict for compatibility with current pipeline
    payload = claude_request.to_dict()

    handles = dumper.begin(request, payload, correlation_id)

    async def generator():
        try:
            # Use unified processing that always returns SSE-formatted chunks
            # Stream decision is made AFTER transformations inside the pipeline
            async for chunk in pipeline_service.process_unified(claude_request, request, correlation_id):
                dumper.write_chunk(handles, chunk.data)
                yield chunk.data
        except httpx.HTTPStatusError as e:
            # Handle HTTP status errors (most specific first)
            print('http request err', e)
            domain_exception = exception_mapper.map_httpx_exception(e, correlation_id)
            error_data, sse_event = error_formatter.format_for_sse(domain_exception)
            dumper.write_chunk(handles, sse_event.encode('utf-8'))
            yield sse_event
        except httpx.HTTPError as e:
            # Handle other HTTP errors (general case)
            print('proxy request err', e)
            domain_exception = exception_mapper.map_httpx_exception(e, correlation_id)
            error_data, sse_event = error_formatter.format_for_sse(domain_exception)
            dumper.write_chunk(handles, sse_event.encode('utf-8'))
            yield sse_event
        except PipelineException as e:
            # Handle domain exceptions
            print('pipeline exception', e)
            error_data, sse_event = error_formatter.format_for_sse(e)
            dumper.write_chunk(handles, sse_event.encode('utf-8'))
            yield sse_event
        except Exception as e:
            # Handle unexpected errors
            err_message = '\n'.join(traceback.format_exception(e))
            print('unknown exception', err_message)
            err = ClaudeError(error=ClaudeErrorDetail(type='api_error', message=err_message))
            error = err.model_dump_json()
            dumper.write_chunk(handles, error)
            yield error
        finally:
            dumper.close(handles)

    return StreamingResponse(generator(), media_type='text/event-stream')
