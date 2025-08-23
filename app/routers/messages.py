import traceback

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.common.models import ClaudeRequest
from app.config.log import get_logger
from app.dependencies.services import get_core_services, get_routing_service, get_service_container
from app.services.error_handling.exceptions import PipelineException
from app.services.error_handling.models import ClaudeError, ClaudeErrorDetail

router = APIRouter()
logger = get_logger(__name__)


@router.post('/v1/messages')
async def messages(
    claude_request: ClaudeRequest,
    request: Request,
):
    """Handle Claude API messages with dynamic routing support."""

    # Get routing service
    routing_service = get_routing_service()
    if not routing_service:
        logger.error('Routing service not available - check user configuration')
        raise HTTPException(status_code=500, detail={'error': {'type': 'configuration_error', 'message': 'Service configuration failed - no routing available'}})

    # Use dynamic routing to determine the appropriate pipeline
    try:
        routing_key, model_id, pipeline_service = routing_service.process_request(claude_request)

        if not pipeline_service:
            if model_id:
                error_msg = f'Model "{model_id}" is not available or not configured properly'
            else:
                error_msg = f'No model configured for routing key "{routing_key}"'

            logger.error(f'Routing failed: {error_msg}')
            raise HTTPException(status_code=400, detail={'error': {'type': 'model_not_found', 'message': error_msg}})

        # Get provider info for logging
        service_container = get_service_container()
        provider_name = service_container.model_registry.get_provider_for_model(model_id) if model_id else 'unknown'
        routing_info = {'routing_key': routing_key, 'model_id': model_id, 'provider': provider_name}

        logger.info(f'Request routed: {routing_key} -> {model_id} -> {provider_name}')

    except Exception as e:
        logger.error(f'Error in request routing: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail={'error': {'type': 'routing_error', 'message': f'Request routing failed: {str(e)}'}})

    # Get core services
    core_services = get_core_services()
    dumper = core_services.dumper
    exception_mapper = core_services.exception_mapper
    error_formatter = core_services.error_formatter

    # Convert ClaudeRequest to dict for compatibility with current pipeline
    payload = claude_request.to_dict()

    # Add routing information to payload for logging/debugging
    payload['_routing'] = routing_info

    handles = dumper.begin(request, payload)

    async def generator():
        try:
            # Use unified processing that always returns SSE-formatted chunks
            # Stream decision is made AFTER transformations inside the pipeline
            async for chunk in pipeline_service.process_unified(claude_request, request):
                dumper.write_chunk(handles, chunk.data)
                yield chunk.data
        except httpx.HTTPStatusError as e:
            # Handle HTTP status errors (most specific first)
            print('http request err', e)
            domain_exception = exception_mapper.map_httpx_exception(e)
            error_data, sse_event = error_formatter.format_for_sse(domain_exception)
            dumper.write_chunk(handles, sse_event.encode('utf-8'))
            yield sse_event
        except httpx.HTTPError as e:
            # Handle other HTTP errors (general case)
            print('proxy request err', e)
            domain_exception = exception_mapper.map_httpx_exception(e)
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
