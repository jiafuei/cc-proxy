import traceback

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.common.models import ClaudeRequest
from app.config.log import get_logger
from app.dependencies.services import get_core_services, get_message_pipeline_service, get_services
from app.services.error_handling.exceptions import PipelineException
from app.services.error_handling.models import ClaudeError, ClaudeErrorDetail
from app.services.lifecycle.service_builder import DynamicServices

router = APIRouter()
logger = get_logger(__name__)


@router.post('/v1/messages')
async def messages(
    claude_request: ClaudeRequest,
    request: Request,
):
    """Handle Claude API messages with dynamic routing support."""

    # Get services with dynamic routing support
    try:
        services = get_services()
    except Exception as e:
        logger.warning(f'Failed to get dynamic services, using fallback: {e}')
        services = get_core_services()

    # Try to use dynamic routing if available
    pipeline_service = None
    routing_info = None

    if isinstance(services, DynamicServices):
        try:
            # Use dynamic routing to determine the appropriate pipeline
            routing_key, model_id, dynamic_pipeline_service = services.process_request_with_routing(claude_request)

            if dynamic_pipeline_service:
                pipeline_service = dynamic_pipeline_service
                routing_info = {'routing_key': routing_key, 'model_id': model_id, 'provider': services.model_registry.get_provider_for_model(model_id)}
                logger.info(f'Using dynamic routing: {routing_key} -> {model_id} -> {routing_info["provider"]}')
            else:
                logger.warning(f'Dynamic routing failed for {routing_key} -> {model_id}, falling back to default pipeline')
        except Exception as e:
            logger.error(f'Error in dynamic routing: {e}', exc_info=True)

    # Fall back to default pipeline if dynamic routing failed or not available
    if pipeline_service is None:
        pipeline_service = get_message_pipeline_service()
        routing_info = {'routing_key': 'fallback', 'model_id': 'default', 'provider': 'anthropic'}
        logger.debug('Using fallback pipeline service')

    # Get core services
    core_services = get_core_services()
    dumper = core_services.dumper
    exception_mapper = core_services.exception_mapper
    error_formatter = core_services.error_formatter

    # Convert ClaudeRequest to dict for compatibility with current pipeline
    payload = claude_request.to_dict()

    # Add routing information to payload for logging/debugging
    if routing_info:
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
