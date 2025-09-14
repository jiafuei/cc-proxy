import httpx
import orjson
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from app.common.anthropic_errors import extract_error_message, map_http_status_to_anthropic_error
from app.common.dumper import Dumper
from app.common.models import AnthropicRequest
from app.common.sse_converter import convert_json_to_sse
from app.common.vars import get_request_context
from app.config.log import get_logger
from app.dependencies.dumper import get_dumper
from app.dependencies.service_container import get_service_container

router = APIRouter()
logger = get_logger(__name__)


@router.post('/v1/messages')
async def messages(payload: AnthropicRequest, request: Request, dumper: Dumper = Depends(get_dumper)):
    """Handle Anthropic API messages with unified context."""

    # Context is already created by middleware
    ctx = get_request_context()
    ctx.original_model = payload.model

    # Ensure adequate max_tokens when thinking is enabled
    if payload.thinking and payload.thinking.budget_tokens > 0:
        if payload.max_tokens and payload.thinking.budget_tokens > payload.max_tokens:
            payload.max_tokens = min(32000, payload.thinking.budget_tokens + 1)

    # Phase 1: Validation
    try:
        service_container = get_service_container()
        if not service_container:
            logger.error('Service container not available - check configuration')
            return ORJSONResponse({'error': {'type': 'api_error', 'message': 'Service configuration failed'}}, status_code=500)
    except Exception as e:
        logger.error(f'Service container initialization failed: {e}')
        return ORJSONResponse({'error': {'type': 'api_error', 'message': 'Service initialization failed'}}, status_code=500)

    # Get routing result - context is automatically updated
    routing_result = service_container.router.get_provider_for_request(payload)
    if not routing_result.provider:
        logger.error('No provider available for request')
        return ORJSONResponse({'error': {'type': 'model_not_found', 'message': 'No suitable provider found for request'}}, status_code=400)

    # Context is now fully populated with routing info
    logger.info('Processing request')

    # Start dumping for debugging/logging
    dumper_handles = dumper.begin(request, payload.to_dict())

    # Phase 2: Get JSON response from provider (always JSON now)
    try:
        json_response = await routing_result.provider.process_operation('messages', payload, request, routing_result.routing_key, dumper, dumper_handles)

        # Phase 3: Route based on ORIGINAL stream parameter from client
        original_stream_requested = payload.stream is True

        if not original_stream_requested:
            # Non-streaming: return JSON directly
            response_bytes = orjson.dumps(json_response)
            dumper.write_response_chunk(dumper_handles, response_bytes)
            dumper.close(dumper_handles)
            logger.info('Finished processing request')
            return ORJSONResponse(json_response)
        else:
            # Streaming: convert JSON to SSE format
            return StreamingResponse(convert_json_to_sse(json_response, dumper, dumper_handles), media_type='text/event-stream')

    except httpx.HTTPStatusError as e:
        # HTTP error from provider - return with original status code
        error_type = map_http_status_to_anthropic_error(e.response.status_code)
        error_message = extract_error_message(e)
        logger.error(f'Provider error(HTTP {e.response.status_code}): {error_message}')
        dumper.close(dumper_handles)
        return ORJSONResponse({'error': {'type': error_type, 'message': error_message}}, status_code=e.response.status_code)
    except Exception as e:
        # Other processing errors
        logger.error(f'Error processing request: {e}', exc_info=True)
        dumper.close(dumper_handles)
        return ORJSONResponse({'error': {'type': 'api_error', 'message': f'Request processing failed: {str(e)}'}}, status_code=500)


@router.post('/v1/messages/count_tokens')
async def count_tokens(payload: AnthropicRequest, request: Request, dumper: Dumper = Depends(get_dumper)):
    """Handle Anthropic API messages count requests with transparent routing to provider."""

    # Get the service container (router + providers)
    service_container = get_service_container()
    if not service_container:
        logger.error('Service container not available - check configuration')
        raise HTTPException(status_code=500, detail={'error': {'type': 'api_error', 'message': 'Service configuration failed'}})

    # Get routing result - context is automatically updated
    routing_result = service_container.router.get_provider_for_request(payload)
    if not routing_result.provider:
        logger.error('No provider available for request')
        raise HTTPException(status_code=400, detail={'error': {'type': 'model_not_found', 'message': 'No suitable provider found for request'}})

    # Start dumping for debugging/logging
    dumper_handles = dumper.begin(request, payload.to_dict())

    try:
        # Check if provider supports count_tokens operation
        if not routing_result.provider.supports_operation('count_tokens'):
            logger.warning(f'Provider {routing_result.provider.name} does not support count_tokens operation')
            error_response = {'error': {'type': 'not_supported_error', 'message': f'Provider {routing_result.provider.name} does not support token counting'}}
            return ORJSONResponse(error_response, status_code=501)

        # Use the new capability-based processing pipeline
        logger.info('Processing count_tokens request using capability system')
        response_json = await routing_result.provider.process_operation('count_tokens', payload, request, routing_result.routing_key, dumper, dumper_handles)

        return ORJSONResponse(response_json)

    except ValueError as e:
        # Handle capability-specific errors
        logger.warning(f'Unsupported operation: {str(e)}')
        error_response = {'error': {'type': 'not_supported_error', 'message': str(e)}}
        return ORJSONResponse(error_response, status_code=501)

    except httpx.HTTPStatusError as e:
        # Map to proper Anthropic error type and extract message
        error_type = map_http_status_to_anthropic_error(e.response.status_code)
        error_message = extract_error_message(e)
        logger.error(f'Provider error(HTTP {e.response.status_code}): {error_message}')

        error_response = {'error': {'type': error_type, 'message': error_message}}
        return ORJSONResponse(error_response, status_code=e.response.status_code)

    except Exception as e:
        logger.error(f'Error processing count request: {e}', exc_info=True)
        error_response = {'error': {'type': 'api_error', 'message': f'Request processing failed: {str(e)}'}}
        return ORJSONResponse(error_response, status_code=500)

    finally:
        dumper.close(dumper_handles)
