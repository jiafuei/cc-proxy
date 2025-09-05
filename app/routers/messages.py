import httpx
import orjson
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from app.common.anthropic_errors import extract_error_message, map_http_status_to_anthropic_error
from app.common.dumper import Dumper
from app.common.models import AnthropicRequest
from app.config.log import get_logger
from app.dependencies.dumper import get_dumper
from app.dependencies.service_container import get_service_container
from app.routers.sse_converter import convert_json_to_sse

router = APIRouter()
logger = get_logger(__name__)


@router.post('/v1/messages')
async def messages(payload: AnthropicRequest, request: Request, dumper: Dumper = Depends(get_dumper)):
    """Handle Anthropic API messages with always-JSON architecture."""

    # Phase 1: Validation
    service_container = get_service_container()
    if not service_container:
        logger.error('Service container not available - check configuration')
        return ORJSONResponse({'error': {'type': 'api_error', 'message': 'Service configuration failed'}}, status_code=500)

    provider, routing_key = service_container.router.get_provider_for_request(payload)
    if not provider:
        logger.error('No provider available for request')
        return ORJSONResponse({'error': {'type': 'model_not_found', 'message': 'No suitable provider found for request'}}, status_code=400)

    logger.info(f'Request routed to provider: {provider.config.name}, route: {routing_key}')

    # Start dumping for debugging/logging
    dumper_handles = dumper.begin(request, payload.to_dict())

    # Phase 2: Get JSON response from provider (always JSON now)
    try:
        json_response = await provider.process_request(payload, request, routing_key, dumper, dumper_handles)

        # Phase 3: Route based on ORIGINAL stream parameter from client
        original_stream_requested = payload.stream is True

        if not original_stream_requested:
            # Non-streaming: return JSON directly
            response_bytes = orjson.dumps(json_response)
            dumper.write_response_chunk(dumper_handles, response_bytes)
            dumper.close(dumper_handles)
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

    # Get provider for request
    provider, routing_key = service_container.router.get_provider_for_request(payload)
    if not provider:
        logger.error('No provider available for request')
        raise HTTPException(status_code=400, detail={'error': {'type': 'model_not_found', 'message': 'No suitable provider found for request'}})

    # Start dumping for debugging/logging
    dumper_handles = dumper.begin(request, payload.to_dict())

    try:
        # Prepare request data and headers manually
        current_request = payload.to_dict()
        current_headers = {k: v for k, v in dict(request.headers).items() if k.lower() not in ('content-length', 'host', 'connection')}

        # Manually set authorization header with provider's API key
        if provider.config.api_key:
            current_headers['authorization'] = f'Bearer {provider.config.api_key}'
            current_headers.pop('x-api-key', None)

        # Dump transformed request and headers
        logger.info(f'Count request routed to provider: {provider.config.name}, route: {routing_key}', headers=current_headers)
        dumper.write_transformed_headers(dumper_handles, current_headers)
        dumper.write_transformed_request(dumper_handles, current_request)

        # Send non-streaming request to provider
        # Create config with count_tokens URL
        count_tokens_config = provider.config.model_copy()
        count_tokens_config.url += '/count_tokens'
        response = await provider._send_request(count_tokens_config, current_request, current_headers)

        # Parse JSON response
        response_json = response.json()

        # Dump response
        response_bytes = orjson.dumps(response_json)
        dumper.write_response_chunk(dumper_handles, response_bytes)

        return ORJSONResponse(response_json)

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
