import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
import orjson

from app.common.anthropic_errors import extract_error_message, map_http_status_to_anthropic_error
from app.common.dumper import Dumper
from app.common.models import AnthropicRequest
from app.config.log import get_logger
from app.dependencies.dumper import get_dumper
from app.dependencies.service_container import get_service_container

router = APIRouter()
logger = get_logger(__name__)


@router.post('/v1/messages')
async def messages(payload: AnthropicRequest, request: Request, dumper: Dumper = Depends(get_dumper)) -> StreamingResponse:
    """Handle Anthropic API messages with simplified routing and provider system."""

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

    logger.info(f'Request routed to provider: {provider.config.name}, route: {routing_key}')

    # Start dumping for debugging/logging
    dumper_handles = dumper.begin(request, payload.to_dict())

    async def generate():
        try:
            # Process through provider (handles transformers + HTTP + streaming)
            async for chunk in provider.process_request(payload, request, routing_key, dumper, dumper_handles):
                dumper.write_response_chunk(dumper_handles, chunk)
                yield chunk

        except httpx.HTTPStatusError as e:
            # Map to proper Anthropic error type and extract message
            error_type = map_http_status_to_anthropic_error(e.response.status_code)
            error_message = extract_error_message(e)
            logger.error(f'Provider error(HTTP {e.response.status_code}): {error_message}')

            # Format error as SSE
            error_chunk = f'event: error\ndata: {{"type": "error", "error": {{"type": f"{error_type}", "message": "{error_message}"}} }}\n\n'

            dumper.write_response_chunk(dumper_handles, error_chunk.encode())
            yield error_chunk.encode()

        except Exception as e:
            logger.error(f'Error processing request: {e}', exc_info=True)

            # Format error as SSE
            error_message = f'Request processing failed: {str(e)}'
            error_chunk = f'event: error\ndata: {{"type": "error", "error": {{"type": "api_error", "message": "{error_message}"}}}}\n\n'

            dumper.write_response_chunk(dumper_handles, error_chunk.encode())
            yield error_chunk.encode()

        finally:
            dumper.close(dumper_handles)

    return StreamingResponse(generate(), media_type='text/event-stream')
