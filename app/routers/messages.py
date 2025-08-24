from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.common.models import ClaudeRequest
from app.config.log import get_logger
from app.dependencies.service_container import get_service_container

router = APIRouter()
logger = get_logger(__name__)


@router.post('/v1/messages')
async def messages(payload: ClaudeRequest, request: Request) -> StreamingResponse:
    """Handle Claude API messages with simplified routing and provider system."""

    # Get the service container (router + providers)
    service_container = get_service_container()
    if not service_container:
        logger.error('Service container not available - check configuration')
        raise HTTPException(status_code=500, detail={'error': {'type': 'api_error', 'message': 'Service configuration failed'}})

    # Get provider for request
    provider = service_container.router.get_provider_for_request(payload)
    if not provider:
        logger.error('No provider available for request')
        raise HTTPException(status_code=400, detail={'error': {'type': 'model_not_found', 'message': 'No suitable provider found for request'}})

    logger.info(f'Request routed to provider: {provider.config.name}')

    # Start dumping for debugging/logging
    dumper_handles = service_container.dumper.begin(request, payload.to_dict())

    async def generate():
        try:
            # Process through provider (handles transformers + HTTP + streaming)
            async for chunk in provider.process_request(payload):
                service_container.dumper.write_chunk(dumper_handles, chunk)
                yield chunk

        except Exception as e:
            logger.error(f'Error processing request: {e}', exc_info=True)

            # Format error as SSE
            error_message = f'Request processing failed: {str(e)}'
            error_chunk = f'event: error\ndata: {{"type": "error", "error": {{"type": "api_error", "message": "{error_message}"}}}}\n\n'

            service_container.dumper.write_chunk(dumper_handles, error_chunk.encode())
            yield error_chunk.encode()

        finally:
            service_container.dumper.close(dumper_handles)

    return StreamingResponse(generate(), media_type='text/event-stream')
