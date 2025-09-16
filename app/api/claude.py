import httpx
import orjson
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from app.api.errors import extract_error_message, map_http_status_to_anthropic_error
from app.api.sse import convert_exchange_to_sse
from app.config.log import get_logger
from app.context import get_request_context
from app.dependencies import get_service_container_dependency
from app.dependencies.container import ServiceContainer
from app.dependencies.dumper import get_dumper
from app.models import AnthropicRequest
from app.observability.dumper import Dumper
from app.routing.exchange import ExchangeRequest

router = APIRouter(prefix='/claude/v1', tags=['claude'])
logger = get_logger(__name__)


def _ensure_thinking_budget(payload: AnthropicRequest) -> None:
    if payload.thinking and payload.thinking.budget_tokens > 0:
        if payload.max_tokens and payload.thinking.budget_tokens > payload.max_tokens:
            payload.max_tokens = min(32000, payload.thinking.budget_tokens + 1)


def _create_exchange_request(payload: AnthropicRequest) -> ExchangeRequest:
    original_stream = bool(payload.stream)
    return ExchangeRequest.from_payload(
        payload,
        channel='claude',
        model=payload.model,
        original_stream=original_stream,
        tools=payload.tools or [],
        metadata={},
        extras={},
    )


@router.post('/messages')
async def messages(
    payload: AnthropicRequest,
    request: Request,
    service_container: ServiceContainer = Depends(get_service_container_dependency),
    dumper: Dumper = Depends(get_dumper),
):
    if not service_container:
        logger.error('Service container not available - check configuration')
        return ORJSONResponse({'error': {'type': 'api_error', 'message': 'Service configuration failed'}}, status_code=500)

    ctx = get_request_context()
    ctx.original_model = payload.model
    _ensure_thinking_budget(payload)

    exchange_request = _create_exchange_request(payload)
    routing_result = service_container.router.route(exchange_request)
    if routing_result.provider is None:
        logger.error('No provider available for model alias %s', routing_result.model_alias)
        return ORJSONResponse({'error': {'type': 'model_not_found', 'message': 'No suitable provider found for request'}}, status_code=400)

    dumper_handles = dumper.begin(request, payload.to_dict())

    try:
        response = await routing_result.provider.execute(
            'messages',
            exchange_request,
            original_request=request,
            dumper=dumper,
            dumper_handles=dumper_handles,
            resolved_model=routing_result.resolved_model_id,
        )

        if not exchange_request.original_stream:
            dumper.write_response_chunk(dumper_handles, orjson.dumps(response.payload))
            dumper.close(dumper_handles)
            return ORJSONResponse(response.payload)

        return StreamingResponse(
            convert_exchange_to_sse(response, dumper, dumper_handles),
            media_type='text/event-stream',
        )

    except httpx.HTTPStatusError as exc:
        error_type = map_http_status_to_anthropic_error(exc.response.status_code)
        error_message = extract_error_message(exc)
        logger.error('Provider error (HTTP %s): %s', exc.response.status_code, error_message)
        dumper.close(dumper_handles)
        return ORJSONResponse({'error': {'type': error_type, 'message': error_message}}, status_code=exc.response.status_code)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error('Error processing request: %s', exc, exc_info=True)
        dumper.close(dumper_handles)
        return ORJSONResponse({'error': {'type': 'api_error', 'message': f'Request processing failed: {exc}'}}, status_code=500)


@router.post('/messages/count_tokens')
async def count_tokens(
    payload: AnthropicRequest,
    request: Request,
    service_container: ServiceContainer = Depends(get_service_container_dependency),
    dumper: Dumper = Depends(get_dumper),
):
    if not service_container:
        logger.error('Service container not available - check configuration')
        raise HTTPException(status_code=500, detail={'error': {'type': 'api_error', 'message': 'Service configuration failed'}})

    exchange_request = _create_exchange_request(payload)
    routing_result = service_container.router.route(exchange_request)
    if routing_result.provider is None:
        logger.error('No provider available for model alias %s', routing_result.model_alias)
        raise HTTPException(status_code=400, detail={'error': {'type': 'model_not_found', 'message': 'No suitable provider found for request'}})

    dumper_handles = dumper.begin(request, payload.to_dict())

    try:
        if not routing_result.provider.supports_operation('count_tokens'):
            logger.warning('Provider %s does not support count_tokens', routing_result.provider.config.name)
            return ORJSONResponse(
                {'error': {'type': 'not_supported_error', 'message': f'Provider {routing_result.provider.config.name} does not support token counting'}},
                status_code=501,
            )

        response = await routing_result.provider.execute(
            'count_tokens',
            exchange_request,
            original_request=request,
            dumper=dumper,
            dumper_handles=dumper_handles,
            resolved_model=routing_result.resolved_model_id,
        )

        dumper.write_response_chunk(dumper_handles, orjson.dumps(response.payload))
        return ORJSONResponse(response.payload)

    except httpx.HTTPStatusError as exc:
        error_type = map_http_status_to_anthropic_error(exc.response.status_code)
        error_message = extract_error_message(exc)
        logger.error('Provider error (HTTP %s): %s', exc.response.status_code, error_message)
        return ORJSONResponse({'error': {'type': error_type, 'message': error_message}}, status_code=exc.response.status_code)
    finally:
        dumper.close(dumper_handles)
