"""Codex channel router scaffolding."""

from typing import Any, Dict

import httpx
import orjson
from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import ORJSONResponse

from app.api.errors import extract_error_message, map_http_status_to_anthropic_error
from app.config.log import get_logger
from app.dependencies import get_service_container_dependency
from app.dependencies.container import ServiceContainer
from app.dependencies.dumper import get_dumper
from app.observability.dumper import Dumper
from app.routing.exchange import ExchangeRequest

router = APIRouter(prefix='/codex/v1', tags=['codex'])
logger = get_logger(__name__)


@router.post('/responses')
async def responses(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    service_container: ServiceContainer = Depends(get_service_container_dependency),
    dumper: Dumper = Depends(get_dumper),
):
    if not service_container:
        logger.error('Service container not available - check configuration')
        return ORJSONResponse({'error': {'type': 'api_error', 'message': 'Service configuration failed'}}, status_code=500)

    model = payload.get('model')
    if not model:
        return ORJSONResponse({'error': {'type': 'invalid_request_error', 'message': 'Request must include a model field'}}, status_code=400)

    original_stream = bool(payload.get('stream'))
    exchange_request = ExchangeRequest.from_payload(
        payload,
        channel='codex',
        model=model,
        original_stream=original_stream,
        tools=payload.get('tools') or [],
        metadata={},
        extras={},
    )

    try:
        routing_result = service_container.router.route(exchange_request)
    except Exception as exc:
        logger.error('Routing failure for codex request: %s', exc)
        return ORJSONResponse({'error': {'type': 'model_not_found', 'message': str(exc)}}, status_code=400)

    if routing_result.provider is None:
        logger.error('No provider available for codex model alias %s', routing_result.model_alias)
        return ORJSONResponse({'error': {'type': 'model_not_found', 'message': 'No provider available for requested model'}}, status_code=400)

    dumper_handles = dumper.begin(request, payload)

    try:
        if not routing_result.provider.supports_operation('responses'):
            logger.warning('Provider %s does not support responses operation', routing_result.provider.config.name)
            return ORJSONResponse(
                {'error': {'type': 'not_supported_error', 'message': f'Provider {routing_result.provider.config.name} does not support responses'}},
                status_code=501,
            )

        response = await routing_result.provider.execute(
            'responses',
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
