"""Legacy shim forwarding /v1 endpoints to the Claude router."""

from fastapi import APIRouter, Depends, Request

from app.api.claude import count_tokens as claude_count_tokens
from app.api.claude import messages as claude_messages
from app.common.dumper import Dumper
from app.common.models import AnthropicRequest
from app.dependencies import get_service_container_dependency
from app.dependencies.dumper import get_dumper
from app.di.container import ServiceContainer

router = APIRouter(tags=['legacy'])


@router.post('/v1/messages')
async def legacy_messages(
    payload: AnthropicRequest,
    request: Request,
    service_container: ServiceContainer = Depends(get_service_container_dependency),
    dumper: Dumper = Depends(get_dumper),
):
    return await claude_messages(payload=payload, request=request, service_container=service_container, dumper=dumper)


@router.post('/v1/messages/count_tokens')
async def legacy_count_tokens(
    payload: AnthropicRequest,
    request: Request,
    service_container: ServiceContainer = Depends(get_service_container_dependency),
    dumper: Dumper = Depends(get_dumper),
):
    return await claude_count_tokens(payload=payload, request=request, service_container=service_container, dumper=dumper)

