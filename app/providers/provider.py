"""Provider clients and manager built around provider descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import Request

from app.config.log import get_logger
from app.config.user_models import ModelConfig, ProviderConfig
from app.observability.dumper import Dumper, DumpHandles
from app.providers.descriptors import ProviderDescriptor
from app.providers.registry import get_descriptor
from app.providers.types import ChannelName, all_channels
from app.routing.exchange import ExchangeRequest, ExchangeResponse
from app.transformers.loader import TransformerLoader

logger = get_logger(__name__)


@dataclass
class ModelMapping:
    """Maps model alias to provider name and actual model ID."""

    provider_name: str
    model_id: str


@dataclass
class TransformerPipeline:
    """Instantiated transformers grouped by stage."""

    request: List[Any]
    response: List[Any]
    stream: List[Any]


class ProviderClient:
    """Provider client aware of descriptors, channels, and transformer pipelines."""

    def __init__(self, config: ProviderConfig, transformer_loader: TransformerLoader):
        self.config = config
        self.descriptor: ProviderDescriptor = get_descriptor(config.type)

        requested_capabilities = set(config.capabilities or self.descriptor.operations)
        unsupported = requested_capabilities - set(self.descriptor.operations)
        if unsupported:
            raise ValueError(f"Capabilities {unsupported} not supported by provider type '{config.type.value}'")

        self.capabilities = requested_capabilities

        self._pipelines: Dict[ChannelName, TransformerPipeline] = {}
        for channel in all_channels():
            channel_defaults = self.descriptor.default_transformers.get(channel, {})
            stage_config = config.channel_transformers(channel)

            stage_configs: Dict[str, List[Dict[str, Any]]] = {}
            for stage in ('request', 'response', 'stream'):
                # Get full override and pre/post configs for this stage
                full_transformers = getattr(stage_config, stage)
                pre_transformers = getattr(stage_config, f'pre_{stage}')
                post_transformers = getattr(stage_config, f'post_{stage}')

                # Build the merged configuration: pre + (full OR defaults) + post
                merged_configs = []

                # Add pre-transformers
                if pre_transformers:
                    merged_configs.extend([cfg.to_loader_dict() for cfg in pre_transformers])

                # Add main transformers (either full override or defaults)
                if full_transformers is not None:
                    # Full override specified (including empty list)
                    merged_configs.extend([cfg.to_loader_dict() for cfg in full_transformers])
                else:
                    # Use provider defaults
                    merged_configs.extend(list(channel_defaults.get(stage, [])))

                # Add post-transformers
                if post_transformers:
                    merged_configs.extend([cfg.to_loader_dict() for cfg in post_transformers])

                stage_configs[stage] = merged_configs

                # Log the composition for debugging
                pre_count = len(pre_transformers or [])
                main_count = len(full_transformers or []) if full_transformers is not None else len(channel_defaults.get(stage, []))
                post_count = len(post_transformers or [])
                main_type = 'override' if full_transformers is not None else 'default'

                if pre_count > 0 or post_count > 0 or main_count > 0:
                    logger.debug(
                        "Provider '%s' channel '%s' stage '%s': %d pre + %d %s + %d post transformers", config.name, channel, stage, pre_count, main_count, main_type, post_count
                    )

            self._pipelines[channel] = TransformerPipeline(
                request=transformer_loader.load_transformers(stage_configs['request']),
                response=transformer_loader.load_transformers(stage_configs['response']),
                stream=transformer_loader.load_transformers(stage_configs['stream']),
            )

        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(config.timeout), http2=True)

        logger.info(
            "Provider '%s' initialized (type=%s, capabilities=%s)",
            config.name,
            config.type.value,
            sorted(self.capabilities),
        )

    # ------------------------------------------------------------------
    # Capability helpers
    # ------------------------------------------------------------------
    def supports_operation(self, operation: str) -> bool:
        return operation in self.capabilities

    def list_operations(self) -> List[str]:
        return sorted(self.capabilities)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _get_pipeline(self, channel: ChannelName) -> TransformerPipeline:
        return self._pipelines[channel]

    def _build_operation_url(self, operation: str, model: Optional[str]) -> str:
        if operation not in self.descriptor.base_url_suffixes:
            raise ValueError(f"Operation '{operation}' not supported by provider type '{self.config.type.value}'")

        suffix = self.descriptor.base_url_suffixes[operation]
        if '{model}' in suffix:
            if not model:
                raise ValueError(f"Operation '{operation}' requires a resolved model identifier")
            suffix = suffix.format(model=model)

        return f'{self.config.base_url.rstrip("/")}{suffix}'

    async def execute(
        self,
        operation: str,
        exchange_request: ExchangeRequest,
        *,
        original_request: Request,
        dumper: Dumper,
        dumper_handles: DumpHandles,
        resolved_model: Optional[str] = None,
    ) -> ExchangeResponse:
        if not self.supports_operation(operation):
            raise ValueError(f"Operation '{operation}' not enabled for provider '{self.config.name}'")

        pipeline = self._get_pipeline(exchange_request.channel)
        request_payload = exchange_request.payload
        if hasattr(request_payload, 'to_dict'):
            current_request = request_payload.to_dict()
        else:
            current_request = dict(request_payload)

        current_headers = dict(original_request.headers)
        # Remove auth headers for security, use transformers to inject proper auth headers
        current_headers.pop('x-api-key', None)
        current_headers.pop('authorization', None)

        current_request['stream'] = False

        for transformer in pipeline.request:
            transform_params = {
                'request': current_request,
                'headers': current_headers,
                'provider_config': self.config,
                'original_request': original_request,
                'routing_key': exchange_request.metadata.get('routing_key'),
                'exchange_request': exchange_request,
            }
            current_request, current_headers = await transformer.transform(transform_params)

        dumper.write_transformed_headers(dumper_handles, current_headers.copy())
        dumper.write_transformed_request(dumper_handles, current_request)

        url = self._build_operation_url(operation, resolved_model)
        response = await self.http_client.post(url, json=current_request, headers=current_headers)

        try:
            response.raise_for_status()
        except Exception:
            await response.aread()
            raise

        response_text = response.text
        dumper.write_pretransformed_response(dumper_handles, response_text)

        response_json = response.json()

        for transformer in pipeline.response:
            response_params = {
                'response': response_json,
                'request': current_request,
                'final_headers': current_headers,
                'provider_config': self.config,
                'original_request': original_request,
                'exchange_request': exchange_request,
            }
            response_json = await transformer.transform_response(response_params)

        return ExchangeResponse(
            channel=exchange_request.channel,
            model=exchange_request.model,
            payload=response_json,
            stream=exchange_request.original_stream,
            metadata={'operation': operation, 'provider': self.config.name},
        )

    async def close(self) -> None:
        await self.http_client.aclose()


class ProviderManager:
    """Manages provider clients and model mappings."""

    def __init__(self, providers_config: List[ProviderConfig], models_config: List[ModelConfig], transformer_loader: TransformerLoader):
        self.providers: Dict[str, ProviderClient] = {}
        self.alias_mappings: Dict[str, ModelMapping] = {}

        self._load_providers(providers_config, transformer_loader)
        self._build_model_mapping(models_config)

    # ------------------------------------------------------------------
    def _load_providers(self, providers_config: List[ProviderConfig], transformer_loader: TransformerLoader) -> None:
        for provider_config in providers_config:
            try:
                provider = ProviderClient(provider_config, transformer_loader)
                self.providers[provider_config.name] = provider
                logger.info("Loaded provider '%s'", provider_config.name)
            except Exception as exc:
                logger.error("Failed to load provider '%s': %s", provider_config.name, exc, exc_info=True)

    def _build_model_mapping(self, models_config: List[ModelConfig]) -> None:
        for model_config in models_config:
            if model_config.provider in self.providers:
                self.alias_mappings[model_config.alias] = ModelMapping(provider_name=model_config.provider, model_id=model_config.id)
            else:
                logger.warning("Model '%s' references unknown provider '%s'", model_config.alias, model_config.provider)

    # ------------------------------------------------------------------
    def get_provider_for_model(self, alias: str) -> Optional[Tuple[ProviderClient, str]]:
        mapping = self.alias_mappings.get(alias)
        if not mapping:
            return None

        provider = self.providers.get(mapping.provider_name)
        return (provider, mapping.model_id) if provider else None

    def get_provider_by_name(self, name: str) -> Optional[ProviderClient]:
        return self.providers.get(name)

    def list_providers(self) -> List[str]:
        return sorted(self.providers.keys())

    def list_models(self) -> List[str]:
        return sorted(self.alias_mappings.keys())

    async def close_all(self) -> None:
        for provider in self.providers.values():
            await provider.close()
