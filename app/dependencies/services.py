from functools import lru_cache

import httpx

from app.common.dumper import Dumper
from app.config import get_config
from app.services.pipeline.error_handler import ErrorHandlingService
from app.services.pipeline.http_client import HttpClientService
from app.services.pipeline.messages_service import MessagesPipelineService
from app.services.pipeline.request_pipeline import RequestPipeline
from app.services.pipeline.response_pipeline import ResponsePipeline
from app.services.pipeline.transformers.anthropic import AnthropicRequestTransformer, AnthropicResponseTransformer, AnthropicStreamTransformer


class Services:
    def __init__(self):
        self.config = get_config()
        self.httpx_client = httpx.AsyncClient(timeout=60 * 5, http2=True)
        self.populate_services()

    def populate_services(self):
        anthropic_request_transformer = AnthropicRequestTransformer(self.config)
        anthropic_response_transformer = AnthropicResponseTransformer()
        anthropic_stream_transformer = AnthropicStreamTransformer()

        request_pipeline = RequestPipeline([anthropic_request_transformer])
        response_pipeline = ResponsePipeline([anthropic_response_transformer], [anthropic_stream_transformer])

        http_client = HttpClientService(self.httpx_client)

        self.messages_pipeline = MessagesPipelineService(request_pipeline, response_pipeline, http_client)

        self.dumper = Dumper(self.config)
        self.error_handler = ErrorHandlingService()


@lru_cache(maxsize=1)
def get_services():
    services = Services()
    return services
