from functools import lru_cache

import httpx

from app.common.dumper import Dumper
from app.config import get_config
from app.services.error_handling.error_formatter import ApiErrorFormatter
from app.services.error_handling.exception_mapper import HttpExceptionMapper
from app.services.pipeline.http_client import HttpClientService
from app.services.pipeline.messages_service import MessagesPipelineService
from app.services.pipeline.request_pipeline import RequestPipeline
from app.services.pipeline.response_pipeline import ResponsePipeline
from app.services.sse_formatter.anthropic_formatter import AnthropicSseFormatter
from app.services.transformers.anthropic.transformers import AnthropicRequestTransformer, AnthropicResponseTransformer, AnthropicStreamTransformer


class Services:
    def __init__(self):
        self.config = get_config()
        self.httpx_client = httpx.AsyncClient(timeout=60 * 5, http2=True)
        self.create_services()

    def create_services(self):
        # Initialize transformers
        anthropic_request_transformer = AnthropicRequestTransformer(self.config)
        anthropic_response_transformer = AnthropicResponseTransformer()
        anthropic_stream_transformer = AnthropicStreamTransformer()

        # Initialize pipelines
        request_pipeline = RequestPipeline([anthropic_request_transformer])
        response_pipeline = ResponsePipeline([anthropic_response_transformer], [anthropic_stream_transformer])

        # Initialize HTTP client
        http_client = HttpClientService(self.httpx_client)

        # Initialize SSE formatter
        sse_formatter = AnthropicSseFormatter()

        # Initialize pipeline service with SSE formatter
        self.messages_pipeline = MessagesPipelineService(request_pipeline, response_pipeline, http_client, sse_formatter)

        # Initialize error handling services
        self.exception_mapper = HttpExceptionMapper()
        self.error_formatter = ApiErrorFormatter()

        # Other services
        self.dumper = Dumper(self.config)


@lru_cache(maxsize=1)
def get_services():
    services = Services()
    return services
