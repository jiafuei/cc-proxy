from functools import lru_cache

import httpx

from app.common.dumper import Dumper
from app.config import get_config
from app.services.anthropic.client import AnthropicStreamingService


class Services:
    def __init__(self):
        self.config = get_config()
        self.httpx_client = httpx.AsyncClient(timeout=60 * 5, http2=True)
        self.populate_services()

    def populate_services(self):
        self.anthropic = AnthropicStreamingService(self.httpx_client, self.config)
        self.dumper = Dumper(self.config)


@lru_cache(maxsize=1)
def get_services():
    services =  Services()
    return services
