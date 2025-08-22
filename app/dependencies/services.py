from functools import lru_cache

import httpx

from app.common.dumper import Dumper
from app.config import get_config
from app.services.anthropic.client import AnthropicStreamingService


class Services:
    def __init__(self):
        self.httpx_client = httpx.AsyncClient(timeout=60 * 5)
        self.anthropic = AnthropicStreamingService(self.httpx_client)
        self.config = get_config()
        self.dumper = Dumper(self.config)


@lru_cache(maxsize=1)
def get_services():
    return Services()
