"""Middleware for capturing service generation per request."""

import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.services.lifecycle.service_provider import DynamicServiceProvider

logger = logging.getLogger(__name__)


class ServiceCaptureMiddleware(BaseHTTPMiddleware):
    """Middleware that captures the current service generation for each request.

    This ensures that each request uses a consistent set of services throughout
    its lifecycle, even if the configuration changes during request processing.
    """

    def __init__(self, app, service_provider: DynamicServiceProvider):
        """Initialize middleware with service provider.

        Args:
            app: FastAPI application
            service_provider: Dynamic service provider
        """
        super().__init__(app)
        self.service_provider = service_provider

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with service generation capture.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response
        """
        # Capture current service generation
        try:
            generation_id, services = self.service_provider.get_current_services()

            # Acquire services for this request
            acquired_services = self.service_provider.acquire_services(generation_id)
            if acquired_services is None:
                logger.error(f'Failed to acquire services for generation {generation_id}')
                # Fall back to current services
                generation_id, services = self.service_provider.get_current_services()
                acquired_services = services

            # Store generation info in request state
            request.state.service_generation_id = generation_id
            request.state.services = acquired_services

            logger.debug(f'Request {id(request)} using service generation {generation_id}')

        except Exception as e:
            logger.error(f'Failed to capture service generation: {e}', exc_info=True)
            # Continue without generation capture - services will fall back to default behavior
            request.state.service_generation_id = None
            request.state.services = None

        try:
            # Process the request
            response = await call_next(request)
            return response

        finally:
            # Release services when request is complete
            generation_id = getattr(request.state, 'service_generation_id', None)
            if generation_id:
                try:
                    self.service_provider.release_services(generation_id)
                    logger.debug(f'Released services for generation {generation_id} after request {id(request)}')
                except Exception as e:
                    logger.error(f'Error releasing services for generation {generation_id}: {e}')


def get_request_services(request: Request):
    """Get services for the current request.

    Args:
        request: Current HTTP request

    Returns:
        Services instance for this request, or None if not captured
    """
    return getattr(request.state, 'services', None)


def get_request_generation_id(request: Request) -> str:
    """Get service generation ID for the current request.

    Args:
        request: Current HTTP request

    Returns:
        Generation ID for this request, or None if not captured
    """
    return getattr(request.state, 'service_generation_id', None)
