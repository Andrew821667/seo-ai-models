"""
Middleware for collecting request metrics.
"""

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..services.metrics_service import get_metrics_service

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP request metrics."""

    def __init__(self, app):
        super().__init__(app)
        self.metrics = get_metrics_service()
        logger.info("MetricsMiddleware initialized")

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and collect metrics."""
        start_time = time.time()

        # Get endpoint path (remove query params)
        endpoint = request.url.path

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # Record error
            self.metrics.record_error(
                error_type=type(e).__name__,
                endpoint=endpoint
            )
            logger.error(f"Request error: {endpoint} - {str(e)}")
            raise

        # Calculate duration
        duration = time.time() - start_time

        # Record metrics
        self.metrics.record_http_request(
            method=request.method,
            endpoint=endpoint,
            status_code=status_code,
            duration=duration
        )

        # Add custom headers
        response.headers["X-Process-Time"] = str(duration)

        return response
