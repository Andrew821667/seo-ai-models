"""
Prometheus metrics endpoint.
"""

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse
import logging

from ..services.metrics_service import get_metrics_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_class=PlainTextResponse)
async def get_metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    This endpoint is typically scraped by Prometheus.
    Public endpoint (no authentication required for monitoring).
    """
    metrics_service = get_metrics_service()

    metrics_data = metrics_service.get_metrics()
    content_type = metrics_service.get_content_type()

    return Response(content=metrics_data, media_type=content_type)


@router.get("/health")
async def metrics_health():
    """
    Health check for metrics service.

    Returns basic health information.
    """
    return {"status": "healthy", "service": "metrics", "metrics_available": True}
