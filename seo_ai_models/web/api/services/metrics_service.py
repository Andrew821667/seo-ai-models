"""
Prometheus metrics service for monitoring.
"""

import logging
import time
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

# Create custom registry
REGISTRY = CollectorRegistry()

# HTTP Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY,
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Database metrics
db_queries_total = Counter(
    'db_queries_total',
    'Total database queries',
    ['operation', 'table'],
    registry=REGISTRY
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation', 'table'],
    registry=REGISTRY
)

# Cache metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type'],
    registry=REGISTRY
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type'],
    registry=REGISTRY
)

# Application metrics
active_users = Gauge(
    'active_users',
    'Number of active users',
    registry=REGISTRY
)

projects_total = Gauge(
    'projects_total',
    'Total number of projects',
    ['status'],
    registry=REGISTRY
)

tasks_total = Gauge(
    'tasks_total',
    'Total number of tasks',
    ['status'],
    registry=REGISTRY
)

webhooks_total = Gauge(
    'webhooks_total',
    'Total number of webhooks',
    ['status'],
    registry=REGISTRY
)

webhook_triggers_total = Counter(
    'webhook_triggers_total',
    'Total webhook triggers',
    ['status'],
    registry=REGISTRY
)

# Error metrics
errors_total = Counter(
    'errors_total',
    'Total errors',
    ['type', 'endpoint'],
    registry=REGISTRY
)


class MetricsService:
    """Service for collecting application metrics."""

    def __init__(self):
        self.registry = REGISTRY
        logger.info("MetricsService initialized")

    def record_http_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Record HTTP request metrics."""
        http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=status_code
        ).inc()

        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

    def record_db_query(
        self, operation: str, table: str, duration: float
    ):
        """Record database query metrics."""
        db_queries_total.labels(
            operation=operation,
            table=table
        ).inc()

        db_query_duration_seconds.labels(
            operation=operation,
            table=table
        ).observe(duration)

    def record_cache_hit(self, cache_type: str = "redis"):
        """Record cache hit."""
        cache_hits_total.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str = "redis"):
        """Record cache miss."""
        cache_misses_total.labels(cache_type=cache_type).inc()

    def record_error(self, error_type: str, endpoint: str):
        """Record error."""
        errors_total.labels(type=error_type, endpoint=endpoint).inc()

    def record_webhook_trigger(self, status: str):
        """Record webhook trigger."""
        webhook_triggers_total.labels(status=status).inc()

    def update_active_users(self, count: int):
        """Update active users count."""
        active_users.set(count)

    def update_projects_count(self, status: str, count: int):
        """Update projects count."""
        projects_total.labels(status=status).set(count)

    def update_tasks_count(self, status: str, count: int):
        """Update tasks count."""
        tasks_total.labels(status=status).set(count)

    def update_webhooks_count(self, status: str, count: int):
        """Update webhooks count."""
        webhooks_total.labels(status=status).set(count)

    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get Prometheus metrics content type."""
        return CONTENT_TYPE_LATEST


# Global metrics service instance
_metrics_service = None


def get_metrics_service() -> MetricsService:
    """Get metrics service singleton."""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service
