"""Middleware for the API."""

from .metrics_middleware import MetricsMiddleware

__all__ = ["MetricsMiddleware"]
