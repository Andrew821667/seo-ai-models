"""
Cached version of WebhookServiceDB with Redis caching.
"""

import logging
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from .webhook_service_db import WebhookServiceDB
from .cache_service import get_cache_service, invalidate_cache

logger = logging.getLogger(__name__)


class WebhookServiceCached(WebhookServiceDB):
    """
    WebhookService with Redis caching layer.
    Extends WebhookServiceDB with caching for read operations.
    """

    def __init__(self, db: Session):
        super().__init__(db)
        self.cache = get_cache_service()
        logger.info("WebhookServiceCached initialized with Redis")

    def _make_cache_key(self, entity: str, *args) -> str:
        """Generate cache key."""
        return f"webhook:{entity}:{':'.join(str(a) for a in args)}"

    def get_webhook(self, webhook_id: str, user_id: str) -> Dict[str, Any]:
        """Get webhook with caching."""
        cache_key = self._make_cache_key("get", webhook_id, user_id)

        cached = self.cache.get(cache_key)
        if cached:
            return cached

        result = super().get_webhook(webhook_id, user_id)

        if result.get("success"):
            self.cache.set(cache_key, result, ttl=300)  # 5 minutes

        return result

    def list_webhooks(
        self,
        project_id: str,
        user_id: str,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List webhooks with caching."""
        cache_key = self._make_cache_key("list", project_id, user_id, status or "all", skip, limit)

        cached = self.cache.get(cache_key)
        if cached:
            return cached

        result = super().list_webhooks(project_id, user_id, status, skip, limit)

        if result.get("success"):
            self.cache.set(cache_key, result, ttl=60)  # 1 minute for lists

        return result

    def create_webhook(
        self, project_id: str, url: str, events: List[str], user_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Create webhook and invalidate cache."""
        result = super().create_webhook(project_id, url, events, user_id, **kwargs)

        if result.get("success"):
            # Invalidate project's webhook list cache
            invalidate_cache(f"webhook:list:{project_id}:*")
            # Invalidate event-based cache
            for event in events:
                invalidate_cache(f"webhook:events:{event}")

        return result

    def update_webhook(
        self, webhook_id: str, user_id: str, update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update webhook and invalidate cache."""
        result = super().update_webhook(webhook_id, user_id, update_data)

        if result.get("success"):
            webhook = result.get("webhook", {})
            project_id = webhook.get("project_id")

            # Invalidate webhook cache
            invalidate_cache(f"webhook:get:{webhook_id}:*")
            if project_id:
                invalidate_cache(f"webhook:list:{project_id}:*")

            # Invalidate event caches if events were updated
            if "events" in update_data:
                invalidate_cache("webhook:events:*")

        return result

    def delete_webhook(self, webhook_id: str, user_id: str) -> Dict[str, Any]:
        """Delete webhook and invalidate cache."""
        # Get webhook before deletion to know what to invalidate
        webhook_info = super().get_webhook(webhook_id, user_id)

        result = super().delete_webhook(webhook_id, user_id)

        if result.get("success") and webhook_info.get("success"):
            webhook = webhook_info.get("webhook", {})
            project_id = webhook.get("project_id")

            # Invalidate all related caches
            invalidate_cache(f"webhook:get:{webhook_id}:*")
            if project_id:
                invalidate_cache(f"webhook:list:{project_id}:*")

            # Invalidate event-based caches
            events = webhook.get("events", [])
            for event in events:
                invalidate_cache(f"webhook:events:{event}")

        return result

    def get_webhooks_by_event(self, event: str) -> List[Dict[str, Any]]:
        """Get webhooks by event with caching."""
        cache_key = f"webhook:events:{event}"

        cached = self.cache.get(cache_key)
        if cached:
            return cached

        result = super().get_webhooks_by_event(event)

        # Cache for shorter time as this is critical data
        self.cache.set(cache_key, result, ttl=30)  # 30 seconds

        return result

    def trigger_webhook(
        self, webhook_id: str, event: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trigger webhook (no caching, but invalidate after)."""
        result = super().trigger_webhook(webhook_id, event, payload)

        if result.get("success"):
            # Invalidate webhook cache as counts changed
            invalidate_cache(f"webhook:get:{webhook_id}:*")

        return result
