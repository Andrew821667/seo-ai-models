"""
Webhook repository for database operations.
"""

import logging
from datetime import datetime
from typing import Optional, List
from sqlalchemy.orm import Session
from ..models import Webhook

logger = logging.getLogger(__name__)


class WebhookRepository:
    """Repository for Webhook operations."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, webhook_data: dict) -> Webhook:
        """Create a new webhook."""
        webhook = Webhook(**webhook_data)
        self.db.add(webhook)
        self.db.commit()
        self.db.refresh(webhook)
        logger.info(f"Created webhook: {webhook.url} (ID: {webhook.id})")
        return webhook

    def get_by_id(self, webhook_id: str) -> Optional[Webhook]:
        """Get webhook by ID."""
        return self.db.query(Webhook).filter(Webhook.id == webhook_id).first()

    def get_by_project(
        self, project_id: str, status: Optional[str] = None, skip: int = 0, limit: int = 100
    ) -> List[Webhook]:
        """Get webhooks by project."""
        query = self.db.query(Webhook).filter(Webhook.project_id == project_id)

        if status:
            query = query.filter(Webhook.status == status)

        return query.offset(skip).limit(limit).all()

    def get_by_event(self, event: str, status: str = "active") -> List[Webhook]:
        """Get webhooks that listen to a specific event."""
        # Using JSON contains for PostgreSQL or JSON extract for SQLite
        return (
            self.db.query(Webhook)
            .filter(Webhook.status == status)
            .filter(Webhook.events.contains([event]))
            .all()
        )

    def get_all(
        self, status: Optional[str] = None, skip: int = 0, limit: int = 100
    ) -> List[Webhook]:
        """Get all webhooks with pagination."""
        query = self.db.query(Webhook)

        if status:
            query = query.filter(Webhook.status == status)

        return query.offset(skip).limit(limit).all()

    def update(self, webhook_id: str, webhook_data: dict) -> Optional[Webhook]:
        """Update webhook."""
        webhook = self.get_by_id(webhook_id)
        if not webhook:
            return None

        for key, value in webhook_data.items():
            if hasattr(webhook, key) and value is not None:
                setattr(webhook, key, value)

        self.db.commit()
        self.db.refresh(webhook)
        logger.info(f"Updated webhook: {webhook.url} (ID: {webhook.id})")
        return webhook

    def delete(self, webhook_id: str) -> bool:
        """Delete webhook."""
        webhook = self.get_by_id(webhook_id)
        if not webhook:
            return False

        self.db.delete(webhook)
        self.db.commit()
        logger.info(f"Deleted webhook: {webhook.url} (ID: {webhook.id})")
        return True

    def increment_success(self, webhook_id: str):
        """Increment success count."""
        webhook = self.get_by_id(webhook_id)
        if webhook:
            webhook.success_count += 1
            webhook.last_triggered_at = datetime.utcnow()
            self.db.commit()

    def increment_error(self, webhook_id: str):
        """Increment error count."""
        webhook = self.get_by_id(webhook_id)
        if webhook:
            webhook.error_count += 1
            webhook.last_triggered_at = datetime.utcnow()
            self.db.commit()
