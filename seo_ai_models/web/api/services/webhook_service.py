"""
Сервисный слой для работы с webhooks.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class Webhook:
    """Класс, представляющий webhook."""

    def __init__(
        self,
        webhook_id: str,
        url: str,
        events: list,
        project_id: Optional[str] = None,
        description: str = "",
        status: str = "active",
        created_at: Optional[datetime] = None,
        secret: Optional[str] = None,
    ):
        self.webhook_id = webhook_id
        self.url = url
        self.events = events
        self.project_id = project_id
        self.description = description
        self.status = status
        self.created_at = created_at or datetime.now()
        self.secret = secret


class WebhookService:
    """Сервис для управления webhooks."""

    def __init__(self):
        self.webhooks: Dict[str, Webhook] = {}
        logger.info("WebhookService initialized")

    def delete_webhook(self, webhook_id: str, user_id: str) -> Dict[str, Any]:
        """
        Удаляет webhook (soft delete).

        Args:
            webhook_id: ID webhook
            user_id: ID пользователя

        Returns:
            Dict[str, Any]: Результат операции
        """
        webhook = self.webhooks.get(webhook_id)

        if not webhook:
            logger.warning(f"Webhook {webhook_id} not found for deletion")
            return {"success": False, "error": "Webhook not found"}

        # Помечаем как деактивированный
        webhook.status = "deleted"

        logger.info(f"Webhook {webhook_id} deleted by user {user_id}")

        return {
            "success": True,
            "webhook_id": webhook_id,
            "message": "Webhook has been deactivated",
        }
