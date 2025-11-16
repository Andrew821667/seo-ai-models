"""
Webhook service with database support - Full CRUD implementation.
"""

import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from ..database.repositories import WebhookRepository, ProjectRepository

logger = logging.getLogger(__name__)


class WebhookServiceDB:
    """
    Service for managing webhooks with database backend.
    Full CRUD implementation with proper validation and access control.
    """

    def __init__(self, db: Session):
        self.db = db
        self.webhook_repo = WebhookRepository(db)
        self.project_repo = ProjectRepository(db)
        logger.info("WebhookServiceDB initialized with database backend")

    def check_project_access(self, project_id: str, user_id: str) -> bool:
        """Check if user has access to project."""
        project = self.project_repo.get_by_id(project_id)
        if not project:
            return False
        return project.owner_id == user_id

    # ===== WEBHOOK CRUD =====

    def create_webhook(
        self,
        project_id: str,
        url: str,
        events: List[str],
        user_id: str,
        description: str = "",
        secret: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Create a new webhook."""
        # Check project access
        if not self.check_project_access(project_id, user_id):
            return {"success": False, "error": "Access denied to project"}

        try:
            # Validate URL
            if not url.startswith(("http://", "https://")):
                return {"success": False, "error": "Invalid webhook URL. Must start with http:// or https://"}

            # Validate events
            if not events or not isinstance(events, list):
                return {"success": False, "error": "Events must be a non-empty list"}

            webhook_data = {
                "id": str(uuid.uuid4()),
                "project_id": project_id,
                "url": url,
                "events": events,
                "description": description,
                "status": "active",
                "secret": secret or str(uuid.uuid4()),
                "success_count": 0,
                "error_count": 0,
                "metadata": metadata or {},
            }

            webhook = self.webhook_repo.create(webhook_data)

            logger.info(f"Webhook created: {webhook.id} for project {project_id}")
            return {
                "success": True,
                "webhook": {
                    "id": webhook.id,
                    "project_id": webhook.project_id,
                    "url": webhook.url,
                    "events": webhook.events,
                    "description": webhook.description,
                    "status": webhook.status,
                    "secret": webhook.secret,
                    "success_count": webhook.success_count,
                    "error_count": webhook.error_count,
                    "last_triggered_at": webhook.last_triggered_at.isoformat() if webhook.last_triggered_at else None,
                    "created_at": webhook.created_at.isoformat(),
                    "updated_at": webhook.updated_at.isoformat(),
                    "metadata": webhook.extra_metadata,
                },
            }
        except Exception as e:
            logger.error(f"Failed to create webhook: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_webhook(self, webhook_id: str, user_id: str) -> Dict[str, Any]:
        """Get webhook by ID."""
        webhook = self.webhook_repo.get_by_id(webhook_id)

        if not webhook:
            return {"success": False, "error": "Webhook not found"}

        # Check project access
        if not self.check_project_access(webhook.project_id, user_id):
            return {"success": False, "error": "Access denied"}

        return {
            "success": True,
            "webhook": {
                "id": webhook.id,
                "project_id": webhook.project_id,
                "url": webhook.url,
                "events": webhook.events,
                "description": webhook.description,
                "status": webhook.status,
                "secret": webhook.secret,
                "success_count": webhook.success_count,
                "error_count": webhook.error_count,
                "last_triggered_at": webhook.last_triggered_at.isoformat() if webhook.last_triggered_at else None,
                "created_at": webhook.created_at.isoformat(),
                "updated_at": webhook.updated_at.isoformat(),
                "metadata": webhook.extra_metadata,
            },
        }

    def list_webhooks(
        self,
        project_id: str,
        user_id: str,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List webhooks for a project."""
        # Check project access
        if not self.check_project_access(project_id, user_id):
            return {"success": False, "error": "Access denied"}

        try:
            webhooks = self.webhook_repo.get_by_project(project_id, status, skip, limit)

            webhook_list = []
            for webhook in webhooks:
                webhook_list.append(
                    {
                        "id": webhook.id,
                        "project_id": webhook.project_id,
                        "url": webhook.url,
                        "events": webhook.events,
                        "description": webhook.description,
                        "status": webhook.status,
                        "secret": webhook.secret,
                        "success_count": webhook.success_count,
                        "error_count": webhook.error_count,
                        "last_triggered_at": webhook.last_triggered_at.isoformat() if webhook.last_triggered_at else None,
                        "created_at": webhook.created_at.isoformat(),
                        "updated_at": webhook.updated_at.isoformat(),
                        "metadata": webhook.extra_metadata,
                    }
                )

            return {"success": True, "webhooks": webhook_list, "total": len(webhook_list)}
        except Exception as e:
            logger.error(f"Failed to list webhooks: {str(e)}")
            return {"success": False, "error": str(e)}

    def update_webhook(
        self, webhook_id: str, user_id: str, update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update webhook."""
        webhook = self.webhook_repo.get_by_id(webhook_id)

        if not webhook:
            return {"success": False, "error": "Webhook not found"}

        # Check project access
        if not self.check_project_access(webhook.project_id, user_id):
            return {"success": False, "error": "Access denied"}

        try:
            # Validate URL if provided
            if "url" in update_data and update_data["url"]:
                if not update_data["url"].startswith(("http://", "https://")):
                    return {"success": False, "error": "Invalid webhook URL. Must start with http:// or https://"}

            # Validate events if provided
            if "events" in update_data and update_data["events"]:
                if not isinstance(update_data["events"], list) or not update_data["events"]:
                    return {"success": False, "error": "Events must be a non-empty list"}

            # Remove None values
            cleaned_data = {k: v for k, v in update_data.items() if v is not None}

            updated_webhook = self.webhook_repo.update(webhook_id, cleaned_data)

            if not updated_webhook:
                return {"success": False, "error": "Failed to update webhook"}

            logger.info(f"Webhook updated: {webhook_id} by user {user_id}")
            return {
                "success": True,
                "webhook": {
                    "id": updated_webhook.id,
                    "project_id": updated_webhook.project_id,
                    "url": updated_webhook.url,
                    "events": updated_webhook.events,
                    "description": updated_webhook.description,
                    "status": updated_webhook.status,
                    "secret": updated_webhook.secret,
                    "success_count": updated_webhook.success_count,
                    "error_count": updated_webhook.error_count,
                    "last_triggered_at": updated_webhook.last_triggered_at.isoformat() if updated_webhook.last_triggered_at else None,
                    "created_at": updated_webhook.created_at.isoformat(),
                    "updated_at": updated_webhook.updated_at.isoformat(),
                    "metadata": updated_webhook.extra_metadata,
                },
            }
        except Exception as e:
            logger.error(f"Failed to update webhook {webhook_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    def delete_webhook(self, webhook_id: str, user_id: str) -> Dict[str, Any]:
        """Delete webhook (hard delete)."""
        webhook = self.webhook_repo.get_by_id(webhook_id)

        if not webhook:
            return {"success": False, "error": "Webhook not found"}

        # Check project access
        if not self.check_project_access(webhook.project_id, user_id):
            return {"success": False, "error": "Access denied. Only project owner can delete webhooks."}

        try:
            success = self.webhook_repo.delete(webhook_id)

            if success:
                logger.info(f"Webhook {webhook_id} deleted by user {user_id}")
                return {
                    "success": True,
                    "webhook_id": webhook_id,
                    "message": "Webhook has been deleted",
                }
            else:
                return {"success": False, "error": "Failed to delete webhook"}
        except Exception as e:
            logger.error(f"Failed to delete webhook {webhook_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    def trigger_webhook(
        self, webhook_id: str, event: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Trigger a webhook with an event and payload.
        This is a placeholder for actual webhook triggering logic.
        """
        webhook = self.webhook_repo.get_by_id(webhook_id)

        if not webhook:
            return {"success": False, "error": "Webhook not found"}

        if webhook.status != "active":
            return {"success": False, "error": "Webhook is not active"}

        if event not in webhook.events:
            return {"success": False, "error": f"Webhook does not listen to event '{event}'"}

        try:
            # TODO: Implement actual HTTP request to webhook URL
            # This would typically use httpx or requests to send POST request
            # with payload and secret in headers

            # For now, just increment success count
            self.webhook_repo.increment_success(webhook_id)

            logger.info(f"Webhook {webhook_id} triggered for event '{event}'")
            return {
                "success": True,
                "webhook_id": webhook_id,
                "event": event,
                "message": "Webhook triggered successfully",
            }
        except Exception as e:
            # Increment error count on failure
            self.webhook_repo.increment_error(webhook_id)
            logger.error(f"Failed to trigger webhook {webhook_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_webhooks_by_event(self, event: str) -> List[Dict[str, Any]]:
        """Get all active webhooks listening to a specific event."""
        try:
            webhooks = self.webhook_repo.get_by_event(event, status="active")

            webhook_list = []
            for webhook in webhooks:
                webhook_list.append(
                    {
                        "id": webhook.id,
                        "project_id": webhook.project_id,
                        "url": webhook.url,
                        "events": webhook.events,
                        "secret": webhook.secret,
                    }
                )

            return webhook_list
        except Exception as e:
            logger.error(f"Failed to get webhooks for event '{event}': {str(e)}")
            return []
