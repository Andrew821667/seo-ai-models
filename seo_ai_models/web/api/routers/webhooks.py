"""
Роутер для управления webhooks для интеграции с внешними системами.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, Request
from typing import List, Dict, Any, Optional
import logging

from ..services.webhook_service_db import WebhookServiceDB
from ..dependencies import get_webhook_service, get_current_user_id

# Создаем объект роутера
router = APIRouter()

logger = logging.getLogger(__name__)


# ===== WEBHOOK ENDPOINTS =====

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_webhook(
    url: str = Body(..., embed=True),
    events: List[str] = Body(..., embed=True),
    project_id: str = Body(..., embed=True),
    description: Optional[str] = Body("", embed=True),
    secret: Optional[str] = Body(None, embed=True),
    metadata: Optional[Dict[str, Any]] = Body(None, embed=True),
    user_id: str = Depends(get_current_user_id),
    service: WebhookServiceDB = Depends(get_webhook_service)
):
    """
    Создание нового webhook.

    Args:
        url: URL для отправки событий.
        events: Список событий, на которые реагирует webhook.
        project_id: ID проекта.
        description: Описание webhook.
        secret: Секретный ключ для подписи (генерируется автоматически если не указан).
        metadata: Дополнительные метаданные.
        user_id: ID текущего пользователя (из токена).
        service: Сервис webhooks (dependency injection).

    Returns:
        Данные созданного webhook.
    """
    result = service.create_webhook(
        project_id=project_id,
        url=url,
        events=events,
        user_id=user_id,
        description=description,
        secret=secret,
        metadata=metadata
    )

    if not result["success"]:
        error = result.get("error", "Unknown error")

        if "access denied" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to project"
            )
        elif "invalid" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error
            )

    return result["webhook"]


@router.get("/", response_model=List[Dict[str, Any]])
async def get_webhooks(
    project_id: str = Query(...),
    status: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    user_id: str = Depends(get_current_user_id),
    service: WebhookServiceDB = Depends(get_webhook_service)
):
    """
    Получение списка webhooks проекта с фильтрацией.

    Args:
        project_id: ID проекта.
        status: Фильтр по статусу.
        skip: Количество пропускаемых записей (для пагинации).
        limit: Максимальное количество возвращаемых записей.
        user_id: ID текущего пользователя (из токена).
        service: Сервис webhooks (dependency injection).

    Returns:
        List[Dict[str, Any]]: Список webhooks.
    """
    result = service.list_webhooks(project_id, user_id, status, skip, limit)

    if not result["success"]:
        error = result.get("error", "Unknown error")

        if "access denied" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error
            )

    return result["webhooks"]


@router.get("/{webhook_id}", response_model=Dict[str, Any])
async def get_webhook(
    webhook_id: str,
    user_id: str = Depends(get_current_user_id),
    service: WebhookServiceDB = Depends(get_webhook_service)
):
    """
    Получение информации о webhook по ID.

    Args:
        webhook_id: ID webhook.
        user_id: ID текущего пользователя (из токена).
        service: Сервис webhooks (dependency injection).

    Returns:
        Dict[str, Any]: Данные webhook.
    """
    result = service.get_webhook(webhook_id, user_id)

    if not result["success"]:
        error = result.get("error", "Unknown error")

        if "not found" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webhook {webhook_id} not found"
            )
        elif "access denied" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error
            )

    return result["webhook"]


@router.put("/{webhook_id}", response_model=Dict[str, Any])
async def update_webhook(
    webhook_id: str,
    url: Optional[str] = Body(None, embed=True),
    events: Optional[List[str]] = Body(None, embed=True),
    description: Optional[str] = Body(None, embed=True),
    status: Optional[str] = Body(None, embed=True),
    metadata: Optional[Dict[str, Any]] = Body(None, embed=True),
    user_id: str = Depends(get_current_user_id),
    service: WebhookServiceDB = Depends(get_webhook_service)
):
    """
    Обновление webhook по ID.

    Args:
        webhook_id: ID webhook.
        url: Новый URL.
        events: Новый список событий.
        description: Новое описание.
        status: Новый статус.
        metadata: Новые метаданные.
        user_id: ID текущего пользователя (из токена).
        service: Сервис webhooks (dependency injection).

    Returns:
        Dict[str, Any]: Обновленные данные webhook.
    """
    update_data = {}
    if url is not None:
        update_data["url"] = url
    if events is not None:
        update_data["events"] = events
    if description is not None:
        update_data["description"] = description
    if status is not None:
        update_data["status"] = status
    if metadata is not None:
        update_data["metadata"] = metadata

    result = service.update_webhook(webhook_id, user_id, update_data)

    if not result["success"]:
        error = result.get("error", "Unknown error")

        if "not found" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webhook {webhook_id} not found"
            )
        elif "access denied" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        elif "invalid" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error
            )

    return result["webhook"]


@router.delete("/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_webhook(
    webhook_id: str,
    user_id: str = Depends(get_current_user_id),
    service: WebhookServiceDB = Depends(get_webhook_service)
):
    """
    Удаление webhook по ID (hard delete).

    Webhook полностью удаляется из базы данных.

    Args:
        webhook_id: ID webhook.
        user_id: ID текущего пользователя.
        service: Сервис webhooks (dependency injection).

    Raises:
        HTTPException: 404 Not Found - webhook не найден.
        HTTPException: 403 Forbidden - недостаточно прав для удаления.

    Returns:
        Статус 204 No Content при успешном удалении.
    """
    logger.info(f"User {user_id} attempting to delete webhook {webhook_id}")

    result = service.delete_webhook(webhook_id, user_id)

    if not result["success"]:
        error = result.get("error", "Unknown error")

        if "not found" in error.lower():
            logger.warning(f"Webhook {webhook_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webhook {webhook_id} not found"
            )
        elif "access denied" in error.lower():
            logger.warning(f"User {user_id} access denied for webhook {webhook_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Only project owner can delete webhooks."
            )
        else:
            logger.error(f"Error deleting webhook {webhook_id}: {error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete webhook: {error}"
            )

    logger.info(f"Webhook {webhook_id} successfully deleted")


@router.post("/{webhook_id}/test", status_code=status.HTTP_200_OK)
async def test_webhook(
    webhook_id: str,
    event: str = Body(..., embed=True),
    payload: Dict[str, Any] = Body(..., embed=True),
    user_id: str = Depends(get_current_user_id),
    service: WebhookServiceDB = Depends(get_webhook_service)
):
    """
    Тестирование webhook путем отправки тестового события.

    Args:
        webhook_id: ID webhook.
        event: Тип события для тестирования.
        payload: Тестовые данные для отправки.
        user_id: ID текущего пользователя (из токена).
        service: Сервис webhooks (dependency injection).

    Returns:
        Результат тестирования webhook.
    """
    # Check if user has access to webhook
    result = service.get_webhook(webhook_id, user_id)

    if not result["success"]:
        error = result.get("error", "Unknown error")

        if "not found" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webhook {webhook_id} not found"
            )
        elif "access denied" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

    # Trigger webhook
    trigger_result = service.trigger_webhook(webhook_id, event, payload)

    if not trigger_result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=trigger_result.get("error", "Failed to trigger webhook")
        )

    return {
        "success": True,
        "message": trigger_result.get("message", "Test event sent successfully"),
        "webhook_id": webhook_id,
        "event": event
    }


# Internal endpoint for getting webhooks by event
@router.get("/internal/by-event/{event}", response_model=List[Dict[str, Any]])
async def get_webhooks_by_event(
    event: str,
    service: WebhookServiceDB = Depends(get_webhook_service)
):
    """
    Получение списка активных webhooks для указанного события.
    Внутренний endpoint для системных операций.

    Args:
        event: Тип события.
        service: Сервис webhooks (dependency injection).

    Returns:
        List[Dict[str, Any]]: Список webhooks.
    """
    webhooks = service.get_webhooks_by_event(event)
    return webhooks


# Receiver for incoming webhook events (from external systems)
@router.post("/incoming/{endpoint_key}", status_code=status.HTTP_200_OK)
async def handle_incoming_webhook(
    endpoint_key: str,
    request: Request
):
    """
    Обработка входящих webhook событий от внешних систем.

    Args:
        endpoint_key: Ключ эндпоинта.
        request: Объект запроса.

    Returns:
        Подтверждение получения события.
    """
    # Get request data
    body = await request.body()
    headers = dict(request.headers)

    # Verify signature if present
    if "X-Webhook-Signature" in headers:
        signature = headers["X-Webhook-Signature"]
        # TODO: Implement signature verification

    # Log event
    logger.info(f"Received webhook event for endpoint {endpoint_key}")

    # TODO: Process event and store in database

    return {
        "success": True,
        "message": "Event received and queued for processing",
        "endpoint_key": endpoint_key
    }
