"""
WebSocket routes for real-time communication.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session
import logging

from .manager import manager, progress_tracker
from ..auth.service import AuthService
from ..auth.models import User
from ..infrastructure.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
    room: str = Query("global"),
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time updates.

    Query params:
    - token: JWT access token
    - room: Room to join (default: "global")

    Broadcasts:
    - analysis_update: Progress updates for analyses
    - autofix_update: AutoFix execution updates
    - system_stats: System statistics
    """
    # Authenticate user
    token_data = AuthService.decode_access_token(token)

    if not token_data:
        await websocket.close(code=1008, reason="Invalid token")
        return

    user = AuthService.get_user_by_id(db, token_data.user_id)

    if not user or not user.is_active:
        await websocket.close(code=1008, reason="User not found or inactive")
        return

    # Connect
    await manager.connect(websocket, user.id, room)

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()

            message_type = data.get("type")

            # Handle different message types
            if message_type == "ping":
                await manager.send_personal_message(
                    {"type": "pong", "timestamp": data.get("timestamp")},
                    websocket
                )

            elif message_type == "subscribe_analysis":
                analysis_id = data.get("analysis_id")
                # Subscribe to specific analysis updates
                # (Implementation can be extended)
                await manager.send_personal_message(
                    {"type": "subscribed", "analysis_id": analysis_id},
                    websocket
                )

            elif message_type == "get_active_analyses":
                # Send list of active analyses
                analyses = progress_tracker.get_active_analyses()
                await manager.send_personal_message(
                    {"type": "active_analyses", "data": analyses},
                    websocket
                )

            elif message_type == "get_stats":
                # Send connection stats (admin only)
                if user.role.value == "admin":
                    stats = manager.get_stats()
                    await manager.send_personal_message(
                        {"type": "stats", "data": stats},
                        websocket
                    )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"Client disconnected: {user.username}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/analysis/{analysis_id}")
async def analysis_websocket(
    websocket: WebSocket,
    analysis_id: str,
    token: str = Query(...),
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for specific analysis monitoring.

    Provides real-time updates for a single analysis.
    """
    # Authenticate
    token_data = AuthService.decode_access_token(token)

    if not token_data:
        await websocket.close(code=1008, reason="Invalid token")
        return

    user = AuthService.get_user_by_id(db, token_data.user_id)

    if not user or not user.is_active:
        await websocket.close(code=1008, reason="User not found or inactive")
        return

    # Connect to analysis-specific room
    room = f"analysis_{analysis_id}"
    await manager.connect(websocket, user.id, room)

    try:
        # Send current analysis state
        if analysis_id in progress_tracker.active_analyses:
            await manager.send_personal_message(
                {
                    "type": "analysis_state",
                    "data": progress_tracker.active_analyses[analysis_id]
                },
                websocket
            )

        while True:
            data = await websocket.receive_json()

            if data.get("type") == "ping":
                await manager.send_personal_message(
                    {"type": "pong"},
                    websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
