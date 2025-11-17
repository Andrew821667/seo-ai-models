"""
WebSocket connection manager for real-time updates.

Manages WebSocket connections and broadcasts analysis progress.
"""

import json
import asyncio
from typing import Dict, List, Set
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        # Active connections by user_id
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Connections by room/channel
        self.rooms: Dict[str, Set[WebSocket]] = {}
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket, user_id: str, room: str = "global"):
        """Accept and register new WebSocket connection."""
        await websocket.accept()

        # Add to user connections
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)

        # Add to room
        if room not in self.rooms:
            self.rooms[room] = set()
        self.rooms[room].add(websocket)

        # Store metadata
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "room": room,
            "connected_at": datetime.utcnow().isoformat(),
        }

        logger.info(f"WebSocket connected: user={user_id}, room={room}")

        # Send welcome message
        await self.send_personal_message(
            {"type": "connection", "status": "connected", "room": room}, websocket
        )

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        metadata = self.connection_metadata.get(websocket)

        if metadata:
            user_id = metadata["user_id"]
            room = metadata["room"]

            # Remove from user connections
            if user_id in self.active_connections:
                self.active_connections[user_id].remove(websocket)
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]

            # Remove from room
            if room in self.rooms:
                self.rooms[room].discard(websocket)
                if not self.rooms[room]:
                    del self.rooms[room]

            # Remove metadata
            del self.connection_metadata[websocket]

            logger.info(f"WebSocket disconnected: user={user_id}, room={room}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific WebSocket."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def send_to_user(self, message: dict, user_id: str):
        """Send message to all connections of a user."""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                await self.send_personal_message(message, connection)

    async def broadcast_to_room(self, message: dict, room: str):
        """Broadcast message to all connections in a room."""
        if room in self.rooms:
            disconnected = []

            for connection in self.rooms[room]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to room {room}: {e}")
                    disconnected.append(connection)

            # Clean up disconnected connections
            for connection in disconnected:
                self.disconnect(connection)

    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all active connections."""
        for room in list(self.rooms.keys()):
            await self.broadcast_to_room(message, room)

    def get_active_users(self) -> List[str]:
        """Get list of active user IDs."""
        return list(self.active_connections.keys())

    def get_room_users(self, room: str) -> List[str]:
        """Get user IDs in a specific room."""
        if room not in self.rooms:
            return []

        user_ids = set()
        for connection in self.rooms[room]:
            metadata = self.connection_metadata.get(connection)
            if metadata:
                user_ids.add(metadata["user_id"])

        return list(user_ids)

    def get_stats(self) -> Dict:
        """Get connection statistics."""
        return {
            "total_connections": sum(len(conns) for conns in self.active_connections.values()),
            "unique_users": len(self.active_connections),
            "rooms": len(self.rooms),
            "rooms_detail": {room: len(connections) for room, connections in self.rooms.items()},
        }


# Global connection manager instance
manager = ConnectionManager()


class AnalysisProgressTracker:
    """Tracks and broadcasts analysis progress."""

    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
        self.active_analyses: Dict[str, Dict] = {}

    async def start_analysis(self, analysis_id: str, user_id: str, url: str):
        """Start tracking analysis."""
        self.active_analyses[analysis_id] = {
            "id": analysis_id,
            "user_id": user_id,
            "url": url,
            "status": "started",
            "progress": 0,
            "current_step": "Initializing",
            "started_at": datetime.utcnow().isoformat(),
            "steps": [],
        }

        await self.broadcast_update(analysis_id)

    async def update_progress(
        self, analysis_id: str, progress: int, current_step: str, details: dict = None
    ):
        """Update analysis progress."""
        if analysis_id not in self.active_analyses:
            return

        analysis = self.active_analyses[analysis_id]
        analysis["progress"] = progress
        analysis["current_step"] = current_step

        # Add step to history
        analysis["steps"].append(
            {
                "step": current_step,
                "progress": progress,
                "timestamp": datetime.utcnow().isoformat(),
                "details": details or {},
            }
        )

        await self.broadcast_update(analysis_id)

    async def complete_analysis(self, analysis_id: str, results: dict):
        """Mark analysis as complete."""
        if analysis_id not in self.active_analyses:
            return

        analysis = self.active_analyses[analysis_id]
        analysis["status"] = "completed"
        analysis["progress"] = 100
        analysis["current_step"] = "Complete"
        analysis["completed_at"] = datetime.utcnow().isoformat()
        analysis["results"] = results

        await self.broadcast_update(analysis_id)

        # Clean up after some time
        await asyncio.sleep(60)  # Keep for 1 minute
        if analysis_id in self.active_analyses:
            del self.active_analyses[analysis_id]

    async def fail_analysis(self, analysis_id: str, error: str):
        """Mark analysis as failed."""
        if analysis_id not in self.active_analyses:
            return

        analysis = self.active_analyses[analysis_id]
        analysis["status"] = "failed"
        analysis["error"] = error
        analysis["failed_at"] = datetime.utcnow().isoformat()

        await self.broadcast_update(analysis_id)

    async def broadcast_update(self, analysis_id: str):
        """Broadcast analysis update."""
        if analysis_id not in self.active_analyses:
            return

        analysis = self.active_analyses[analysis_id]
        user_id = analysis["user_id"]

        message = {"type": "analysis_update", "data": analysis}

        # Send to user
        await self.manager.send_to_user(message, user_id)

        # Also broadcast to global room for admins/observers
        await self.manager.broadcast_to_room(message, "global")

    def get_active_analyses(self) -> List[Dict]:
        """Get all active analyses."""
        return list(self.active_analyses.values())


# Global progress tracker
progress_tracker = AnalysisProgressTracker(manager)
