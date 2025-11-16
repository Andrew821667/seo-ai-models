"""
Cache management and monitoring endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
import logging

from ..dependencies import get_cache_service_dependency, get_current_user_id
from ..services.cache_service import CacheService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stats", response_model=Dict[str, Any])
async def get_cache_stats(
    cache: CacheService = Depends(get_cache_service_dependency),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get cache statistics.

    Returns cache hit/miss rates and other metrics.
    Requires authentication.
    """
    logger.info(f"User {user_id} requesting cache stats")
    stats = cache.get_stats()

    # Calculate hit rate if available
    if stats.get("enabled"):
        hits = stats.get("keyspace_hits", 0)
        misses = stats.get("keyspace_misses", 0)
        total = hits + misses

        if total > 0:
            stats["hit_rate"] = round((hits / total) * 100, 2)
        else:
            stats["hit_rate"] = 0.0

    return stats


@router.delete("/clear")
async def clear_cache(
    cache: CacheService = Depends(get_cache_service_dependency),
    user_id: str = Depends(get_current_user_id)
):
    """
    Clear all cache.

    WARNING: This will clear ALL cached data.
    Requires authentication.
    """
    logger.warning(f"User {user_id} clearing all cache")

    if not cache.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache service is not available"
        )

    success = cache.clear_all()

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )

    return {
        "success": True,
        "message": "All cache cleared successfully"
    }


@router.delete("/invalidate/{pattern}")
async def invalidate_cache_pattern(
    pattern: str,
    cache: CacheService = Depends(get_cache_service_dependency),
    user_id: str = Depends(get_current_user_id)
):
    """
    Invalidate cache entries matching pattern.

    Examples:
    - /invalidate/project:* - Clear all project caches
    - /invalidate/webhook:list:* - Clear all webhook list caches

    Requires authentication.
    """
    logger.info(f"User {user_id} invalidating cache pattern: {pattern}")

    if not cache.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache service is not available"
        )

    deleted = cache.delete_pattern(pattern)

    return {
        "success": True,
        "pattern": pattern,
        "keys_deleted": deleted,
        "message": f"Invalidated {deleted} cache entries"
    }


@router.get("/health")
async def cache_health(
    cache: CacheService = Depends(get_cache_service_dependency)
):
    """
    Check cache service health.

    Public endpoint (no authentication required).
    """
    return {
        "status": "healthy" if cache.enabled else "degraded",
        "enabled": cache.enabled,
        "service": "redis"
    }
