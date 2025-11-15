"""
Redis cache service for caching API responses.
"""

import json
import logging
from typing import Optional, Any, Callable
from functools import wraps
import redis
import os

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)
CACHE_TTL = int(os.environ.get("CACHE_TTL", 300))  # 5 minutes default


class CacheService:
    """Redis cache service."""

    def __init__(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logger.info(f"Redis cache connected: {REDIS_HOST}:{REDIS_PORT}")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis cache disabled: {str(e)}")
            self.redis_client = None
            self.enabled = False

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self.enabled:
            return None

        try:
            value = self.redis_client.get(key)
            if value:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(value)
            logger.debug(f"Cache MISS: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache get error for {key}: {str(e)}")
            return None

    def set(self, key: str, value: Any, ttl: int = CACHE_TTL) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        try:
            serialized = json.dumps(value)
            self.redis_client.setex(key, ttl, serialized)
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Cache set error for {key}: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        if not self.enabled:
            return False

        try:
            result = self.redis_client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return bool(result)
        except Exception as e:
            logger.error(f"Cache delete error for {key}: {str(e)}")
            return False

    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.

        Args:
            pattern: Pattern to match (e.g., "user:*")

        Returns:
            Number of keys deleted
        """
        if not self.enabled:
            return 0

        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cache DELETE pattern {pattern}: {deleted} keys")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache delete pattern error for {pattern}: {str(e)}")
            return 0

    def clear_all(self) -> bool:
        """Clear all cache."""
        if not self.enabled:
            return False

        try:
            self.redis_client.flushdb()
            logger.info("Cache CLEARED all")
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
            return False

    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}

        try:
            info = self.redis_client.info("stats")
            return {
                "enabled": True,
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
            }
        except Exception as e:
            logger.error(f"Cache stats error: {str(e)}")
            return {"enabled": True, "error": str(e)}


# Global cache instance
_cache_service = None


def get_cache_service() -> CacheService:
    """Get cache service singleton."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service


def cache_result(ttl: int = CACHE_TTL, key_prefix: str = ""):
    """
    Decorator to cache function results.

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key

    Usage:
        @cache_result(ttl=300, key_prefix="user")
        def get_user(user_id: str):
            return fetch_user_from_db(user_id)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache_service()

            # Generate cache key from function name and arguments
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            # Try to get from cache
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator


def invalidate_cache(pattern: str):
    """
    Delete cache entries matching pattern.

    Args:
        pattern: Pattern to match (e.g., "user:*")
    """
    cache = get_cache_service()
    return cache.delete_pattern(pattern)
