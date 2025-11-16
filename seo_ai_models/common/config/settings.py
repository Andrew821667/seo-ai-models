"""
Application settings configuration.
"""

import secrets

# Default settings
settings = {
    "jwt_secret_key": secrets.token_urlsafe(32),
    "access_token_expire_minutes": 30,
    "allowed_origins": ["*"],  # In production, specify exact origins
}


def get(key: str, default=None):
    """Get setting value."""
    return settings.get(key, default)
