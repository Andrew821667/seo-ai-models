# -*- coding: utf-8 -*-
"""
Перечисления для Freemium-модели.
"""

from enum import Enum, auto


class FreemiumPlan(str, Enum):
    """Планы Freemium-модели."""

    FREE = "free"
    MICRO = "micro"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
