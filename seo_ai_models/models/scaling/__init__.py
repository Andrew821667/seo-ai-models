# -*- coding: utf-8 -*-
"""
Инициализация пакета scaling.
"""

# Импортируем основные компоненты для удобства доступа
from seo_ai_models.models.scaling.performance.distributed_processing import DistributedProcessing, Task, TaskPriority, TaskStatus
from seo_ai_models.models.scaling.monitoring.system_monitor import SystemMonitor
from seo_ai_models.models.scaling.monitoring.auto_scaling import AutoScaling, CPUUtilizationPolicy
