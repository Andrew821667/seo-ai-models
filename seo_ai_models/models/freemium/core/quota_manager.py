# -*- coding: utf-8 -*-
"""
QuotaManager - Менеджер квот и ограничений для Freemium-модели.
Отслеживает и контролирует использование ресурсов в бесплатном плане.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from seo_ai_models.models.freemium.core.enums import FreemiumPlan

logger = logging.getLogger(__name__)

class QuotaManager:
    """
    Управляет квотами и ограничениями для Freemium-модели.
    
    Отслеживает использование ресурсов пользователем в рамках
    бесплатного плана и применяет ограничения.
    """
    
    # Стандартные квоты и периоды для бесплатного плана
    DEFAULT_QUOTAS = {
        'analyze_content': 5,       # количество анализов контента
        'analyze_url': 5,           # количество анализов URL
        'llm_tokens': 10000,        # количество токенов LLM
        'api_calls': 0,             # количество API-вызовов (недоступно в бесплатном плане)
        'report_generation': 3      # количество генераций отчетов
    }
    
    # Периоды сброса квот (в днях)
    QUOTA_PERIODS = {
        'analyze_content': 30,      # сброс раз в 30 дней
        'analyze_url': 30,          # сброс раз в 30 дней
        'llm_tokens': 30,           # сброс раз в 30 дней
        'api_calls': 30,            # сброс раз в 30 дней
        'report_generation': 30     # сброс раз в 30 дней
    }
    
    def __init__(
        self, 
        user_id: Optional[str] = None,
        plan: Union[str, FreemiumPlan] = FreemiumPlan.FREE,
        storage_path: Optional[str] = None
    ):
        """
        Инициализирует QuotaManager.
        
        Args:
            user_id: Идентификатор пользователя
            plan: План пользователя
            storage_path: Путь для хранения данных о квотах
        """
        self.user_id = user_id or "anonymous"
        self.plan = plan if isinstance(plan, FreemiumPlan) else FreemiumPlan(plan)
        self.storage_path = storage_path or os.path.join(
            os.path.expanduser("~"), 
            ".seo_ai_models", 
            "quotas"
        )
        
        # Создаем директорию для хранения данных о квотах, если она не существует
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Загружаем текущие квоты пользователя
        self.user_quotas = self._load_user_quotas()
        
        # Проверяем и обновляем периоды квот
        self._check_quota_periods()
    
    def _get_quota_file_path(self) -> str:
        """
        Возвращает путь к файлу с квотами пользователя.
        
        Returns:
            Путь к файлу с квотами
        """
        return os.path.join(self.storage_path, f"{self.user_id}_quotas.json")
    
    def _load_user_quotas(self) -> Dict[str, Any]:
        """
        Загружает текущие квоты пользователя из файла.
        
        Returns:
            Словарь с квотами пользователя
        """
        quota_file = self._get_quota_file_path()
        
        if os.path.exists(quota_file):
            try:
                with open(quota_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading quotas for user {self.user_id}: {e}")
                # В случае ошибки создаем новые квоты
                return self._initialize_user_quotas()
        else:
            # Если файл не существует, создаем новые квоты
            return self._initialize_user_quotas()
    
    def _save_user_quotas(self):
        """Сохраняет текущие квоты пользователя в файл."""
        quota_file = self._get_quota_file_path()
        
        try:
            with open(quota_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_quotas, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Error saving quotas for user {self.user_id}: {e}")
    
    def _initialize_user_quotas(self) -> Dict[str, Any]:
        """
        Инициализирует квоты пользователя.
        
        Returns:
            Словарь с инициализированными квотами
        """
        now = datetime.now().isoformat()
        
        quotas = {
            'user_id': self.user_id,
            'plan': self.plan.value,
            'last_update': now,
            'quotas': {}
        }
        
        # Инициализируем квоты для разных типов операций
        for operation, limit in self.DEFAULT_QUOTAS.items():
            reset_date = (datetime.now() + timedelta(days=self.QUOTA_PERIODS[operation])).isoformat()
            
            quotas['quotas'][operation] = {
                'limit': limit,
                'used': 0,
                'reset_date': reset_date
            }
        
        # Сохраняем инициализированные квоты
        self.user_quotas = quotas
        self._save_user_quotas()
        
        return quotas
    
    def _check_quota_periods(self):
        """Проверяет периоды квот и сбрасывает их, если период истек."""
        now = datetime.now()
        updated = False
        
        for operation, quota_info in self.user_quotas['quotas'].items():
            reset_date = datetime.fromisoformat(quota_info['reset_date'])
            
            if now >= reset_date:
                # Период квоты истек, сбрасываем использование
                quota_info['used'] = 0
                quota_info['reset_date'] = (now + timedelta(days=self.QUOTA_PERIODS[operation])).isoformat()
                updated = True
        
        if updated:
            self.user_quotas['last_update'] = now.isoformat()
            self._save_user_quotas()
    
    def check_quota(self, operation: str) -> bool:
        """
        Проверяет, не превышена ли квота для указанной операции.
        
        Args:
            operation: Тип операции
            
        Returns:
            True, если квота не превышена, иначе False
        """
        self._check_quota_periods()
        
        if operation not in self.user_quotas['quotas']:
            logger.warning(f"Unknown operation type: {operation}")
            return False
        
        quota_info = self.user_quotas['quotas'][operation]
        return quota_info['used'] < quota_info['limit']
    
    def update_quota(self, operation: str, amount: int = 1) -> bool:
        """
        Обновляет использование квоты для указанной операции.
        
        Args:
            operation: Тип операции
            amount: Количество единиц использования
            
        Returns:
            True, если квота успешно обновлена, иначе False
        """
        self._check_quota_periods()
        
        if operation not in self.user_quotas['quotas']:
            logger.warning(f"Unknown operation type: {operation}")
            return False
        
        quota_info = self.user_quotas['quotas'][operation]
        quota_info['used'] += amount
        self.user_quotas['last_update'] = datetime.now().isoformat()
        
        self._save_user_quotas()
        return True
    
    def check_and_update_quota(self, operation: str, amount: int = 1) -> bool:
        """
        Проверяет и обновляет квоту для указанной операции.
        
        Args:
            operation: Тип операции
            amount: Количество единиц использования
            
        Returns:
            True, если квота не превышена и успешно обновлена, иначе False
        """
        if not self.check_quota(operation):
            return False
        
        return self.update_quota(operation, amount)
    
    def get_remaining_quota(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Возвращает информацию об оставшейся квоте.
        
        Args:
            operation: Тип операции (если None, возвращает для всех типов)
            
        Returns:
            Словарь с информацией об оставшейся квоте
        """
        self._check_quota_periods()
        
        result = {'plan': self.plan.value}
        
        if operation:
            if operation not in self.user_quotas['quotas']:
                logger.warning(f"Unknown operation type: {operation}")
                return {'error': f"Unknown operation type: {operation}"}
            
            quota_info = self.user_quotas['quotas'][operation]
            result[operation] = {
                'remaining': quota_info['limit'] - quota_info['used'],
                'limit': quota_info['limit'],
                'used': quota_info['used'],
                'reset_date': quota_info['reset_date']
            }
        else:
            result['quotas'] = {}
            
            for op, quota_info in self.user_quotas['quotas'].items():
                result['quotas'][op] = {
                    'remaining': quota_info['limit'] - quota_info['used'],
                    'limit': quota_info['limit'],
                    'used': quota_info['used'],
                    'reset_date': quota_info['reset_date']
                }
        
        return result
    
    def reset_quota(self, operation: Optional[str] = None):
        """
        Сбрасывает использование квоты для указанной операции.
        
        Args:
            operation: Тип операции (если None, сбрасывает для всех типов)
        """
        if operation:
            if operation in self.user_quotas['quotas']:
                quota_info = self.user_quotas['quotas'][operation]
                quota_info['used'] = 0
                quota_info['reset_date'] = (datetime.now() + timedelta(days=self.QUOTA_PERIODS[operation])).isoformat()
            else:
                logger.warning(f"Unknown operation type: {operation}")
        else:
            for op, quota_info in self.user_quotas['quotas'].items():
                quota_info['used'] = 0
                quota_info['reset_date'] = (datetime.now() + timedelta(days=self.QUOTA_PERIODS[op])).isoformat()
        
        self.user_quotas['last_update'] = datetime.now().isoformat()
        self._save_user_quotas()
