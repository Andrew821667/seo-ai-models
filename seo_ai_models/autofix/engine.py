"""
AutoFixEngine - Automatic SEO Problem Fixing System

Система автоматического исправления SEO проблем с поддержкой:
- Классификации проблем по сложности
- Автоматического исправления безопасных проблем
- Backup и rollback механизмов
- Approval workflow для сложных изменений
- Верификации результатов
"""

import logging
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
import copy

logger = logging.getLogger(__name__)


class FixComplexity(str, Enum):
    """Сложность исправления проблемы."""

    TRIVIAL = "trivial"  # Можно исправить автоматически без риска
    SIMPLE = "simple"  # Можно автоматически, минимальный риск
    MODERATE = "moderate"  # Требует проверки, средний риск
    COMPLEX = "complex"  # Требует ручного вмешательства
    CRITICAL = "critical"  # Критическое изменение, требует approval


class FixStatus(str, Enum):
    """Статус исправления."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    REQUIRES_APPROVAL = "requires_approval"


class FixAction:
    """Описание действия для исправления проблемы."""

    def __init__(
        self,
        action_id: str,
        problem_type: str,
        description: str,
        complexity: FixComplexity,
        execute_func: Callable,
        verify_func: Optional[Callable] = None,
        rollback_func: Optional[Callable] = None,
        metadata: Optional[Dict] = None,
    ):
        self.action_id = action_id
        self.problem_type = problem_type
        self.description = description
        self.complexity = complexity
        self.execute_func = execute_func
        self.verify_func = verify_func
        self.rollback_func = rollback_func
        self.metadata = metadata or {}
        self.status = FixStatus.PENDING
        self.backup = None
        self.result = None
        self.error = None
        self.executed_at = None


class AutoFixEngine:
    """
    Движок автоматического исправления SEO проблем.

    Workflow:
    1. Analyze - анализ и выявление проблем
    2. Plan - создание плана исправлений
    3. Execute - автоматическое исправление (для простых)
    4. Review - отправка сложных на review
    5. Verify - проверка результатов
    6. Rollback - откат при ошибках
    """

    def __init__(self, cms_connector=None, auto_execute: bool = True):
        """
        Инициализация движка.

        Args:
            cms_connector: Коннектор к CMS для применения изменений
            auto_execute: Автоматически выполнять простые исправления
        """
        self.cms_connector = cms_connector
        self.auto_execute = auto_execute
        self.actions_registry = {}
        self.executed_actions = []
        self.pending_approvals = []

        logger.info(f"AutoFixEngine initialized (auto_execute={auto_execute})")

    def register_fix_action(
        self,
        problem_type: str,
        complexity: FixComplexity,
        execute_func: Callable,
        verify_func: Optional[Callable] = None,
        rollback_func: Optional[Callable] = None,
    ):
        """
        Регистрация действия для типа проблемы.

        Args:
            problem_type: Тип проблемы (например, "missing_alt_tags")
            complexity: Сложность исправления
            execute_func: Функция для выполнения исправления
            verify_func: Функция для проверки результата
            rollback_func: Функция для отката изменений
        """
        self.actions_registry[problem_type] = {
            "complexity": complexity,
            "execute": execute_func,
            "verify": verify_func,
            "rollback": rollback_func,
        }
        logger.debug(f"Registered fix action: {problem_type} ({complexity})")

    def analyze_and_plan(self, analysis_results: Dict[str, Any]) -> List[FixAction]:
        """
        Анализ результатов и создание плана исправлений.

        Args:
            analysis_results: Результаты анализа сайта

        Returns:
            Список действий для исправления
        """
        actions = []

        # Извлекаем проблемы из анализа
        problems = self._extract_problems(analysis_results)

        for problem in problems:
            problem_type = problem.get("type")

            if problem_type in self.actions_registry:
                action_config = self.actions_registry[problem_type]

                action = FixAction(
                    action_id=f"{problem_type}_{datetime.now().timestamp()}",
                    problem_type=problem_type,
                    description=problem.get("description", ""),
                    complexity=action_config["complexity"],
                    execute_func=action_config["execute"],
                    verify_func=action_config["verify"],
                    rollback_func=action_config["rollback"],
                    metadata=problem.get("metadata", {}),
                )

                actions.append(action)
                logger.info(f"Planned action: {action.description} ({action.complexity})")
            else:
                logger.warning(f"No fix registered for problem type: {problem_type}")

        return actions

    def execute_plan(
        self, actions: List[FixAction], require_approval_for: List[FixComplexity] = None
    ) -> Dict[str, Any]:
        """
        Выполнение плана исправлений.

        Args:
            actions: Список действий для выполнения
            require_approval_for: Список сложностей, требующих approval

        Returns:
            Результаты выполнения
        """
        if require_approval_for is None:
            require_approval_for = [FixComplexity.COMPLEX, FixComplexity.CRITICAL]

        results = {
            "total": len(actions),
            "executed": 0,
            "failed": 0,
            "requires_approval": 0,
            "skipped": 0,
            "actions": [],
        }

        for action in actions:
            # Проверяем, требуется ли approval
            if action.complexity in require_approval_for:
                action.status = FixStatus.REQUIRES_APPROVAL
                self.pending_approvals.append(action)
                results["requires_approval"] += 1
                logger.info(f"Action requires approval: {action.description}")
                continue

            # Автоматическое выполнение для простых исправлений
            if self.auto_execute or action.complexity == FixComplexity.TRIVIAL:
                result = self._execute_action(action)
                results["actions"].append(result)

                if result["status"] == "completed":
                    results["executed"] += 1
                elif result["status"] == "failed":
                    results["failed"] += 1
            else:
                results["skipped"] += 1
                logger.debug(f"Skipped action (auto_execute=False): {action.description}")

        return results

    def _execute_action(self, action: FixAction) -> Dict[str, Any]:
        """
        Выполнение одного действия с backup и rollback.

        Args:
            action: Действие для выполнения

        Returns:
            Результат выполнения
        """
        action.status = FixStatus.IN_PROGRESS
        action.executed_at = datetime.now()

        try:
            # Создаем backup текущего состояния
            logger.info(f"Creating backup for: {action.description}")
            action.backup = self._create_backup(action)

            # Выполняем исправление
            logger.info(f"Executing fix: {action.description}")
            action.result = action.execute_func(
                cms_connector=self.cms_connector, metadata=action.metadata
            )

            # Верификация результата
            if action.verify_func:
                logger.debug(f"Verifying result for: {action.description}")
                verification = action.verify_func(action.result)

                if not verification.get("success"):
                    # Откат если верификация не прошла
                    logger.warning(f"Verification failed, rolling back: {action.description}")
                    self._rollback_action(action)
                    action.status = FixStatus.ROLLED_BACK
                    action.error = verification.get("error")

                    return {
                        "action_id": action.action_id,
                        "status": "rolled_back",
                        "reason": action.error,
                    }

            # Успешное выполнение
            action.status = FixStatus.COMPLETED
            self.executed_actions.append(action)

            logger.info(f"✅ Successfully executed: {action.description}")

            return {
                "action_id": action.action_id,
                "status": "completed",
                "description": action.description,
                "result": action.result,
            }

        except Exception as e:
            # Ошибка при выполнении - откат
            logger.error(f"Error executing action: {action.description} - {str(e)}")
            action.error = str(e)
            action.status = FixStatus.FAILED

            # Пытаемся откатить
            if action.backup:
                try:
                    self._rollback_action(action)
                    action.status = FixStatus.ROLLED_BACK
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {str(rollback_error)}")

            return {"action_id": action.action_id, "status": "failed", "error": str(e)}

    def _create_backup(self, action: FixAction) -> Dict[str, Any]:
        """Создание backup перед изменением."""
        # В зависимости от типа проблемы создаем backup
        # Например, для мета-тегов сохраняем старые значения
        return {
            "timestamp": datetime.now().isoformat(),
            "action_id": action.action_id,
            "metadata": copy.deepcopy(action.metadata),
        }

    def _rollback_action(self, action: FixAction):
        """Откат изменений."""
        if action.rollback_func and action.backup:
            logger.info(f"Rolling back: {action.description}")
            action.rollback_func(cms_connector=self.cms_connector, backup=action.backup)

    def _extract_problems(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Извлечение списка проблем из результатов анализа."""
        problems = []

        # Проверяем разные секции анализа
        if "issues" in analysis_results:
            for issue in analysis_results["issues"]:
                problems.append(
                    {
                        "type": issue.get("type"),
                        "description": issue.get("description"),
                        "metadata": issue.get("metadata", {}),
                    }
                )

        # Проверяем рекомендации
        if "recommendations" in analysis_results:
            for rec in analysis_results["recommendations"]:
                if rec.get("auto_fixable", False):
                    problems.append(
                        {
                            "type": rec.get("type"),
                            "description": rec.get("description"),
                            "metadata": rec.get("metadata", {}),
                        }
                    )

        return problems

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Получение списка действий, требующих approval."""
        return [
            {
                "action_id": action.action_id,
                "description": action.description,
                "complexity": action.complexity,
                "problem_type": action.problem_type,
                "metadata": action.metadata,
            }
            for action in self.pending_approvals
        ]

    def approve_and_execute(self, action_id: str) -> Dict[str, Any]:
        """Одобрение и выполнение действия."""
        action = next((a for a in self.pending_approvals if a.action_id == action_id), None)

        if not action:
            return {"success": False, "error": "Action not found"}

        # Удаляем из pending
        self.pending_approvals.remove(action)

        # Выполняем
        result = self._execute_action(action)

        return {"success": True, "result": result}

    def get_execution_report(self) -> Dict[str, Any]:
        """Получение отчета о выполненных исправлениях."""
        return {
            "total_executed": len(self.executed_actions),
            "pending_approvals": len(self.pending_approvals),
            "actions": [
                {
                    "action_id": action.action_id,
                    "description": action.description,
                    "status": action.status,
                    "executed_at": action.executed_at.isoformat() if action.executed_at else None,
                    "error": action.error,
                }
                for action in self.executed_actions
            ],
        }
