"""
Анализ важности факторов для LLM-оптимизации.

Модуль предоставляет функционал для анализа важности различных факторов
контента при оптимизации для LLM-поисковиков.
"""

import re
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from ..service.multi_model_agent import MultiModelAgent
from ..service.cost_estimator import CostEstimator
from ..common.utils import parse_json_response, extract_scores_from_text


class FeatureImportanceAnalyzer:
    """
    Анализ важности факторов для LLM-оптимизации.
    """

    def __init__(self, llm_service: LLMService, prompt_generator: PromptGenerator):
        """
        Инициализирует анализатор важности факторов.

        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM
            prompt_generator: Экземпляр PromptGenerator для генерации промптов
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        self.multi_model_agent = MultiModelAgent(llm_service, prompt_generator)
        self.cost_estimator = CostEstimator()

        # Настройка логгирования
        self.logger = logging.getLogger(__name__)

        # Список стандартных факторов для оценки важности
        self.standard_factors = [
            "Ясность и структурированность",
            "Информативность",
            "Точность формулировок",
            "Релевантность теме",
            "Уникальность информации",
            "Авторитетность источников",
            "Опыт автора",
            "Экспертиза в области",
            "Надежность информации",
            "Использование ключевых слов",
            "Семантическая связность",
            "Качество заголовков",
            "Использование списков",
            "Использование таблиц",
            "Наличие примеров",
            "Наличие данных и статистики",
            "Использование цитат",
            "Наличие выводов",
            "Длина контента",
            "Читабельность",
        ]

    def analyze_feature_importance(
        self,
        factors: Optional[List[str]] = None,
        llm_type: str = "generic",
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Анализирует важность различных факторов для LLM-оптимизации.

        Args:
            factors: Список факторов для анализа (опционально)
            llm_type: Тип LLM (generic, chat, search)
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            Dict[str, Any]: Результат анализа важности факторов
        """
        # Если факторы не указаны, используем стандартный набор
        if not factors:
            factors = self.standard_factors

        # Выбираем оптимальную модель для анализа
        provider, model = self.multi_model_agent.select_optimal_model(
            "analysis", len(" ".join(factors)) * 5, budget
        )

        # Генерируем промпт для анализа важности факторов
        prompt = self._generate_importance_prompt(factors, llm_type)

        # Генерируем ответ
        result = self.llm_service.generate(prompt, provider)

        # Обрабатываем результат анализа
        return self._process_importance_result(result, factors, llm_type, provider, model)

    def compare_feature_importance(
        self,
        llm_types: List[str] = ["generic", "chat", "search"],
        factors: Optional[List[str]] = None,
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Сравнивает важность факторов для разных типов LLM.

        Args:
            llm_types: Список типов LLM для сравнения
            factors: Список факторов для анализа (опционально)
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            Dict[str, Any]: Результат сравнения важности факторов
        """
        # Если факторы не указаны, используем стандартный набор
        if not factors:
            factors = self.standard_factors

        # Анализируем важность факторов для каждого типа LLM
        results = []

        # Распределяем бюджет на типы LLM
        type_budget = None
        if budget is not None:
            type_budget = budget / len(llm_types)

        for llm_type in llm_types:
            result = self.analyze_feature_importance(factors, llm_type, type_budget)
            results.append(result)

        # Формируем сравнительный анализ
        comparison = {
            "llm_types": llm_types,
            "factors": factors,
            "importance_by_type": {},
            "most_important_factors": {},
            "least_important_factors": {},
            "variation_between_types": {},
            "cost": sum(result.get("cost", 0) for result in results),
        }

        # Заполняем важность факторов по типам
        for i, llm_type in enumerate(llm_types):
            comparison["importance_by_type"][llm_type] = results[i].get("factor_importance", {})

        # Определяем наиболее и наименее важные факторы для каждого типа
        for llm_type in llm_types:
            importances = comparison["importance_by_type"][llm_type]
            sorted_factors = sorted(importances.items(), key=lambda x: x[1], reverse=True)

            # Наиболее важные факторы (топ-3)
            comparison["most_important_factors"][llm_type] = [
                {"factor": factor, "importance": importance}
                for factor, importance in sorted_factors[:3]
            ]

            # Наименее важные факторы (последние 3)
            comparison["least_important_factors"][llm_type] = [
                {"factor": factor, "importance": importance}
                for factor, importance in sorted_factors[-3:]
            ]

        # Анализируем вариацию важности между типами
        for factor in factors:
            # Собираем важность фактора для всех типов
            importances = [
                comparison["importance_by_type"][llm_type].get(factor, 0) for llm_type in llm_types
            ]

            # Вычисляем вариацию
            variation = max(importances) - min(importances) if importances else 0

            comparison["variation_between_types"][factor] = variation

        return comparison

    def analyze_content_factors(
        self,
        content: str,
        factors: Optional[List[str]] = None,
        llm_type: str = "generic",
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Анализирует наличие и качество важных факторов в контенте.

        Args:
            content: Текст для анализа
            factors: Список факторов для анализа (опционально)
            llm_type: Тип LLM (generic, chat, search)
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            Dict[str, Any]: Результат анализа факторов в контенте
        """
        # Если факторы не указаны, используем стандартный набор
        if not factors:
            factors = self.standard_factors

        # Выбираем оптимальную модель для анализа
        provider, model = self.multi_model_agent.select_optimal_model(
            "analysis", len(content), budget
        )

        # Генерируем промпт для анализа контента
        prompt = self._generate_content_analysis_prompt(content, factors, llm_type)

        # Генерируем ответ
        result = self.llm_service.generate(prompt, provider)

        # Обрабатываем результат анализа
        return self._process_content_analysis_result(
            result, factors, content, llm_type, provider, model
        )

    def _generate_importance_prompt(self, factors: List[str], llm_type: str) -> str:
        """
        Генерирует промпт для анализа важности факторов.

        Args:
            factors: Список факторов для анализа
            llm_type: Тип LLM

        Returns:
            str: Промпт для анализа
        """
        # Формируем список факторов
        factors_list = "\n".join([f"- {factor}" for factor in factors])

        # Определяем тип LLM для промпта
        llm_type_desc = {
            "generic": "обычных LLM (языковых моделей)",
            "chat": "чат-моделей LLM (ChatGPT, Claude и т.д.)",
            "search": "поисковых LLM (Perplexity, Bing AI и т.д.)",
        }.get(llm_type, "обычных LLM (языковых моделей)")

        # Базовый промпт для анализа важности факторов
        base_prompt = f"""
        Ты эксперт по оптимизации контента для LLM-моделей и поисковых систем на базе LLM.
        
        Проанализируй важность следующих факторов для оптимизации контента для {llm_type_desc}:
        
        {factors_list}
        
        Оцени каждый фактор по шкале от 1 до 10, где:
        - 1: минимальная важность, практически не влияет на результаты
        - 10: критическая важность, существенно влияет на результаты
        
        Для каждого фактора дай подробное объяснение, почему он важен или не важен,
        и как именно он влияет на то, как LLM обрабатывает и ранжирует контент.
        
        Сформируй результат в JSON формате со следующей структурой:
        {{
            "factor_importance": {{
                "Фактор 1": 0,
                "Фактор 2": 0,
                ...
            }},
            "factor_analysis": {{
                "Фактор 1": "Анализ важности фактора 1",
                "Фактор 2": "Анализ важности фактора 2",
                ...
            }},
            "top_factors": ["Фактор 1", "Фактор 2", "Фактор 3"],
            "bottom_factors": ["Фактор 8", "Фактор 9", "Фактор 10"],
            "summary": "Общее резюме анализа"
        }}
        """

        return base_prompt

    def _generate_content_analysis_prompt(
        self, content: str, factors: List[str], llm_type: str
    ) -> str:
        """
        Генерирует промпт для анализа контента.

        Args:
            content: Текст для анализа
            factors: Список факторов для анализа
            llm_type: Тип LLM

        Returns:
            str: Промпт для анализа
        """
        # Формируем список факторов
        factors_list = "\n".join([f"- {factor}" for factor in factors])

        # Определяем тип LLM для промпта
        llm_type_desc = {
            "generic": "обычных LLM (языковых моделей)",
            "chat": "чат-моделей LLM (ChatGPT, Claude и т.д.)",
            "search": "поисковых LLM (Perplexity, Bing AI и т.д.)",
        }.get(llm_type, "обычных LLM (языковых моделей)")

        # Базовый промпт для анализа контента
        base_prompt = f"""
        Ты эксперт по оптимизации контента для LLM-моделей и поисковых систем на базе LLM.
        
        Проанализируй представленный контент с точки зрения следующих факторов,
        важных для оптимизации под {llm_type_desc}:
        
        {factors_list}
        
        Оцени каждый фактор по шкале от 1 до 10, где:
        - 1: фактор полностью отсутствует или очень слабо выражен
        - 10: фактор отлично реализован в контенте
        
        Для каждого фактора дай подробный анализ и конкретные рекомендации по улучшению.
        
        Также дай общую оценку контента для оптимизации под LLM по шкале от 1 до 10.
        
        Сформируй результат в JSON формате со следующей структурой:
        {{
            "factor_scores": {{
                "Фактор 1": 0,
                "Фактор 2": 0,
                ...
            }},
            "factor_analysis": {{
                "Фактор 1": "Анализ фактора 1 в контенте",
                "Фактор 2": "Анализ фактора 2 в контенте",
                ...
            }},
            "recommendations": {{
                "Фактор 1": ["Рекомендация 1", "Рекомендация 2"],
                "Фактор 2": ["Рекомендация 1", "Рекомендация 2"],
                ...
            }},
            "overall_score": 0,
            "summary": "Общее резюме анализа"
        }}
        """

        # Формируем финальный промпт
        final_prompt = base_prompt + "\n\nТекст для анализа:\n" + content

        return final_prompt

    def _process_importance_result(
        self, result: Dict[str, Any], factors: List[str], llm_type: str, provider: str, model: str
    ) -> Dict[str, Any]:
        """
        Обрабатывает результат анализа важности факторов.

        Args:
            result: Результат анализа от LLM
            factors: Список факторов для анализа
            llm_type: Тип LLM
            provider: Провайдер LLM
            model: Модель LLM

        Returns:
            Dict[str, Any]: Обработанный результат анализа
        """
        # Извлекаем текст ответа
        text = result.get("text", "")

        # Пытаемся извлечь JSON из ответа
        importance_data = parse_json_response(text)

        # Если не удалось извлечь JSON, пытаемся структурировать ответ сами
        if not importance_data or "factor_importance" not in importance_data:
            importance_data = {
                "factor_importance": {},
                "factor_analysis": {},
                "top_factors": [],
                "bottom_factors": [],
                "summary": "",
            }

            # Извлекаем оценки важности факторов
            for factor in factors:
                pattern = f"{re.escape(factor)}.*?(\d+)[^а-я0-9]*"
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    importance_data["factor_importance"][factor] = int(match.group(1))

            # Продолжение файла feature_importance_analyzer.py
            # Пытаемся извлечь анализ факторов
            sections = re.split(r"\n\s*#{1,3}\s+", text)

            for section in sections:
                lines = section.strip().split("\n")
                if not lines:
                    continue

                section_title = lines[0].strip().lower()
                section_content = "\n".join(lines[1:]).strip()

                # Ищем упоминания факторов в названии секции
                for factor in factors:
                    if factor.lower() in section_title.lower():
                        importance_data["factor_analysis"][factor] = section_content
                        break

            # Извлекаем топ факторы
            top_section = re.search(
                r"(?:топ|лучш|важн).*?факторы?.*?:(.*?)(?:\n\n|\n#|$)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if top_section:
                top_factors = re.findall(r"[*-]\s*(.+?)(?:\n|$)", top_section.group(1))
                importance_data["top_factors"] = top_factors
            else:
                # Если секции нет, формируем топ факторы из оценок
                sorted_factors = sorted(
                    importance_data["factor_importance"].items(), key=lambda x: x[1], reverse=True
                )
                importance_data["top_factors"] = [factor for factor, _ in sorted_factors[:3]]

            # Извлекаем наименее важные факторы
            bottom_section = re.search(
                r"(?:наимен|наихуд|менее).*?факторы?.*?:(.*?)(?:\n\n|\n#|$)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if bottom_section:
                bottom_factors = re.findall(r"[*-]\s*(.+?)(?:\n|$)", bottom_section.group(1))
                importance_data["bottom_factors"] = bottom_factors
            else:
                # Если секции нет, формируем наименее важные факторы из оценок
                sorted_factors = sorted(
                    importance_data["factor_importance"].items(), key=lambda x: x[1]
                )
                importance_data["bottom_factors"] = [factor for factor, _ in sorted_factors[:3]]

            # Извлекаем общее резюме
            summary_section = re.search(
                r"(?:резюме|вывод|заключ).*?:(.*?)(?:\n\n|\n#|$)", text, re.IGNORECASE | re.DOTALL
            )
            if summary_section:
                importance_data["summary"] = summary_section.group(1).strip()
            else:
                # Если резюме не найдено, используем первый абзац
                paragraphs = text.split("\n\n")
                if paragraphs:
                    importance_data["summary"] = paragraphs[0].strip()

            # Добавляем исходный текст ответа
            importance_data["raw_text"] = text

        # Формируем итоговый результат
        analyzed_result = {
            "factor_importance": importance_data.get("factor_importance", {}),
            "factor_analysis": importance_data.get("factor_analysis", {}),
            "top_factors": importance_data.get("top_factors", []),
            "bottom_factors": importance_data.get("bottom_factors", []),
            "summary": importance_data.get("summary", ""),
            "llm_type": llm_type,
            "provider": provider,
            "model": model,
            "tokens": result.get("tokens", {}),
            "cost": result.get("cost", 0),
        }

        return analyzed_result

    def _process_content_analysis_result(
        self,
        result: Dict[str, Any],
        factors: List[str],
        content: str,
        llm_type: str,
        provider: str,
        model: str,
    ) -> Dict[str, Any]:
        """
        Обрабатывает результат анализа контента.

        Args:
            result: Результат анализа от LLM
            factors: Список факторов для анализа
            content: Анализируемый текст
            llm_type: Тип LLM
            provider: Провайдер LLM
            model: Модель LLM

        Returns:
            Dict[str, Any]: Обработанный результат анализа
        """
        # Извлекаем текст ответа
        text = result.get("text", "")

        # Пытаемся извлечь JSON из ответа
        analysis_data = parse_json_response(text)

        # Если не удалось извлечь JSON, пытаемся структурировать ответ сами
        if not analysis_data or "factor_scores" not in analysis_data:
            analysis_data = {
                "factor_scores": {},
                "factor_analysis": {},
                "recommendations": {},
                "overall_score": 0,
                "summary": "",
            }

            # Извлекаем оценки факторов
            for factor in factors:
                pattern = f"{re.escape(factor)}.*?(\d+)[^а-я0-9]*"
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    analysis_data["factor_scores"][factor] = int(match.group(1))

            # Извлекаем общую оценку
            overall_match = re.search(r"общ[а-я]+\s+оценка.*?(\d+)[^а-я0-9]*", text, re.IGNORECASE)
            if overall_match:
                analysis_data["overall_score"] = int(overall_match.group(1))

            # Пытаемся извлечь анализ факторов
            sections = re.split(r"\n\s*#{1,3}\s+", text)

            for section in sections:
                lines = section.strip().split("\n")
                if not lines:
                    continue

                section_title = lines[0].strip().lower()
                section_content = "\n".join(lines[1:]).strip()

                # Ищем упоминания факторов в названии секции
                for factor in factors:
                    if factor.lower() in section_title.lower():
                        analysis_data["factor_analysis"][factor] = section_content

                        # Извлекаем рекомендации
                        recommendations = re.findall(r"[*-]\s*(.+?)(?:\n|$)", section_content)
                        if recommendations:
                            analysis_data["recommendations"][factor] = recommendations

                        break

            # Извлекаем общее резюме
            summary_section = re.search(
                r"(?:резюме|вывод|заключ).*?:(.*?)(?:\n\n|\n#|$)", text, re.IGNORECASE | re.DOTALL
            )
            if summary_section:
                analysis_data["summary"] = summary_section.group(1).strip()
            else:
                # Если резюме не найдено, используем первый абзац
                paragraphs = text.split("\n\n")
                if paragraphs:
                    analysis_data["summary"] = paragraphs[0].strip()

            # Добавляем исходный текст ответа
            analysis_data["raw_text"] = text

        # Формируем итоговый результат
        analyzed_result = {
            "factor_scores": analysis_data.get("factor_scores", {}),
            "factor_analysis": analysis_data.get("factor_analysis", {}),
            "recommendations": analysis_data.get("recommendations", {}),
            "overall_score": analysis_data.get("overall_score", 0),
            "summary": analysis_data.get("summary", ""),
            "llm_type": llm_type,
            "content_length": len(content),
            "provider": provider,
            "model": model,
            "tokens": result.get("tokens", {}),
            "cost": result.get("cost", 0),
        }

        return analyzed_result
