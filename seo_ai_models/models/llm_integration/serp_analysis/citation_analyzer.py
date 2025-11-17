"""
Анализатор цитирования в ответах LLM.

Модуль предоставляет функционал для анализа цитирования контента
в ответах LLM-моделей, отслеживания частоты цитирования
и определения факторов, влияющих на цитируемость.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from ..service.cost_estimator import CostEstimator
from ..analyzers.citability_scorer import CitabilityScorer
from .llm_serp_analyzer import LLMSerpAnalyzer
from ..common.utils import parse_json_response


class CitationAnalyzer:
    """
    Анализатор цитирования в ответах LLM.
    """

    def __init__(self, llm_service: LLMService, prompt_generator: PromptGenerator):
        """
        Инициализирует анализатор цитирования.

        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM
            prompt_generator: Экземпляр PromptGenerator для генерации промптов
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        self.cost_estimator = CostEstimator()
        self.citability_scorer = CitabilityScorer(llm_service, prompt_generator)
        self.serp_analyzer = LLMSerpAnalyzer(llm_service, prompt_generator)

        # Настройка логгирования
        self.logger = logging.getLogger(__name__)

        # Инициализация хранилища данных цитирования
        self.citation_history = {}

    def analyze_citation(
        self,
        content: str,
        queries: List[str],
        llm_engines: Optional[List[str]] = None,
        num_samples: int = 2,
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Анализирует цитирование контента в ответах LLM для заданных запросов.

        Args:
            content: Контент для анализа
            queries: Список поисковых запросов
            llm_engines: Список LLM-поисковиков для анализа
            num_samples: Количество запросов для каждого поисковика
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            Dict[str, Any]: Результаты анализа цитирования
        """
        # Распределяем бюджет между запросами
        query_budget = None
        if budget is not None:
            query_budget = budget / len(queries)

        # Анализируем цитирование для каждого запроса
        query_results = []
        total_tokens = 0
        total_cost = 0

        for i, query in enumerate(queries):
            self.logger.info(f"Анализ цитирования для запроса {i+1} из {len(queries)}: {query}")

            # Анализируем SERP для запроса
            serp_result = self.serp_analyzer.analyze_serp(
                query, content, llm_engines, num_samples, query_budget
            )

            # Добавляем результат запроса
            query_results.append(
                {
                    "query": query,
                    "citation_rate": serp_result.get("citation_rate", 0),
                    "visibility_score": serp_result.get("visibility_score", 0),
                    "engines_results": serp_result.get("engines_results", {}),
                }
            )

            # Обновляем статистику
            total_tokens += serp_result.get("tokens", {}).get("total", 0)
            total_cost += serp_result.get("cost", 0)

            # Сохраняем данные в историю цитирования
            self._save_citation_data(content, query, serp_result)

        # Оцениваем общую цитируемость контента
        citability_result = self.citability_scorer.score_citability(
            content, queries=queries, budget=budget * 0.2 if budget else None
        )

        # Обновляем статистику
        total_tokens += citability_result.get("tokens", {}).get("total", 0)
        total_cost += citability_result.get("cost", 0)

        # Анализируем факторы цитирования
        citation_factors = self._analyze_citation_factors(content, query_results)

        # Рассчитываем общую цитируемость
        avg_citation_rate = (
            sum(qr.get("citation_rate", 0) for qr in query_results) / len(query_results)
            if query_results
            else 0
        )
        avg_visibility_score = (
            sum(qr.get("visibility_score", 0) for qr in query_results) / len(query_results)
            if query_results
            else 0
        )

        # Формируем итоговый результат
        result = {
            "content_hash": hash(content) % 10000000,  # Упрощенный хэш для идентификации контента
            "queries_count": len(queries),
            "engines": llm_engines,
            "samples_per_engine": num_samples,
            "citation_rate": avg_citation_rate,
            "visibility_score": avg_visibility_score,
            "citability_score": citability_result.get("citability_score", 0),
            "citation_factors": citation_factors,
            "queries_results": query_results,
            "factor_scores": citability_result.get("factor_scores", {}),
            "suggested_improvements": citability_result.get("suggested_improvements", {}),
            "tokens": {"total": total_tokens},
            "cost": total_cost,
            "timestamp": datetime.now().isoformat(),
        }

        return result

    def _save_citation_data(self, content: str, query: str, serp_result: Dict[str, Any]) -> None:
        """
        Сохраняет данные о цитировании в истории.

        Args:
            content: Контент для анализа
            query: Поисковый запрос
            serp_result: Результат анализа SERP
        """
        # Создаем упрощенный хэш для идентификации контента
        content_hash = hash(content) % 10000000

        # Создаем запись в истории, если ее еще нет
        if content_hash not in self.citation_history:
            self.citation_history[content_hash] = {
                "content_length": len(content),
                "queries": {},
                "last_updated": datetime.now().isoformat(),
            }

        # Обновляем данные для запроса
        if query not in self.citation_history[content_hash]["queries"]:
            self.citation_history[content_hash]["queries"][query] = []

        # Добавляем новую запись
        self.citation_history[content_hash]["queries"][query].append(
            {
                "citation_rate": serp_result.get("citation_rate", 0),
                "visibility_score": serp_result.get("visibility_score", 0),
                "engines": serp_result.get("engines", []),
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Обновляем время последнего обновления
        self.citation_history[content_hash]["last_updated"] = datetime.now().isoformat()

    def _analyze_citation_factors(
        self, content: str, query_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Анализирует факторы, влияющие на цитирование.

        Args:
            content: Контент для анализа
            query_results: Результаты анализа цитирования для запросов

        Returns:
            Dict[str, Any]: Анализ факторов цитирования
        """
        # Анализируем корреляцию между различными факторами и цитируемостью

        # Анализ длины контента
        content_length = len(content)

        # Анализ структуры контента
        headers_count = len(re.findall(r"\n#{1,3}\s+", content))
        paragraphs_count = len(re.split(r"\n\s*\n", content))
        sentence_count = len(re.split(r"[.!?]", content))
        avg_sentence_length = len(content) / sentence_count if sentence_count > 0 else 0

        # Анализ ключевых слов
        query_keywords = set()
        for query_result in query_results:
            query = query_result.get("query", "")
            words = re.findall(r"\b\w{3,}\b", query.lower())
            query_keywords.update(words)

        # Частота ключевых слов в контенте
        keyword_frequency = {}
        for keyword in query_keywords:
            matches = re.findall(r"\b" + re.escape(keyword) + r"\b", content.lower())
            keyword_frequency[keyword] = len(matches)

        # Высчитываем общую плотность ключевых слов
        total_words = len(re.findall(r"\b\w+\b", content.lower()))
        keyword_density = sum(keyword_frequency.values()) / total_words if total_words > 0 else 0

        # Анализ заголовков
        headers = re.findall(r"\n(#{1,3}\s+.*?)(?:\n|$)", content)
        headers_with_keywords = 0

        for header in headers:
            header_text = re.sub(r"^#{1,3}\s+", "", header).lower()
            if any(keyword in header_text for keyword in query_keywords):
                headers_with_keywords += 1

        headers_with_keywords_ratio = headers_with_keywords / len(headers) if headers else 0

        # Формируем результат анализа факторов
        factors_analysis = {
            "content_structure": {
                "content_length": content_length,
                "headers_count": headers_count,
                "paragraphs_count": paragraphs_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": avg_sentence_length,
            },
            "keywords": {
                "query_keywords": list(query_keywords),
                "keyword_frequency": keyword_frequency,
                "keyword_density": keyword_density,
                "headers_with_keywords": headers_with_keywords,
                "headers_with_keywords_ratio": headers_with_keywords_ratio,
            },
            "citation_correlation": {
                "length_correlation": self._calculate_correlation(
                    content_length, [qr.get("citation_rate", 0) for qr in query_results]
                ),
                "keyword_density_correlation": self._calculate_correlation(
                    keyword_density, [qr.get("citation_rate", 0) for qr in query_results]
                ),
                "headers_correlation": self._calculate_correlation(
                    headers_count, [qr.get("citation_rate", 0) for qr in query_results]
                ),
            },
        }

        return factors_analysis

    def _calculate_correlation(self, value: float, rates: List[float]) -> float:
        """
        Рассчитывает простую корреляцию между значением и списком рейтингов.

        Args:
            value: Значение фактора
            rates: Список рейтингов цитирования

        Returns:
            float: Коэффициент корреляции
        """
        # Для простой корреляции используем нормализованное значение
        # В реальном проекте здесь было бы более сложное вычисление корреляции
        normalized_value = min(1.0, value / 10000.0)  # Произвольная нормализация

        avg_rate = sum(rates) / len(rates) if rates else 0

        # Простая оценка корреляции на основе сравнения нормализованного значения и среднего рейтинга
        # В реальном проекте использовался бы коэффициент корреляции Пирсона или Спирмена
        correlation = 1.0 - abs(normalized_value - avg_rate)

        return correlation

    def get_citation_history(
        self, content: str, time_period: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Получает историю цитирования контента.

        Args:
            content: Контент для анализа
            time_period: Период в днях для анализа истории (опционально)

        Returns:
            Dict[str, Any]: История цитирования
        """
        # Создаем упрощенный хэш для идентификации контента
        content_hash = hash(content) % 10000000

        # Проверяем наличие контента в истории
        if content_hash not in self.citation_history:
            return {
                "content_hash": content_hash,
                "content_length": len(content),
                "queries": {},
                "history_available": False,
                "message": "История цитирования для данного контента отсутствует",
            }

        # Получаем историю
        history = self.citation_history[content_hash]

        # Если указан период, фильтруем данные
        if time_period is not None:
            cutoff_date = datetime.now() - timedelta(days=time_period)
            cutoff_date_str = cutoff_date.isoformat()

            # Фильтруем данные по дате
            filtered_queries = {}

            for query, records in history["queries"].items():
                filtered_records = [
                    record for record in records if record.get("timestamp", "") >= cutoff_date_str
                ]

                if filtered_records:
                    filtered_queries[query] = filtered_records

            # Обновляем историю
            filtered_history = {
                "content_hash": content_hash,
                "content_length": history.get("content_length", len(content)),
                "queries": filtered_queries,
                "last_updated": history.get("last_updated", ""),
                "time_period": time_period,
                "history_available": bool(filtered_queries),
            }

            return filtered_history

        # Возвращаем полную историю
        return {
            "content_hash": content_hash,
            "content_length": history.get("content_length", len(content)),
            "queries": history.get("queries", {}),
            "last_updated": history.get("last_updated", ""),
            "history_available": True,
        }
