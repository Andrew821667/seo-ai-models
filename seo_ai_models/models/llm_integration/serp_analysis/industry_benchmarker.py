"""
Бенчмаркинг по отраслям для LLM-оптимизации.

Модуль предоставляет функционал для определения средних показателей цитируемости
по отраслям, установки бенчмарков для разных типов контента и отраслей,
анализа отраслевых трендов и предоставления рекомендаций.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from .llm_serp_analyzer import LLMSerpAnalyzer
from .citation_analyzer import CitationAnalyzer
from ..common.utils import parse_json_response


class IndustryBenchmarker:
    """
    Бенчмаркинг по отраслям для LLM-оптимизации.
    """

    def __init__(self, llm_service: LLMService, prompt_generator: PromptGenerator):
        """
        Инициализирует бенчмаркер по отраслям.

        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM
            prompt_generator: Экземпляр PromptGenerator для генерации промптов
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        self.serp_analyzer = LLMSerpAnalyzer(llm_service, prompt_generator)
        self.citation_analyzer = CitationAnalyzer(llm_service, prompt_generator)

        # Настройка логгирования
        self.logger = logging.getLogger(__name__)

        # Инициализация хранилища данных бенчмарков
        self.industry_benchmarks = {}

        # Определение поддерживаемых отраслей
        self.supported_industries = [
            "technology",
            "finance",
            "healthcare",
            "education",
            "ecommerce",
            "travel",
            "food",
            "fitness",
            "beauty",
            "real_estate",
            "legal",
            "automotive",
            "entertainment",
            "media",
            "manufacturing",
            "b2b",
            "other",
        ]

        # Определение поддерживаемых типов контента
        self.supported_content_types = [
            "article",
            "blog_post",
            "product_description",
            "how_to_guide",
            "news",
            "review",
            "comparison",
            "case_study",
            "faq",
            "landing_page",
            "other",
        ]

        # Инициализация базовых бенчмарков
        self._initialize_baseline_benchmarks()

    def _initialize_baseline_benchmarks(self) -> None:
        """
        Инициализирует базовые бенчмарки на основе экспертных знаний.
        """
        # Базовые бенчмарки для всех отраслей
        base_benchmarks = {
            "citation_rate": 0.25,  # 25% шанс цитирования
            "visibility_score": 0.3,  # 30% видимость
            "content_metrics": {
                "min_content_length": 800,  # Минимальная длина контента
                "optimal_content_length": 1500,  # Оптимальная длина контента
                "min_headers_count": 3,  # Минимальное количество заголовков
                "optimal_headers_count": 5,  # Оптимальное количество заголовков
                "optimal_keyword_density": 0.02,  # Оптимальная плотность ключевых слов
            },
        }

        # Инициализируем бенчмарки для каждой отрасли
        for industry in self.supported_industries:
            # Копируем базовые бенчмарки и корректируем их для конкретной отрасли
            industry_benchmarks = dict(base_benchmarks)

            # Корректируем бенчмарки в зависимости от отрасли
            if industry == "technology":
                industry_benchmarks["citation_rate"] = 0.30
                industry_benchmarks["visibility_score"] = 0.35
                industry_benchmarks["content_metrics"]["optimal_content_length"] = 1800
            elif industry == "healthcare":
                industry_benchmarks["citation_rate"] = 0.35
                industry_benchmarks["visibility_score"] = 0.40
                industry_benchmarks["content_metrics"]["optimal_content_length"] = 2000
            elif industry == "finance":
                industry_benchmarks["citation_rate"] = 0.28
                industry_benchmarks["visibility_score"] = 0.33
                industry_benchmarks["content_metrics"]["optimal_content_length"] = 1600

            # Инициализируем бенчмарки для каждого типа контента в отрасли
            content_type_benchmarks = {}

            for content_type in self.supported_content_types:
                # Копируем отраслевые бенчмарки и корректируем их для конкретного типа контента
                type_benchmarks = dict(industry_benchmarks)

                # Корректируем бенчмарки в зависимости от типа контента
                if content_type == "how_to_guide":
                    type_benchmarks["citation_rate"] += 0.05
                    type_benchmarks["visibility_score"] += 0.05
                    type_benchmarks["content_metrics"]["optimal_content_length"] += 300
                elif content_type == "product_description":
                    type_benchmarks["citation_rate"] -= 0.05
                    type_benchmarks["visibility_score"] -= 0.05
                    type_benchmarks["content_metrics"]["optimal_content_length"] -= 300

                # Добавляем бенчмарки для типа контента
                content_type_benchmarks[content_type] = type_benchmarks

            # Добавляем бенчмарки для отрасли
            self.industry_benchmarks[industry] = {
                "overall": industry_benchmarks,
                "content_types": content_type_benchmarks,
                "last_updated": datetime.now().isoformat(),
            }

    def get_industry_benchmarks(
        self, industry: str, content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Получает бенчмарки для указанной отрасли и типа контента.

        Args:
            industry: Отрасль
            content_type: Тип контента (опционально)

        Returns:
            Dict[str, Any]: Бенчмарки для отрасли и типа контента
        """
        # Проверяем поддержку отрасли
        if industry not in self.supported_industries:
            industry = "other"

        # Получаем бенчмарки для отрасли
        industry_data = self.industry_benchmarks.get(industry, {})

        # Если тип контента не указан, возвращаем общие бенчмарки для отрасли
        if content_type is None:
            return {
                "industry": industry,
                "content_type": None,
                "benchmarks": industry_data.get("overall", {}),
                "last_updated": industry_data.get("last_updated", ""),
            }

        # Проверяем поддержку типа контента
        if content_type not in self.supported_content_types:
            content_type = "other"

        # Получаем бенчмарки для типа контента
        content_type_data = industry_data.get("content_types", {}).get(content_type, {})

        # Возвращаем бенчмарки для отрасли и типа контента
        return {
            "industry": industry,
            "content_type": content_type,
            "benchmarks": content_type_data,
            "last_updated": industry_data.get("last_updated", ""),
        }

    def analyze_industry_benchmarks(
        self,
        content: str,
        queries: List[str],
        industry: str,
        content_type: str,
        llm_engines: Optional[List[str]] = None,
        num_samples: int = 2,
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Анализирует контент относительно бенчмарков отрасли и типа контента.

        Args:
            content: Контент для анализа
            queries: Список поисковых запросов
            industry: Отрасль
            content_type: Тип контента
            llm_engines: Список LLM-поисковиков для анализа
            num_samples: Количество запросов для каждого поисковика
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            Dict[str, Any]: Результаты анализа относительно бенчмарков
        """
        # Проверяем поддержку отрасли и типа контента
        if industry not in self.supported_industries:
            industry = "other"

        if content_type not in self.supported_content_types:
            content_type = "other"

        # Получаем бенчмарки
        benchmarks = self.get_industry_benchmarks(industry, content_type).get("benchmarks", {})

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
                    "benchmark_comparison": {
                        "citation_rate_diff": serp_result.get("citation_rate", 0)
                        - benchmarks.get("citation_rate", 0),
                        "visibility_score_diff": serp_result.get("visibility_score", 0)
                        - benchmarks.get("visibility_score", 0),
                    },
                }
            )

            # Обновляем статистику
            total_tokens += serp_result.get("tokens", {}).get("total", 0)
            total_cost += serp_result.get("cost", 0)

        # Оцениваем контент
        content_metrics = self._analyze_content_metrics(content)

        # Сравниваем метрики контента с бенчмарками
        benchmark_content_metrics = benchmarks.get("content_metrics", {})

        content_metrics_comparison = {}
        for metric, value in content_metrics.items():
            benchmark_value = benchmark_content_metrics.get(f"optimal_{metric}", 0)
            if benchmark_value > 0:
                content_metrics_comparison[f"{metric}_vs_benchmark"] = value / benchmark_value
                content_metrics_comparison[f"{metric}_diff"] = value - benchmark_value

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

        # Сравниваем с бенчмарками
        citation_rate_vs_benchmark = (
            avg_citation_rate / benchmarks.get("citation_rate", 1)
            if benchmarks.get("citation_rate", 0) > 0
            else 0
        )
        visibility_vs_benchmark = (
            avg_visibility_score / benchmarks.get("visibility_score", 1)
            if benchmarks.get("visibility_score", 0) > 0
            else 0
        )

        # Анализируем показатели относительно бенчмарков
        benchmark_analysis = self._generate_benchmark_analysis(
            avg_citation_rate, avg_visibility_score, content_metrics, benchmarks
        )

        # Формируем рекомендации
        recommendations = self._generate_benchmark_recommendations(
            avg_citation_rate,
            avg_visibility_score,
            content_metrics,
            benchmarks,
            industry,
            content_type,
        )

        # Формируем итоговый результат
        result = {
            "industry": industry,
            "content_type": content_type,
            "queries_count": len(queries),
            "engines": llm_engines,
            "samples_per_engine": num_samples,
            "citation_rate": avg_citation_rate,
            "visibility_score": avg_visibility_score,
            "citation_rate_vs_benchmark": citation_rate_vs_benchmark,
            "visibility_vs_benchmark": visibility_vs_benchmark,
            "content_metrics": content_metrics,
            "content_metrics_comparison": content_metrics_comparison,
            "benchmarks": benchmarks,
            "benchmark_analysis": benchmark_analysis,
            "recommendations": recommendations,
            "queries_results": query_results,
            "tokens": {"total": total_tokens},
            "cost": total_cost,
            "timestamp": datetime.now().isoformat(),
        }

        # Обновляем бенчмарки (если это нужно в будущем)
        # self._update_benchmarks(industry, content_type, avg_citation_rate, avg_visibility_score, content_metrics)

        return result

    def _analyze_content_metrics(self, content: str) -> Dict[str, Any]:
        """
        Анализирует метрики контента.

        Args:
            content: Контент для анализа

        Returns:
            Dict[str, Any]: Метрики контента
        """
        # Анализируем длину контента
        content_length = len(content)

        # Анализируем структуру контента
        headers_count = len(re.findall(r"\n#{1,3}\s+", content))
        paragraphs_count = len(re.split(r"\n\s*\n", content))
        sentence_count = len(re.split(r"[.!?]", content))

        # Вычисляем среднюю длину предложения
        avg_sentence_length = content_length / sentence_count if sentence_count > 0 else 0

        # Вычисляем плотность ключевых слов (упрощенно)
        total_words = len(re.findall(r"\b\w+\b", content.lower()))

        # Формируем метрики контента
        metrics = {
            "content_length": content_length,
            "headers_count": headers_count,
            "paragraphs_count": paragraphs_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "total_words": total_words,
        }

        return metrics

    def _generate_benchmark_analysis(
        self,
        citation_rate: float,
        visibility_score: float,
        content_metrics: Dict[str, Any],
        benchmarks: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Генерирует анализ показателей относительно бенчмарков.

        Args:
            citation_rate: Частота цитирования
            visibility_score: Оценка видимости
            content_metrics: Метрики контента
            benchmarks: Бенчмарки для сравнения

        Returns:
            Dict[str, Any]: Анализ показателей
        """
        # Анализируем частоту цитирования
        benchmark_citation_rate = benchmarks.get("citation_rate", 0)
        citation_rate_diff = citation_rate - benchmark_citation_rate

        if citation_rate_diff >= 0.05:
            citation_analysis = (
                "Частота цитирования значительно выше бенчмарка. Контент хорошо цитируется."
            )
        elif citation_rate_diff >= 0:
            citation_analysis = "Частота цитирования соответствует или немного выше бенчмарка. Контент цитируется удовлетворительно."
        elif citation_rate_diff >= -0.05:
            citation_analysis = (
                "Частота цитирования немного ниже бенчмарка. Требуются небольшие улучшения."
            )
        else:
            citation_analysis = "Частота цитирования значительно ниже бенчмарка. Требуется существенная оптимизация."

        # Анализируем видимость
        benchmark_visibility = benchmarks.get("visibility_score", 0)
        visibility_diff = visibility_score - benchmark_visibility

        if visibility_diff >= 0.05:
            visibility_analysis = "Видимость значительно выше бенчмарка. Контент хорошо виден в результатах LLM-поисковиков."
        elif visibility_diff >= 0:
            visibility_analysis = "Видимость соответствует или немного выше бенчмарка. Контент достаточно хорошо виден."
        elif visibility_diff >= -0.05:
            visibility_analysis = "Видимость немного ниже бенчмарка. Требуются небольшие улучшения."
        else:
            visibility_analysis = (
                "Видимость значительно ниже бенчмарка. Требуется существенная оптимизация."
            )

        # Анализируем длину контента
        benchmark_content_length = benchmarks.get("content_metrics", {}).get(
            "optimal_content_length", 0
        )
        content_length = content_metrics.get("content_length", 0)

        if content_length < benchmarks.get("content_metrics", {}).get("min_content_length", 0):
            length_analysis = "Длина контента ниже минимально рекомендуемой. Необходимо добавить больше информации."
        elif content_length < benchmark_content_length * 0.8:
            length_analysis = "Длина контента ниже оптимальной. Рекомендуется расширить содержание."
        elif content_length <= benchmark_content_length * 1.2:
            length_analysis = "Длина контента близка к оптимальной."
        else:
            length_analysis = "Длина контента превышает оптимальную. Возможно, контент слишком длинный или неструктурированный."

        # Анализируем структуру контента
        benchmark_headers_count = benchmarks.get("content_metrics", {}).get(
            "optimal_headers_count", 0
        )
        headers_count = content_metrics.get("headers_count", 0)

        if headers_count < benchmarks.get("content_metrics", {}).get("min_headers_count", 0):
            headers_analysis = "Количество заголовков ниже минимально рекомендуемого. Необходимо добавить структуру."
        elif headers_count < benchmark_headers_count * 0.8:
            headers_analysis = (
                "Количество заголовков ниже оптимального. Рекомендуется улучшить структуру."
            )
        elif headers_count <= benchmark_headers_count * 1.2:
            headers_analysis = "Количество заголовков близко к оптимальному."
        else:
            headers_analysis = "Количество заголовков превышает оптимальное. Возможно, структура слишком фрагментирована."

        # Формируем итоговый анализ
        overall_analysis = ""
        if citation_rate_diff >= 0 and visibility_diff >= 0:
            overall_analysis = (
                "Контент соответствует или превосходит отраслевые бенчмарки для LLM-поисковиков."
            )
        elif citation_rate_diff >= -0.05 and visibility_diff >= -0.05:
            overall_analysis = (
                "Контент близок к отраслевым бенчмаркам, но требует небольших улучшений."
            )
        else:
            overall_analysis = "Контент не соответствует отраслевым бенчмаркам. Требуется существенная оптимизация."

        # Формируем итоговый анализ
        analysis = {
            "overall": overall_analysis,
            "citation_rate": citation_analysis,
            "visibility_score": visibility_analysis,
            "content_length": length_analysis,
            "headers_count": headers_analysis,
        }

        return analysis

    def _generate_benchmark_recommendations(
        self,
        citation_rate: float,
        visibility_score: float,
        content_metrics: Dict[str, Any],
        benchmarks: Dict[str, Any],
        industry: str,
        content_type: str,
    ) -> List[str]:
        """
        Генерирует рекомендации на основе сравнения с бенчмарками.

        Args:
            citation_rate: Частота цитирования
            visibility_score: Оценка видимости
            content_metrics: Метрики контента
            benchmarks: Бенчмарки для сравнения
            industry: Отрасль
            content_type: Тип контента

        Returns:
            List[str]: Список рекомендаций
        """
        recommendations = []

        # Рекомендации по частоте цитирования
        benchmark_citation_rate = benchmarks.get("citation_rate", 0)
        citation_rate_diff = citation_rate - benchmark_citation_rate

        if citation_rate_diff < -0.05:
            recommendations.append(
                "Улучшите цитируемость контента, добавив больше уникальной и полезной информации."
            )
            recommendations.append(
                "Добавьте фактические данные, статистику и результаты исследований для повышения авторитетности."
            )

        # Рекомендации по видимости
        benchmark_visibility = benchmarks.get("visibility_score", 0)
        visibility_diff = visibility_score - benchmark_visibility

        if visibility_diff < -0.05:
            recommendations.append(
                "Повысьте видимость контента, улучшив его структуру и релевантность."
            )
            recommendations.append(
                "Добавьте больше конкретных примеров и практических рекомендаций."
            )

        # Рекомендации по длине контента
        benchmark_content_length = benchmarks.get("content_metrics", {}).get(
            "optimal_content_length", 0
        )
        content_length = content_metrics.get("content_length", 0)

        if content_length < benchmarks.get("content_metrics", {}).get("min_content_length", 0):
            recommendations.append(
                f"Увеличьте длину контента минимум до {benchmarks.get('content_metrics', {}).get('min_content_length', 0)} символов."
            )
        elif content_length < benchmark_content_length * 0.8:
            recommendations.append(
                f"Расширьте содержание примерно до {benchmark_content_length} символов, добавив больше полезной информации."
            )
        elif content_length > benchmark_content_length * 1.2:
            recommendations.append(
                f"Сократите и структурируйте контент, оставив самую важную информацию (примерно до {benchmark_content_length} символов)."
            )

        # Рекомендации по структуре контента
        benchmark_headers_count = benchmarks.get("content_metrics", {}).get(
            "optimal_headers_count", 0
        )
        headers_count = content_metrics.get("headers_count", 0)

        if headers_count < benchmarks.get("content_metrics", {}).get("min_headers_count", 0):
            recommendations.append(
                f"Добавьте минимум {benchmarks.get('content_metrics', {}).get('min_headers_count', 0)} заголовков для лучшей структуры."
            )
        elif headers_count < benchmark_headers_count * 0.8:
            recommendations.append(
                f"Улучшите структуру контента, добавив больше заголовков (примерно до {benchmark_headers_count})."
            )

        # Отраслевые рекомендации
        if industry == "technology":
            recommendations.append("Добавьте больше технических деталей и спецификаций.")
        elif industry == "healthcare":
            recommendations.append(
                "Убедитесь, что все медицинские утверждения подкреплены ссылками на надежные источники."
            )
        elif industry == "finance":
            recommendations.append("Добавьте конкретные примеры и сценарии использования.")

        # Рекомендации по типу контента
        if content_type == "how_to_guide":
            recommendations.append(
                "Разбейте инструкции на четкие пошаговые действия с пронумерованными списками."
            )
        elif content_type == "product_description":
            recommendations.append(
                "Добавьте больше информации о преимуществах и уникальных особенностях продукта."
            )
        elif content_type == "case_study":
            recommendations.append(
                "Включите реальные данные, результаты и выводы, демонстрирующие эффективность решения."
            )

        # Если рекомендаций мало, добавляем общие
        if len(recommendations) < 3:
            recommendations.append(
                "Регулярно отслеживайте показатели цитируемости и видимости для выявления трендов."
            )
            recommendations.append(
                "Используйте маркированные списки и таблицы для лучшей структуризации информации."
            )

        return recommendations

    def _update_benchmarks(
        self,
        industry: str,
        content_type: str,
        citation_rate: float,
        visibility_score: float,
        content_metrics: Dict[str, Any],
    ) -> None:
        """
        Обновляет бенчмарки на основе новых данных.

        Args:
            industry: Отрасль
            content_type: Тип контента
            citation_rate: Частота цитирования
            visibility_score: Оценка видимости
            content_metrics: Метрики контента
        """
        # В реальной реализации этот метод мог бы обновлять бенчмарки на основе новых данных,
        # например, вычислять скользящее среднее или применять другие методы аггрегации.
        # Для простоты примера, мы оставляем этот метод пустым.
        try:
            import time

            # Инициализируем кэш если его нет
            if not hasattr(self, "_benchmark_cache"):
                self._benchmark_cache = {}

            # Получаем ключ для кэша
            industry = content_metrics.get("industry", "general")
            content_type = content_metrics.get("content_type", "article")
            cache_key = industry + "_" + content_type

            # Создаем или обновляем запись в кэше
            if cache_key not in self._benchmark_cache:
                self._benchmark_cache[cache_key] = {
                    "metrics": [],
                    "last_updated": time.time(),
                    "sample_count": 0,
                }

            # Добавляем новые метрики
            self._benchmark_cache[cache_key]["metrics"].append(
                {
                    "timestamp": time.time(),
                    "word_count": content_metrics.get("word_count", 0),
                    "readability_score": content_metrics.get("readability_score", 0),
                    "eeat_score": content_metrics.get("eeat_score", 0),
                }
            )

            # Обновляем метаданные
            self._benchmark_cache[cache_key]["sample_count"] += 1
            self._benchmark_cache[cache_key]["last_updated"] = time.time()

            # Ограничиваем размер кэша (последние 500 записей)
            if len(self._benchmark_cache[cache_key]["metrics"]) > 500:
                self._benchmark_cache[cache_key]["metrics"] = self._benchmark_cache[cache_key][
                    "metrics"
                ][-500:]

            # Логируем обновление
            import logging

            logger = logging.getLogger(__name__)
            logger.debug("Benchmark cache updated for " + cache_key)

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error("Error updating benchmark cache: " + str(e))

    def get_supported_industries_and_types(self) -> Dict[str, List[str]]:
        """
        Возвращает список поддерживаемых отраслей и типов контента.

        Returns:
            Dict[str, List[str]]: Списки поддерживаемых отраслей и типов контента
        """
        return {
            "industries": self.supported_industries,
            "content_types": self.supported_content_types,
        }
