"""
Базовый генератор рекомендаций для микро-бизнеса.

Модуль предоставляет функциональность для генерации базовых
рекомендаций по SEO-оптимизации без использования LLM.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple

from seo_ai_models.common.utils.text_processing import (
    tokenize_text,
    extract_sentences,
    extract_paragraphs,
)


class BasicRecommender:
    """
    Базовый генератор рекомендаций для микро-бизнеса.

    Класс отвечает за генерацию базовых рекомендаций по
    SEO-оптимизации без использования LLM.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует генератор рекомендаций.

        Args:
            config: Конфигурация
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Параметры для рекомендаций
        self.min_word_count = self.config.get("min_word_count", 300)
        self.optimal_word_count = self.config.get("optimal_word_count", 800)
        self.min_keyword_density = self.config.get("min_keyword_density", 0.5)
        self.max_keyword_density = self.config.get("max_keyword_density", 2.5)
        self.min_headings = self.config.get("min_headings", 3)
        self.min_paragraphs = self.config.get("min_paragraphs", 5)
        self.min_internal_links = self.config.get("min_internal_links", 2)
        self.min_external_links = self.config.get("min_external_links", 1)
        self.min_readability = self.config.get("min_readability", 60)  # Flesch Reading Ease

        self.logger.info("BasicRecommender инициализирован")

    def generate_recommendations(
        self,
        metrics: Dict[str, Any],
        keywords_metrics: Optional[Dict[str, Any]] = None,
        structure_metrics: Optional[Dict[str, Any]] = None,
        readability_metrics: Optional[Dict[str, Any]] = None,
        content_type: str = "article",
        industry: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Генерирует рекомендации на основе метрик.

        Args:
            metrics: Базовые метрики контента
            keywords_metrics: Метрики ключевых слов
            structure_metrics: Метрики структуры
            readability_metrics: Метрики читабельности
            content_type: Тип контента
            industry: Отрасль
            **kwargs: Дополнительные параметры

        Returns:
            Рекомендации
        """
        self.logger.info("Начало генерации рекомендаций")

        recommendations = []

        # Корректируем параметры в зависимости от типа контента и отрасли
        self._adjust_parameters(content_type, industry)

        # Рекомендации по длине контента
        recommendations.extend(self._get_content_length_recommendations(metrics))

        # Рекомендации по ключевым словам
        if keywords_metrics:
            recommendations.extend(self._get_keyword_recommendations(keywords_metrics))

        # Рекомендации по структуре
        if structure_metrics:
            recommendations.extend(self._get_structure_recommendations(structure_metrics))

        # Рекомендации по читабельности
        if readability_metrics:
            recommendations.extend(self._get_readability_recommendations(readability_metrics))

        # Общие рекомендации
        recommendations.extend(self._get_general_recommendations(metrics))

        # Сортируем рекомендации по приоритету и категории
        sorted_recommendations = self._sort_recommendations(recommendations)

        # Группируем рекомендации по категориям
        categorized_recommendations = self._categorize_recommendations(sorted_recommendations)

        self.logger.info(f"Сгенерировано {len(recommendations)} рекомендаций")
        return {
            "recommendations": sorted_recommendations,
            "categorized_recommendations": categorized_recommendations,
            "total_recommendations": len(recommendations),
            "high_priority_count": sum(1 for r in recommendations if r.get("priority") == "high"),
            "medium_priority_count": sum(
                1 for r in recommendations if r.get("priority") == "medium"
            ),
            "low_priority_count": sum(1 for r in recommendations if r.get("priority") == "low"),
        }

    def _adjust_parameters(self, content_type: str, industry: Optional[str] = None) -> None:
        """
        Корректирует параметры в зависимости от типа контента и отрасли.

        Args:
            content_type: Тип контента
            industry: Отрасль
        """
        # Корректировка в зависимости от типа контента
        if content_type == "blog_post":
            self.optimal_word_count = 800
            self.min_headings = 4
        elif content_type == "product_page":
            self.optimal_word_count = 500
            self.min_headings = 2
        elif content_type == "landing_page":
            self.optimal_word_count = 400
            self.min_headings = 3
        elif content_type == "news":
            self.optimal_word_count = 600
            self.min_headings = 3

        # Корректировка в зависимости от отрасли
        if industry == "finance":
            self.optimal_word_count += 200
            self.min_readability = 55  # Более сложный текст допустим
        elif industry == "education":
            self.optimal_word_count += 100
        elif industry == "ecommerce":
            self.min_internal_links += 1

    def _get_content_length_recommendations(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Генерирует рекомендации по длине контента.

        Args:
            metrics: Базовые метрики контента

        Returns:
            Рекомендации по длине контента
        """
        recommendations = []

        word_count = metrics.get("word_count", 0)

        if word_count < self.min_word_count:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "content_length",
                    "title": "Увеличить длину контента",
                    "description": (
                        f"Текущая длина контента ({word_count} слов) меньше рекомендуемого "
                        f"минимума ({self.min_word_count} слов). Увеличьте объем текста, "
                        f"добавив больше ценной информации по теме."
                    ),
                    "current_value": word_count,
                    "recommended_value": self.min_word_count,
                }
            )
        elif word_count < self.optimal_word_count:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "content_length",
                    "title": "Расширить контент",
                    "description": (
                        f"Хотя текущая длина контента ({word_count} слов) превышает минимум, "
                        f"рекомендуется увеличить объем до оптимального значения "
                        f"({self.optimal_word_count} слов) для лучшего ранжирования."
                    ),
                    "current_value": word_count,
                    "recommended_value": self.optimal_word_count,
                }
            )

        return recommendations

    def _get_keyword_recommendations(
        self, keywords_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Генерирует рекомендации по ключевым словам.

        Args:
            keywords_metrics: Метрики ключевых слов

        Returns:
            Рекомендации по ключевым словам
        """
        recommendations = []

        keyword_metrics = keywords_metrics.get("keyword_metrics", {})

        for keyword, metrics in keyword_metrics.items():
            # Проверка плотности ключевых слов
            density = metrics.get("density", 0)

            if density < self.min_keyword_density:
                recommendations.append(
                    {
                        "priority": "high",
                        "category": "keywords",
                        "title": f'Увеличить плотность ключевого слова "{keyword}"',
                        "description": (
                            f'Текущая плотность ключевого слова "{keyword}" ({density}%) '
                            f"ниже рекомендуемого минимума ({self.min_keyword_density}%). "
                            f"Добавьте больше вхождений этого ключевого слова в текст."
                        ),
                        "current_value": density,
                        "recommended_value": self.min_keyword_density,
                    }
                )
            elif density > self.max_keyword_density:
                recommendations.append(
                    {
                        "priority": "medium",
                        "category": "keywords",
                        "title": f'Снизить плотность ключевого слова "{keyword}"',
                        "description": (
                            f'Текущая плотность ключевого слова "{keyword}" ({density}%) '
                            f"выше рекомендуемого максимума ({self.max_keyword_density}%). "
                            f"Это может восприниматься как переоптимизация. Уменьшите "
                            f"количество вхождений этого ключевого слова."
                        ),
                        "current_value": density,
                        "recommended_value": self.max_keyword_density,
                    }
                )

            # Проверка наличия в заголовках
            if not metrics.get("in_headings", False):
                recommendations.append(
                    {
                        "priority": "medium",
                        "category": "keywords",
                        "title": f'Добавить ключевое слово "{keyword}" в заголовки',
                        "description": (
                            f'Ключевое слово "{keyword}" отсутствует в заголовках. '
                            f"Добавьте его в заголовок H1 или подзаголовки для "
                            f"улучшения SEO-оптимизации."
                        ),
                        "current_value": False,
                        "recommended_value": True,
                    }
                )

            # Проверка наличия в первом абзаце
            if not metrics.get("in_first_paragraph", False):
                recommendations.append(
                    {
                        "priority": "medium",
                        "category": "keywords",
                        "title": f'Добавить ключевое слово "{keyword}" в первый абзац',
                        "description": (
                            f'Ключевое слово "{keyword}" отсутствует в первом абзаце. '
                            f"Добавьте его в начало текста для улучшения "
                            f"релевантности контента."
                        ),
                        "current_value": False,
                        "recommended_value": True,
                    }
                )

        return recommendations

    def _get_structure_recommendations(
        self, structure_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Генерирует рекомендации по структуре контента.

        Args:
            structure_metrics: Метрики структуры

        Returns:
            Рекомендации по структуре
        """
        recommendations = []

        # Проверка количества заголовков
        heading_count = structure_metrics.get("heading_count", 0)
        if heading_count < self.min_headings:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "structure",
                    "title": "Добавить больше заголовков",
                    "description": (
                        f"В тексте недостаточно заголовков ({heading_count}). "
                        f"Рекомендуется использовать не менее {self.min_headings} "
                        f"заголовков для лучшей структуризации контента."
                    ),
                    "current_value": heading_count,
                    "recommended_value": self.min_headings,
                }
            )

        # Проверка количества параграфов
        paragraph_count = structure_metrics.get("paragraph_count", 0)
        if paragraph_count < self.min_paragraphs:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "structure",
                    "title": "Увеличить количество параграфов",
                    "description": (
                        f"В тексте недостаточно параграфов ({paragraph_count}). "
                        f"Рекомендуется использовать не менее {self.min_paragraphs} "
                        f"параграфов для лучшей читабельности."
                    ),
                    "current_value": paragraph_count,
                    "recommended_value": self.min_paragraphs,
                }
            )

        # Проверка иерархии заголовков
        if not structure_metrics.get("heading_hierarchy", True):
            recommendations.append(
                {
                    "priority": "high",
                    "category": "structure",
                    "title": "Исправить иерархию заголовков",
                    "description": (
                        "В тексте нарушена иерархия заголовков. Убедитесь, что "
                        "заголовки следуют логической структуре: H1 -> H2 -> H3, "
                        "без пропуска уровней."
                    ),
                    "current_value": False,
                    "recommended_value": True,
                }
            )

        # Проверка наличия списков
        list_count = structure_metrics.get("list_count", 0)
        if list_count == 0:
            recommendations.append(
                {
                    "priority": "low",
                    "category": "structure",
                    "title": "Добавить маркированные или нумерованные списки",
                    "description": (
                        "В тексте отсутствуют списки. Добавьте маркированные или "
                        "нумерованные списки для лучшей структуризации и "
                        "читабельности контента."
                    ),
                    "current_value": 0,
                    "recommended_value": 1,
                }
            )

        # Проверка других проблем структуры
        for issue in structure_metrics.get("structure_issues", []):
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "structure",
                    "title": "Проблема структуры",
                    "description": issue,
                    "current_value": None,
                    "recommended_value": None,
                }
            )

        return recommendations

    def _get_readability_recommendations(
        self, readability_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Генерирует рекомендации по читабельности.

        Args:
            readability_metrics: Метрики читабельности

        Returns:
            Рекомендации по читабельности
        """
        recommendations = []

        # Проверка индекса читабельности
        flesch_reading_ease = readability_metrics.get("flesch_reading_ease", 0)
        if flesch_reading_ease < self.min_readability:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "readability",
                    "title": "Улучшить читабельность текста",
                    "description": (
                        f"Индекс читабельности текста ({flesch_reading_ease}) "
                        f"ниже рекомендуемого значения ({self.min_readability}). "
                        f"Используйте более короткие предложения и простые слова "
                        f"для улучшения читабельности."
                    ),
                    "current_value": flesch_reading_ease,
                    "recommended_value": self.min_readability,
                }
            )

        # Проверка средней длины предложения
        avg_sentence_length = readability_metrics.get("avg_sentence_length", 0)
        if avg_sentence_length > 20:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "readability",
                    "title": "Сократить длину предложений",
                    "description": (
                        f"Средняя длина предложения ({avg_sentence_length} слов) "
                        f"превышает оптимальное значение (15-20 слов). Разбейте "
                        f"длинные предложения на более короткие для улучшения "
                        f"читабельности."
                    ),
                    "current_value": avg_sentence_length,
                    "recommended_value": 15,
                }
            )

        # Проверка средней длины слова
        avg_word_length = readability_metrics.get("avg_word_length", 0)
        if avg_word_length > 6:
            recommendations.append(
                {
                    "priority": "low",
                    "category": "readability",
                    "title": "Использовать более простые слова",
                    "description": (
                        f"Средняя длина слова ({avg_word_length} символов) "
                        f"превышает оптимальное значение (4-6 символов). "
                        f"Используйте более простые и короткие слова для "
                        f"улучшения читабельности."
                    ),
                    "current_value": avg_word_length,
                    "recommended_value": 5,
                }
            )

        return recommendations

    def _get_general_recommendations(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Генерирует общие рекомендации.

        Args:
            metrics: Базовые метрики контента

        Returns:
            Общие рекомендации
        """
        recommendations = []

        # Проверка количества ссылок
        link_count = metrics.get("link_count", 0)
        if link_count < self.min_internal_links + self.min_external_links:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "links",
                    "title": "Увеличить количество ссылок",
                    "description": (
                        f"В тексте недостаточно ссылок ({link_count}). "
                        f"Рекомендуется добавить внутренние ({self.min_internal_links}) "
                        f"и внешние ({self.min_external_links}) ссылки для улучшения "
                        f"SEO-оптимизации и полезности контента."
                    ),
                    "current_value": link_count,
                    "recommended_value": self.min_internal_links + self.min_external_links,
                }
            )

        # Проверка количества изображений
        image_count = metrics.get("image_count", 0)
        if image_count == 0:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "multimedia",
                    "title": "Добавить изображения",
                    "description": (
                        "В тексте отсутствуют изображения. Добавьте релевантные "
                        "изображения с атрибутом ALT для улучшения "
                        "визуальной привлекательности и SEO."
                    ),
                    "current_value": 0,
                    "recommended_value": 1,
                }
            )

        # Рекомендация по мета-тегам (общая)
        recommendations.append(
            {
                "priority": "high",
                "category": "meta",
                "title": "Проверить мета-теги",
                "description": (
                    "Убедитесь, что страница имеет уникальный мета-заголовок "
                    "и мета-описание, включающие целевые ключевые слова "
                    "и соответствующие рекомендуемой длине (50-60 символов "
                    "для заголовка, 150-160 для описания)."
                ),
                "current_value": None,
                "recommended_value": None,
            }
        )

        return recommendations

    def _sort_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Сортирует рекомендации по приоритету и категории.

        Args:
            recommendations: Список рекомендаций

        Returns:
            Отсортированный список рекомендаций
        """
        # Определяем порядок приоритетов
        priority_order = {"high": 0, "medium": 1, "low": 2}

        # Определяем порядок категорий
        category_order = {
            "content_length": 0,
            "meta": 1,
            "keywords": 2,
            "structure": 3,
            "readability": 4,
            "links": 5,
            "multimedia": 6,
        }

        # Сортируем сначала по приоритету, затем по категории
        return sorted(
            recommendations,
            key=lambda r: (
                priority_order.get(r.get("priority"), 99),
                category_order.get(r.get("category"), 99),
            ),
        )

    def _categorize_recommendations(
        self, recommendations: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Группирует рекомендации по категориям.

        Args:
            recommendations: Список рекомендаций

        Returns:
            Рекомендации, сгруппированные по категориям
        """
        categorized = {}

        for recommendation in recommendations:
            category = recommendation.get("category", "other")

            if category not in categorized:
                categorized[category] = []

            categorized[category].append(recommendation)

        return categorized
