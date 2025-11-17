"""
Сверхлегкая версия SEO Advisor для микро-бизнеса.

Модуль предоставляет облегченную версию SEO Advisor для микро-бизнеса,
с минимальным использованием ресурсов и без зависимости от LLM.
"""

import logging
import re
import math
from typing import Dict, List, Any, Optional, Set, Tuple

# Импорты для статистических методов
import numpy as np

# Импорты из общих утилит
from seo_ai_models.common.utils.text_processing import (
    tokenize_text,
    extract_sentences,
    extract_paragraphs,
)


class MicroAdvisor:
    """
    Сверхлегкая версия SEO Advisor для микро-бизнеса.

    Класс реализует базовый анализ контента с использованием
    только статистических методов без зависимости от LLM.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует MicroAdvisor.

        Args:
            config: Конфигурация
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Базовые настройки
        self.min_word_count = self.config.get("min_word_count", 300)
        self.min_heading_count = self.config.get("min_heading_count", 3)
        self.min_paragraphs = self.config.get("min_paragraphs", 5)
        self.min_internal_links = self.config.get("min_internal_links", 2)
        self.min_external_links = self.config.get("min_external_links", 1)

        # Флаги для включения/отключения анализаторов
        self.enable_content_analysis = self.config.get("enable_content_analysis", True)
        self.enable_keyword_analysis = self.config.get("enable_keyword_analysis", True)
        self.enable_readability_analysis = self.config.get("enable_readability_analysis", True)
        self.enable_structure_analysis = self.config.get("enable_structure_analysis", True)

        self.logger.info("MicroAdvisor инициализирован")

    def analyze_content(
        self,
        content: str,
        keywords: Optional[List[str]] = None,
        url: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Анализирует контент с использованием только статистических методов.

        Args:
            content: Текст для анализа
            keywords: Ключевые слова
            url: URL страницы
            **kwargs: Дополнительные параметры

        Returns:
            Результаты анализа
        """
        self.logger.info("Начало анализа контента")

        results = {
            "success": True,
            "message": "Анализ выполнен успешно",
        }

        # Базовый анализ контента
        if self.enable_content_analysis:
            results["basic_metrics"] = self._analyze_basic_metrics(content)

        # Анализ ключевых слов
        if self.enable_keyword_analysis and keywords:
            results["keywords_basic"] = self._analyze_keywords(content, keywords)

        # Анализ читабельности
        if self.enable_readability_analysis:
            results["readability"] = self._analyze_readability(content)

        # Анализ структуры
        if self.enable_structure_analysis:
            results["structure_basic"] = self._analyze_structure(content)

        # Базовые рекомендации
        results["core_recommendations"] = self._generate_recommendations(
            content=content,
            keywords=keywords,
            url=url,
            metrics=results.get("basic_metrics", {}),
            structure=results.get("structure_basic", {}),
            keywords_metrics=results.get("keywords_basic", {}),
        )

        self.logger.info("Анализ контента завершен")
        return results

    def _analyze_basic_metrics(self, content: str) -> Dict[str, Any]:
        """
        Анализирует базовые метрики контента.

        Args:
            content: Текст для анализа

        Returns:
            Базовые метрики контента
        """
        words = tokenize_text(content)
        sentences = extract_sentences(content)
        paragraphs = extract_paragraphs(content)

        # Подсчет заголовков
        heading_pattern = r"<h[1-6][^>]*>.*?</h[1-6]>|^#{1,6}\s+.+$"
        headings = re.findall(heading_pattern, content, re.MULTILINE | re.IGNORECASE)

        # Подсчет ссылок
        link_pattern = r"<a\s+[^>]*href[^>]*>.*?</a>|\[.*?\]\(.*?\)"
        links = re.findall(link_pattern, content, re.IGNORECASE)

        # Подсчет изображений
        image_pattern = r"<img\s+[^>]*src[^>]*>|!\[.*?\]\(.*?\)"
        images = re.findall(image_pattern, content, re.IGNORECASE)

        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "heading_count": len(headings),
            "link_count": len(links),
            "image_count": len(images),
            "avg_words_per_sentence": len(words) / max(len(sentences), 1),
            "avg_sentences_per_paragraph": len(sentences) / max(len(paragraphs), 1),
            "content_length": len(content),
        }

    def _analyze_keywords(self, content: str, keywords: List[str]) -> Dict[str, Any]:
        """
        Анализирует использование ключевых слов в контенте.

        Args:
            content: Текст для анализа
            keywords: Ключевые слова

        Returns:
            Метрики ключевых слов
        """
        words = tokenize_text(content.lower())
        total_words = len(words)

        # Для каждого ключевого слова
        keyword_metrics = {}
        for keyword in keywords:
            keyword_lower = keyword.lower()
            keyword_count = content.lower().count(keyword_lower)

            # Вычисляем плотность
            density = (keyword_count * len(keyword_lower.split())) / max(total_words, 1) * 100

            # Проверка на наличие в заголовках
            heading_pattern = (
                r"<h[1-6][^>]*>.*?"
                + re.escape(keyword_lower)
                + r".*?</h[1-6]>|^#{1,6}\s+.*?"
                + re.escape(keyword_lower)
                + r".*?$"
            )
            in_headings = bool(
                re.search(heading_pattern, content.lower(), re.MULTILINE | re.IGNORECASE)
            )

            # Проверка на наличие в первом абзаце
            paragraphs = extract_paragraphs(content)
            in_first_paragraph = (
                paragraphs[0].lower().find(keyword_lower) >= 0 if paragraphs else False
            )

            # Метрики для ключевого слова
            keyword_metrics[keyword] = {
                "count": keyword_count,
                "density": round(density, 2),
                "in_headings": in_headings,
                "in_first_paragraph": in_first_paragraph,
                "prominence": self._calculate_prominence(content, keyword),
            }

        return {
            "keyword_metrics": keyword_metrics,
            "total_keywords": sum(m["count"] for m in keyword_metrics.values()),
            "avg_density": (
                np.mean([m["density"] for m in keyword_metrics.values()]) if keyword_metrics else 0
            ),
        }

    def _calculate_prominence(self, content: str, keyword: str) -> float:
        """
        Вычисляет выраженность ключевого слова в тексте.

        Args:
            content: Текст для анализа
            keyword: Ключевое слово

        Returns:
            Значение выраженности (0-1)
        """
        # Если ключевое слово отсутствует, возвращаем 0
        if keyword.lower() not in content.lower():
            return 0.0

        # Находим первое вхождение ключевого слова
        pos = content.lower().find(keyword.lower())
        content_length = len(content)

        # Выраженность зависит от позиции первого вхождения
        # (чем ближе к началу, тем выше выраженность)
        if pos == 0:
            return 1.0
        else:
            # Обратно пропорционально позиции, но не меньше 0.1
            return max(0.1, 1.0 - (pos / (content_length / 2)))

    def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """
        Анализирует читабельность текста.

        Args:
            content: Текст для анализа

        Returns:
            Метрики читабельности
        """
        words = tokenize_text(content)
        sentences = extract_sentences(content)

        # Если предложений нет или слов нет, устанавливаем базовые значения
        if not sentences or not words:
            return {
                "flesch_reading_ease": 0,
                "avg_word_length": 0,
                "avg_sentence_length": 0,
                "readability_score": "Не определено",
            }

        # Средняя длина слова в символах
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Средняя длина предложения в словах
        avg_sentence_length = len(words) / len(sentences)

        # Вычисляем Flesch Reading Ease
        # FRE = 206.835 - (1.015 * ASL) - (84.6 * ASW)
        # ASL = avg_sentence_length
        # ASW = avg_word_length в слогах (упрощенно: avg_word_length / 3)
        fre = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_word_length / 3))

        # Определяем оценку читабельности
        readability_score = "Сложный"
        if fre >= 90:
            readability_score = "Очень легкий"
        elif fre >= 80:
            readability_score = "Легкий"
        elif fre >= 70:
            readability_score = "Средний"
        elif fre >= 60:
            readability_score = "Умеренно сложный"
        elif fre >= 50:
            readability_score = "Сложный"

        return {
            "flesch_reading_ease": round(fre, 2),
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "readability_score": readability_score,
        }

    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """
        Анализирует структуру контента.

        Args:
            content: Текст для анализа

        Returns:
            Метрики структуры
        """
        # Находим все заголовки
        heading_pattern = r"<h([1-6])[^>]*>(.*?)</h\1>|^(#{1,6})\s+(.+)$"
        headings = []

        for match in re.finditer(heading_pattern, content, re.MULTILINE | re.IGNORECASE):
            if match.group(1):  # HTML заголовки
                level = int(match.group(1))
                text = match.group(2)
            else:  # Markdown заголовки
                level = len(match.group(3))
                text = match.group(4)

            headings.append(
                {
                    "level": level,
                    "text": text,
                    "position": match.start(),
                }
            )

        # Анализируем параграфы
        paragraphs = extract_paragraphs(content)

        # Анализируем маркированные списки
        list_pattern = r"<[ou]l[^>]*>.*?</[ou]l>|(?:^[-*+]\s+.+$)+"
        lists = re.findall(list_pattern, content, re.MULTILINE | re.IGNORECASE | re.DOTALL)

        # Анализируем плотность тегов разметки
        html_tags = re.findall(r"<[^>]+>", content)

        structure_metrics = {
            "heading_count": len(headings),
            "heading_levels": self._get_heading_levels(headings),
            "heading_hierarchy": self._check_heading_hierarchy(headings),
            "paragraph_count": len(paragraphs),
            "list_count": len(lists),
            "html_tag_count": len(html_tags),
            "has_structure_issues": False,
            "structure_issues": [],
        }

        # Проверка на проблемы структуры
        structure_issues = []

        if len(headings) < self.min_heading_count:
            structure_issues.append(f"Недостаточно заголовков (минимум {self.min_heading_count})")

        if len(paragraphs) < self.min_paragraphs:
            structure_issues.append(f"Недостаточно параграфов (минимум {self.min_paragraphs})")

        if not structure_metrics["heading_hierarchy"]:
            structure_issues.append("Нарушена иерархия заголовков")

        if structure_issues:
            structure_metrics["has_structure_issues"] = True
            structure_metrics["structure_issues"] = structure_issues

        return structure_metrics

    def _get_heading_levels(self, headings: List[Dict[str, Any]]) -> Dict[int, int]:
        """
        Получает распределение заголовков по уровням.

        Args:
            headings: Список заголовков

        Returns:
            Распределение заголовков по уровням
        """
        levels = {}
        for heading in headings:
            level = heading["level"]
            levels[level] = levels.get(level, 0) + 1

        return levels

    def _check_heading_hierarchy(self, headings: List[Dict[str, Any]]) -> bool:
        """
        Проверяет иерархию заголовков.

        Args:
            headings: Список заголовков

        Returns:
            True, если иерархия корректна
        """
        if not headings:
            return True

        # Проверка правильного порядка заголовков (не должно быть скачков более чем на 1)
        current_level = None
        for heading in headings:
            level = heading["level"]

            if current_level is None:
                current_level = level
            else:
                # Нельзя перескакивать уровни (например, с h1 на h3)
                if level > current_level and level > current_level + 1:
                    return False

                current_level = level

        return True

    def _generate_recommendations(
        self,
        content: str,
        keywords: Optional[List[str]] = None,
        url: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        structure: Optional[Dict[str, Any]] = None,
        keywords_metrics: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Генерирует базовые рекомендации по контенту.

        Args:
            content: Текст для анализа
            keywords: Ключевые слова
            url: URL страницы
            metrics: Метрики контента
            structure: Метрики структуры
            keywords_metrics: Метрики ключевых слов

        Returns:
            Список рекомендаций
        """
        recommendations = []

        # Если нет метрик, возвращаем базовые рекомендации
        if not metrics:
            return [
                {
                    "priority": "high",
                    "category": "content",
                    "recommendation": "Рекомендуется написать контент объемом не менее 300 слов",
                }
            ]

        # Рекомендации по длине контента
        word_count = metrics.get("word_count", 0)
        if word_count < self.min_word_count:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "content",
                    "recommendation": f"Увеличьте длину контента, текущий объем {word_count} слов меньше рекомендуемого минимума {self.min_word_count} слов",
                }
            )

        # Рекомендации по структуре
        if structure and structure.get("has_structure_issues"):
            for issue in structure.get("structure_issues", []):
                recommendations.append(
                    {
                        "priority": "medium",
                        "category": "structure",
                        "recommendation": f"Проблема структуры: {issue}",
                    }
                )

        # Рекомендации по ключевым словам
        if keywords and keywords_metrics:
            keyword_metrics = keywords_metrics.get("keyword_metrics", {})
            for keyword, metric in keyword_metrics.items():
                # Проверка плотности ключевых слов
                if metric["density"] < 0.5:
                    recommendations.append(
                        {
                            "priority": "medium",
                            "category": "keywords",
                            "recommendation": f'Увеличьте плотность ключевого слова "{keyword}" (текущая: {metric["density"]}%, рекомендуется: 0.5-2.5%)',
                        }
                    )
                elif metric["density"] > 2.5:
                    recommendations.append(
                        {
                            "priority": "medium",
                            "category": "keywords",
                            "recommendation": f'Снизьте плотность ключевого слова "{keyword}" (текущая: {metric["density"]}%, рекомендуется: 0.5-2.5%)',
                        }
                    )

                # Рекомендации по размещению в заголовках
                if not metric["in_headings"]:
                    recommendations.append(
                        {
                            "priority": "medium",
                            "category": "keywords",
                            "recommendation": f'Добавьте ключевое слово "{keyword}" в заголовки',
                        }
                    )

                # Рекомендации по размещению в первом абзаце
                if not metric["in_first_paragraph"]:
                    recommendations.append(
                        {
                            "priority": "medium",
                            "category": "keywords",
                            "recommendation": f'Добавьте ключевое слово "{keyword}" в первый абзац',
                        }
                    )

        # Сортируем рекомендации по приоритету
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order[x["priority"]])

        return recommendations
