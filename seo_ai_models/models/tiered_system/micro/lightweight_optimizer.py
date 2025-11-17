"""
Легковесный оптимизатор для микро-бизнеса.

Модуль предоставляет функциональность для легковесной оптимизации
контента с минимальными вычислительными ресурсами.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple

# Импорты из общих утилит
from seo_ai_models.common.utils.text_processing import (
    tokenize_text,
    extract_sentences,
    extract_paragraphs,
)


class LightweightOptimizer:
    """
    Легковесный оптимизатор для микро-бизнеса.

    Класс отвечает за легковесную оптимизацию контента
    с минимальными вычислительными ресурсами.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует легковесный оптимизатор.

        Args:
            config: Конфигурация
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Настройки оптимизатора
        self.optimal_keyword_density = self.config.get("optimal_keyword_density", 1.5)
        self.optimal_title_length = self.config.get("optimal_title_length", 55)
        self.optimal_description_length = self.config.get("optimal_description_length", 155)

        self.logger.info("LightweightOptimizer инициализирован")

    def optimize_content(
        self, content: str, keywords: List[str], recommendations: List[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """
        Оптимизирует контент на основе рекомендаций.

        Args:
            content: Исходный текст
            keywords: Ключевые слова
            recommendations: Рекомендации
            **kwargs: Дополнительные параметры

        Returns:
            Результаты оптимизации
        """
        self.logger.info("Начало оптимизации контента")

        optimized_content = content
        changes = []
        optimization_stats = {}

        # Сортируем рекомендации по приоритету и категории
        prioritized_recommendations = self._prioritize_recommendations(recommendations)

        # Применяем оптимизации в зависимости от рекомендаций
        # Оптимизация заголовков
        if any(r.get("category") == "structure" for r in prioritized_recommendations):
            optimized_content, heading_changes = self._optimize_headings(
                optimized_content, keywords
            )
            changes.extend(heading_changes)

        # Оптимизация ключевых слов
        if any(r.get("category") == "keywords" for r in prioritized_recommendations):
            optimized_content, keyword_changes = self._optimize_keywords(
                optimized_content, keywords
            )
            changes.extend(keyword_changes)

        # Оптимизация читабельности
        if any(r.get("category") == "readability" for r in prioritized_recommendations):
            optimized_content, readability_changes = self._optimize_readability(optimized_content)
            changes.extend(readability_changes)

        # Оптимизация мета-тегов
        meta_title, meta_description, meta_changes = self._optimize_meta_tags(content, keywords)
        changes.extend(meta_changes)

        # Вычисляем статистику оптимизации
        optimization_stats = self._calculate_optimization_stats(
            original_content=content, optimized_content=optimized_content, keywords=keywords
        )

        self.logger.info(f"Выполнено {len(changes)} оптимизаций")
        return {
            "optimized_content": optimized_content,
            "meta_title": meta_title,
            "meta_description": meta_description,
            "changes": changes,
            "optimization_stats": optimization_stats,
        }

    def _prioritize_recommendations(
        self, recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Приоритизирует рекомендации для оптимизации.

        Args:
            recommendations: Список рекомендаций

        Returns:
            Приоритизированный список рекомендаций
        """
        # Определяем порядок приоритетов
        priority_order = {"high": 0, "medium": 1, "low": 2}

        # Определяем порядок категорий для оптимизации
        category_order = {
            "meta": 0,
            "content_length": 1,
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

    def _optimize_headings(
        self, content: str, keywords: List[str]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Оптимизирует заголовки в тексте.

        Args:
            content: Исходный текст
            keywords: Ключевые слова

        Returns:
            Оптимизированный текст и список изменений
        """
        optimized_content = content
        changes = []

        # Находим все заголовки в тексте
        heading_pattern = r"<h([1-6])[^>]*>(.*?)</h\1>|^(#{1,6})\s+(.+)$"
        headings = []

        for match in re.finditer(heading_pattern, content, re.MULTILINE | re.IGNORECASE):
            if match.group(1):  # HTML заголовки
                level = int(match.group(1))
                text = match.group(2)
                start = match.start()
                end = match.end()
                full_text = match.group(0)
                is_html = True
            else:  # Markdown заголовки
                level = len(match.group(3))
                text = match.group(4)
                start = match.start()
                end = match.end()
                full_text = match.group(0)
                is_html = False

            headings.append(
                {
                    "level": level,
                    "text": text,
                    "start": start,
                    "end": end,
                    "full_text": full_text,
                    "is_html": is_html,
                }
            )

        # Проверяем заголовки на наличие ключевых слов
        headings_with_missing_keywords = []

        for heading in headings:
            # Проверяем, есть ли ключевые слова в заголовке
            has_keyword = any(keyword.lower() in heading["text"].lower() for keyword in keywords)

            if not has_keyword and heading["level"] <= 2:  # Только H1 и H2
                headings_with_missing_keywords.append(heading)

        # Оптимизируем заголовки с отсутствующими ключевыми словами
        offset = 0  # Смещение из-за изменений в длине текста

        for heading in headings_with_missing_keywords:
            # Выбираем ключевое слово, которое лучше всего подходит к заголовку
            best_keyword = self._find_best_keyword_for_heading(heading["text"], keywords)

            # Формируем новый текст заголовка
            if best_keyword:
                if heading["is_html"]:
                    original_text = heading["text"]
                    new_text = f"{original_text} - {best_keyword}"
                    new_heading = f"<h{heading['level']}>{new_text}</h{heading['level']}>"
                else:
                    original_text = heading["text"]
                    new_text = f"{original_text} - {best_keyword}"
                    new_heading = f"{'#' * heading['level']} {new_text}"

                # Заменяем заголовок в тексте
                start = heading["start"] + offset
                end = heading["end"] + offset

                optimized_content = (
                    optimized_content[:start] + new_heading + optimized_content[end:]
                )

                # Обновляем смещение
                offset += len(new_heading) - (end - start)

                # Добавляем изменение в список
                changes.append(
                    {
                        "type": "heading_optimization",
                        "description": f"Добавлено ключевое слово '{best_keyword}' в заголовок",
                        "original": heading["full_text"],
                        "optimized": new_heading,
                    }
                )

        return optimized_content, changes

    def _find_best_keyword_for_heading(
        self, heading_text: str, keywords: List[str]
    ) -> Optional[str]:
        """
        Находит наиболее подходящее ключевое слово для заголовка.

        Args:
            heading_text: Текст заголовка
            keywords: Ключевые слова

        Returns:
            Наиболее подходящее ключевое слово
        """
        if not keywords:
            return None

        # Проверяем, есть ли ключевые слова уже в заголовке
        for keyword in keywords:
            if keyword.lower() in heading_text.lower():
                return None  # Ключевое слово уже есть

        # Выбираем ключевое слово, которое лучше всего подходит к заголовку
        # (в данном случае, просто берем первое, но можно реализовать более сложную логику)
        return keywords[0]

    def _optimize_keywords(
        self, content: str, keywords: List[str]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Оптимизирует использование ключевых слов в тексте.

        Args:
            content: Исходный текст
            keywords: Ключевые слова

        Returns:
            Оптимизированный текст и список изменений
        """
        optimized_content = content
        changes = []

        if not keywords:
            return optimized_content, changes

        # Извлекаем параграфы
        paragraphs = extract_paragraphs(content)

        if not paragraphs:
            return optimized_content, changes

        # Проверяем, есть ли ключевые слова в первом параграфе
        first_paragraph = paragraphs[0]
        missing_keywords_in_first_paragraph = []

        for keyword in keywords:
            if keyword.lower() not in first_paragraph.lower():
                missing_keywords_in_first_paragraph.append(keyword)

        # Оптимизируем первый параграф, добавляя отсутствующие ключевые слова
        if missing_keywords_in_first_paragraph:
            # Выбираем наиболее подходящее ключевое слово
            best_keyword = missing_keywords_in_first_paragraph[0]

            # Добавляем ключевое слово в конец первого параграфа
            original_paragraph = first_paragraph

            # Проверяем, заканчивается ли параграф точкой
            if original_paragraph.strip().endswith("."):
                new_paragraph = (
                    original_paragraph.rstrip() + f" Это также относится к теме {best_keyword}."
                )
            else:
                new_paragraph = (
                    original_paragraph.rstrip() + f". Это также относится к теме {best_keyword}."
                )

            # Заменяем первый параграф в тексте
            optimized_content = optimized_content.replace(original_paragraph, new_paragraph, 1)

            # Добавляем изменение в список
            changes.append(
                {
                    "type": "first_paragraph_optimization",
                    "description": f"Добавлено ключевое слово '{best_keyword}' в первый абзац",
                    "original": original_paragraph,
                    "optimized": new_paragraph,
                }
            )

        # Вычисляем плотность ключевых слов
        words = tokenize_text(content.lower())
        total_words = len(words)

        for keyword in keywords:
            keyword_lower = keyword.lower()
            keyword_count = content.lower().count(keyword_lower)

            # Вычисляем плотность
            keyword_words = len(keyword_lower.split())
            density = (keyword_count * keyword_words) / max(total_words, 1) * 100

            # Если плотность слишком низкая, добавляем ключевое слово в текст
            if density < 0.5:
                # Находим последний параграф
                if len(paragraphs) > 1:
                    last_paragraph = paragraphs[-1]

                    # Добавляем ключевое слово в конец последнего параграфа
                    original_paragraph = last_paragraph

                    # Проверяем, заканчивается ли параграф точкой
                    if original_paragraph.strip().endswith("."):
                        new_paragraph = (
                            original_paragraph.rstrip() + f" Важно учитывать {keyword} при этом."
                        )
                    else:
                        new_paragraph = (
                            original_paragraph.rstrip() + f". Важно учитывать {keyword} при этом."
                        )

                    # Заменяем последний параграф в тексте
                    optimized_content = optimized_content.replace(
                        original_paragraph, new_paragraph, 1
                    )

                    # Добавляем изменение в список
                    changes.append(
                        {
                            "type": "keyword_density_optimization",
                            "description": f"Увеличена плотность ключевого слова '{keyword}'",
                            "original": original_paragraph,
                            "optimized": new_paragraph,
                        }
                    )

        return optimized_content, changes

    def _optimize_readability(self, content: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Оптимизирует читабельность текста.

        Args:
            content: Исходный текст

        Returns:
            Оптимизированный текст и список изменений
        """
        optimized_content = content
        changes = []

        # Извлекаем предложения
        sentences = extract_sentences(content)

        if not sentences:
            return optimized_content, changes

        # Находим слишком длинные предложения (более 30 слов)
        long_sentences = []

        for sentence in sentences:
            words_count = len(tokenize_text(sentence))

            if words_count > 30:
                long_sentences.append(sentence)

        # Оптимизируем длинные предложения
        for sentence in long_sentences[:3]:  # Оптимизируем только первые 3 длинных предложения
            # Примитивная оптимизация - разбиваем предложение на две части
            words = tokenize_text(sentence)
            middle_index = len(words) // 2

            # Находим подходящее место для разделения (ближайшее к середине)
            split_index = middle_index

            for i in range(middle_index, min(len(words), middle_index + 5)):
                if words[i].lower() in [
                    "и",
                    "или",
                    "но",
                    "однако",
                    "также",
                    "тем",
                    "этом",
                    "этой",
                    "этих",
                ]:
                    split_index = i
                    break

            # Формируем два новых предложения
            first_part = " ".join(words[:split_index])
            second_part = " ".join(words[split_index:])

            # Капитализируем первую букву второго предложения
            if second_part:
                second_part = second_part[0].upper() + second_part[1:]

            # Добавляем точку в конце первого предложения, если её нет
            if not first_part.endswith("."):
                first_part += "."

            # Формируем новый текст
            new_sentence = first_part + " " + second_part

            # Заменяем предложение в тексте
            optimized_content = optimized_content.replace(sentence, new_sentence, 1)

            # Добавляем изменение в список
            changes.append(
                {
                    "type": "readability_optimization",
                    "description": "Разбито длинное предложение для улучшения читабельности",
                    "original": sentence,
                    "optimized": new_sentence,
                }
            )

        return optimized_content, changes

    def _optimize_meta_tags(
        self, content: str, keywords: List[str]
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Оптимизирует мета-теги.

        Args:
            content: Исходный текст
            keywords: Ключевые слова

        Returns:
            Оптимизированный мета-заголовок, мета-описание и список изменений
        """
        changes = []

        # Извлекаем заголовки из текста
        heading_pattern = r"<h1[^>]*>(.*?)</h1>|^#\s+(.+)$"
        h1_match = re.search(heading_pattern, content, re.MULTILINE | re.IGNORECASE)

        h1_text = ""
        if h1_match:
            h1_text = h1_match.group(1) or h1_match.group(2) or ""

        # Генерируем мета-заголовок
        meta_title = self._generate_meta_title(h1_text, keywords)
        changes.append(
            {
                "type": "meta_title_optimization",
                "description": "Создан оптимизированный мета-заголовок",
                "original": h1_text,
                "optimized": meta_title,
            }
        )

        # Извлекаем первый параграф для мета-описания
        paragraphs = extract_paragraphs(content)
        first_paragraph = paragraphs[0] if paragraphs else ""

        # Генерируем мета-описание
        meta_description = self._generate_meta_description(first_paragraph, keywords)
        changes.append(
            {
                "type": "meta_description_optimization",
                "description": "Создано оптимизированное мета-описание",
                "original": (
                    first_paragraph[:100] + "..." if len(first_paragraph) > 100 else first_paragraph
                ),
                "optimized": meta_description,
            }
        )

        return meta_title, meta_description, changes

    def _generate_meta_title(self, h1_text: str, keywords: List[str]) -> str:
        """
        Генерирует оптимизированный мета-заголовок.

        Args:
            h1_text: Текст заголовка H1
            keywords: Ключевые слова

        Returns:
            Оптимизированный мета-заголовок
        """
        # Используем h1_text в качестве основы для мета-заголовка
        meta_title = h1_text

        # Если заголовок пустой, формируем его из ключевых слов
        if not meta_title and keywords:
            meta_title = f"{keywords[0].capitalize()} - "
            if len(keywords) > 1:
                meta_title += f"{keywords[1]}"

        # Если заголовок слишком длинный, обрезаем его
        if len(meta_title) > self.optimal_title_length:
            meta_title = meta_title[: self.optimal_title_length] + "..."

        # Если заголовок не содержит главное ключевое слово, добавляем его
        if keywords and keywords[0].lower() not in meta_title.lower():
            # Если заголовок слишком длинный для добавления ключевого слова,
            # обрезаем его еще больше
            if len(meta_title) + len(keywords[0]) + 3 > self.optimal_title_length:
                meta_title = meta_title[: self.optimal_title_length - len(keywords[0]) - 3] + "..."

            meta_title = f"{meta_title} | {keywords[0]}"

        return meta_title

    def _generate_meta_description(self, first_paragraph: str, keywords: List[str]) -> str:
        """
        Генерирует оптимизированное мета-описание.

        Args:
            first_paragraph: Текст первого параграфа
            keywords: Ключевые слова

        Returns:
            Оптимизированное мета-описание
        """
        # Используем первый параграф в качестве основы для мета-описания
        meta_description = first_paragraph

        # Если параграф пустой, формируем описание из ключевых слов
        if not meta_description and keywords:
            keyword_phrase = ", ".join(keywords[:3])
            meta_description = f"Информация о {keyword_phrase}. Узнайте больше на нашем сайте."

        # Если описание слишком длинное, обрезаем его
        if len(meta_description) > self.optimal_description_length:
            meta_description = meta_description[: self.optimal_description_length] + "..."

        # Если описание не содержит главное ключевое слово, добавляем его
        if keywords and keywords[0].lower() not in meta_description.lower():
            # Обрезаем описание, чтобы добавить ключевое слово
            if len(meta_description) + len(keywords[0]) + 30 > self.optimal_description_length:
                meta_description = (
                    meta_description[: self.optimal_description_length - len(keywords[0]) - 30]
                    + "..."
                )

            meta_description = f"{meta_description} Узнайте больше о {keywords[0]}."

        return meta_description

    def _calculate_optimization_stats(
        self, original_content: str, optimized_content: str, keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Вычисляет статистику оптимизации.

        Args:
            original_content: Исходный текст
            optimized_content: Оптимизированный текст
            keywords: Ключевые слова

        Returns:
            Статистика оптимизации
        """
        # Статистика по длине
        original_length = len(original_content)
        optimized_length = len(optimized_content)
        length_change = optimized_length - original_length
        length_change_percent = (
            (length_change / original_length) * 100 if original_length > 0 else 0
        )

        # Статистика по ключевым словам
        original_keyword_count = sum(
            original_content.lower().count(keyword.lower()) for keyword in keywords
        )
        optimized_keyword_count = sum(
            optimized_content.lower().count(keyword.lower()) for keyword in keywords
        )
        keyword_change = optimized_keyword_count - original_keyword_count

        # Статистика по параграфам
        original_paragraphs = extract_paragraphs(original_content)
        optimized_paragraphs = extract_paragraphs(optimized_content)
        paragraph_change = len(optimized_paragraphs) - len(original_paragraphs)

        # Статистика по предложениям
        original_sentences = extract_sentences(original_content)
        optimized_sentences = extract_sentences(optimized_content)
        sentence_change = len(optimized_sentences) - len(original_sentences)

        return {
            "original_length": original_length,
            "optimized_length": optimized_length,
            "length_change": length_change,
            "length_change_percent": round(length_change_percent, 2),
            "original_keyword_count": original_keyword_count,
            "optimized_keyword_count": optimized_keyword_count,
            "keyword_change": keyword_change,
            "original_paragraph_count": len(original_paragraphs),
            "optimized_paragraph_count": len(optimized_paragraphs),
            "paragraph_change": paragraph_change,
            "original_sentence_count": len(original_sentences),
            "optimized_sentence_count": len(optimized_sentences),
            "sentence_change": sentence_change,
        }
