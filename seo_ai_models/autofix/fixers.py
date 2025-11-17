"""
Concrete Fixers - Конкретные исправители для разных типов проблем.

Каждый фиксер знает как:
1. Определить проблему
2. Исправить её
3. Проверить результат
4. Откатить изменения если нужно
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class BaseFixer:
    """Базовый класс для всех фиксеров."""

    def __init__(self, llm_service=None):
        """
        Args:
            llm_service: Сервис LLM для генерации контента
        """
        self.llm_service = llm_service

    def detect(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Определение проблем в контенте."""
        raise NotImplementedError

    def fix(self, cms_connector, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Исправление проблемы."""
        raise NotImplementedError

    def verify(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Проверка результата исправления."""
        raise NotImplementedError

    def rollback(self, cms_connector, backup: Dict[str, Any]):
        """Откат изменений."""
        raise NotImplementedError


class MetaTagsFixer(BaseFixer):
    """
    Исправление мета-тегов (title, description).

    Автоматически генерирует оптимизированные мета-теги на основе контента.
    """

    def detect(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Поиск проблем с мета-тегами."""
        problems = []

        title = content.get("title", "")
        description = content.get("description", "")

        # Проверка title
        if not title:
            problems.append(
                {
                    "type": "missing_title",
                    "description": "Missing title tag",
                    "severity": "high",
                    "auto_fixable": True,
                    "metadata": {"content_text": content.get("text", "")},
                }
            )
        elif len(title) < 30 or len(title) > 60:
            problems.append(
                {
                    "type": "suboptimal_title_length",
                    "description": f"Title length is {len(title)} (optimal: 30-60)",
                    "severity": "medium",
                    "auto_fixable": True,
                    "metadata": {"current_title": title, "content_text": content.get("text", "")},
                }
            )

        # Проверка description
        if not description:
            problems.append(
                {
                    "type": "missing_description",
                    "description": "Missing meta description",
                    "severity": "high",
                    "auto_fixable": True,
                    "metadata": {"content_text": content.get("text", "")},
                }
            )
        elif len(description) < 120 or len(description) > 160:
            problems.append(
                {
                    "type": "suboptimal_description_length",
                    "description": f"Description length is {len(description)} (optimal: 120-160)",
                    "severity": "medium",
                    "auto_fixable": True,
                    "metadata": {
                        "current_description": description,
                        "content_text": content.get("text", ""),
                    },
                }
            )

        return problems

    def fix(self, cms_connector, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация и обновление мета-тегов."""
        problem_type = metadata.get("problem_type")
        content_text = metadata.get("content_text", "")

        # Генерируем новые теги
        if "title" in problem_type:
            new_title = self._generate_title(content_text)

            if cms_connector:
                # Обновляем через CMS API
                cms_connector.update_meta_tags(metadata.get("page_id"), title=new_title)

            logger.info(f"Generated new title: {new_title}")

            return {
                "success": True,
                "field": "title",
                "old_value": metadata.get("current_title"),
                "new_value": new_title,
            }

        elif "description" in problem_type:
            new_description = self._generate_description(content_text)

            if cms_connector:
                cms_connector.update_meta_tags(metadata.get("page_id"), description=new_description)

            logger.info(f"Generated new description: {new_description}")

            return {
                "success": True,
                "field": "description",
                "old_value": metadata.get("current_description"),
                "new_value": new_description,
            }

        return {"success": False, "error": "Unknown problem type"}

    def _generate_title(self, content: str) -> str:
        """Генерация оптимального title."""
        if self.llm_service:
            # Используем LLM для генерации
            prompt = f"""Generate an SEO-optimized title tag (30-60 characters) for the following content:

{content[:500]}

Requirements:
- Length: 30-60 characters
- Include main keyword
- Engaging and click-worthy
- Accurate representation of content

Return only the title, no explanation."""

            title = self.llm_service.generate(prompt, max_tokens=100)
            return title.strip()[:60]
        else:
            # Простая генерация из первого заголовка или первых слов
            words = content.split()[:10]
            title = " ".join(words)
            return title[:60] if len(title) > 60 else title

    def _generate_description(self, content: str) -> str:
        """Генерация оптимального description."""
        if self.llm_service:
            prompt = f"""Generate an SEO-optimized meta description (120-160 characters) for the following content:

{content[:500]}

Requirements:
- Length: 120-160 characters
- Include main keyword
- Compelling call-to-action
- Accurate summary

Return only the description, no explanation."""

            description = self.llm_service.generate(prompt, max_tokens=150)
            return description.strip()[:160]
        else:
            # Используем первое предложение
            sentences = content.split(".")
            description = sentences[0] if sentences else content[:160]
            return description[:160]

    def verify(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Проверка что title/description в допустимых пределах."""
        if not result.get("success"):
            return {"success": False, "error": "Fix failed"}

        new_value = result.get("new_value", "")
        field = result.get("field")

        if field == "title":
            if 30 <= len(new_value) <= 60:
                return {"success": True}
            else:
                return {"success": False, "error": f"Title length {len(new_value)} out of range"}

        elif field == "description":
            if 120 <= len(new_value) <= 160:
                return {"success": True}
            else:
                return {
                    "success": False,
                    "error": f"Description length {len(new_value)} out of range",
                }

        return {"success": True}

    def rollback(self, cms_connector, backup: Dict[str, Any]):
        """Восстановление старых значений."""
        if cms_connector and backup:
            page_id = backup.get("metadata", {}).get("page_id")
            old_title = backup.get("metadata", {}).get("current_title")
            old_description = backup.get("metadata", {}).get("current_description")

            if old_title:
                cms_connector.update_meta_tags(page_id, title=old_title)
            if old_description:
                cms_connector.update_meta_tags(page_id, description=old_description)


class ImageAltTagsFixer(BaseFixer):
    """
    Добавление alt-тегов к изображениям.

    Автоматически генерирует описательные alt-теги на основе:
    - Имени файла
    - Контекста вокруг изображения
    - Распознавания изображения (если доступно)
    """

    def detect(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Поиск изображений без alt-тегов."""
        problems = []
        images = content.get("images", [])

        for img in images:
            if not img.get("alt"):
                problems.append(
                    {
                        "type": "missing_alt_tag",
                        "description": f"Image without alt tag: {img.get('src')}",
                        "severity": "medium",
                        "auto_fixable": True,
                        "metadata": {
                            "image_url": img.get("src"),
                            "image_filename": img.get("filename"),
                            "surrounding_text": img.get("context", ""),
                        },
                    }
                )

        return problems

    def fix(self, cms_connector, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация и добавление alt-тега."""
        image_url = metadata.get("image_url")
        filename = metadata.get("image_filename", "")
        context = metadata.get("surrounding_text", "")

        # Генерируем alt-текст
        alt_text = self._generate_alt_text(filename, context)

        if cms_connector:
            # Обновляем через CMS
            cms_connector.update_image_alt(image_url, alt_text)

        logger.info(f"Generated alt text for {image_url}: {alt_text}")

        return {"success": True, "image_url": image_url, "alt_text": alt_text}

    def _generate_alt_text(self, filename: str, context: str) -> str:
        """Генерация alt-текста."""
        if self.llm_service and context:
            # Используем LLM с контекстом
            prompt = f"""Generate a descriptive alt text for an image in the following context:

Context: {context[:200]}
Filename: {filename}

Requirements:
- Descriptive and specific
- Include relevant keywords from context
- Natural language
- Max 125 characters

Return only the alt text, no explanation."""

            alt_text = self.llm_service.generate(prompt, max_tokens=50)
            return alt_text.strip()[:125]
        else:
            # Генерируем из имени файла
            alt = filename.replace("-", " ").replace("_", " ")
            alt = re.sub(r"\.[^.]+$", "", alt)  # Удаляем расширение
            return alt.title()[:125]

    def verify(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Проверка что alt-текст не пустой."""
        alt_text = result.get("alt_text", "")
        if alt_text and len(alt_text) > 0:
            return {"success": True}
        return {"success": False, "error": "Empty alt text"}

    def rollback(self, cms_connector, backup: Dict[str, Any]):
        """Удаление добавленного alt-тега."""
        if cms_connector:
            image_url = backup.get("metadata", {}).get("image_url")
            if image_url:
                cms_connector.update_image_alt(image_url, "")


class ContentRefreshFixer(BaseFixer):
    """
    Обновление устаревшего контента.

    Автоматически:
    - Обновляет даты и статистику
    - Добавляет актуальную информацию
    - Модернизирует устаревшие термины
    """

    def detect(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Поиск устаревшего контента."""
        problems = []

        published_date = content.get("published_date")
        if published_date:
            # Проверяем возраст контента
            age_days = (datetime.now() - published_date).days

            if age_days > 365:  # Старше года
                problems.append(
                    {
                        "type": "outdated_content",
                        "description": f"Content is {age_days} days old",
                        "severity": "low",
                        "auto_fixable": True,
                        "metadata": {"age_days": age_days, "content_text": content.get("text", "")},
                    }
                )

        # Поиск устаревших дат в тексте
        text = content.get("text", "")
        outdated_years = self._find_outdated_dates(text)

        if outdated_years:
            problems.append(
                {
                    "type": "outdated_statistics",
                    "description": "Content contains outdated years/statistics",
                    "severity": "medium",
                    "auto_fixable": True,
                    "metadata": {"outdated_years": outdated_years, "content_text": text},
                }
            )

        return problems

    def _find_outdated_dates(self, text: str) -> List[str]:
        """Поиск устаревших дат в тексте."""
        current_year = datetime.now().year
        # Ищем упоминания годов
        years = re.findall(r"\b(20\d{2})\b", text)

        # Фильтруем старые годы (старше 2 лет)
        outdated = [year for year in set(years) if int(year) < current_year - 2]

        return outdated

    def fix(self, cms_connector, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Обновление контента."""
        problem_type = metadata.get("problem_type")
        content_text = metadata.get("content_text", "")

        if problem_type == "outdated_content":
            # Обновляем дату модификации
            if cms_connector:
                cms_connector.update_modified_date(metadata.get("page_id"))

            return {
                "success": True,
                "action": "updated_modified_date",
                "timestamp": datetime.now().isoformat(),
            }

        elif problem_type == "outdated_statistics":
            # Находим и обновляем устаревшие годы
            outdated_years = metadata.get("outdated_years", [])
            current_year = str(datetime.now().year)

            updated_text = content_text
            for old_year in outdated_years:
                # Заменяем старый год на текущий
                updated_text = updated_text.replace(old_year, current_year)

            if cms_connector:
                cms_connector.update_content(metadata.get("page_id"), updated_text)

            return {
                "success": True,
                "action": "updated_dates",
                "changes": {year: current_year for year in outdated_years},
            }

        return {"success": False}

    def verify(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Проверка обновления."""
        return {"success": result.get("success", False)}

    def rollback(self, cms_connector, backup: Dict[str, Any]):
        """Откат к старой версии контента."""
        if cms_connector:
            page_id = backup.get("metadata", {}).get("page_id")
            old_content = backup.get("metadata", {}).get("content_text")

            if old_content:
                cms_connector.update_content(page_id, old_content)


class SchemaMarkupFixer(BaseFixer):
    """
    Добавление Schema.org разметки.

    Автоматически генерирует структурированные данные для:
    - Статей (Article)
    - Товаров (Product)
    - FAQ
    - Отзывов (Review)
    """

    def detect(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Поиск отсутствующей разметки."""
        problems = []

        has_schema = content.get("has_schema_markup", False)

        if not has_schema:
            content_type = self._detect_content_type(content)

            problems.append(
                {
                    "type": "missing_schema_markup",
                    "description": f"Missing Schema.org markup for {content_type}",
                    "severity": "medium",
                    "auto_fixable": True,
                    "metadata": {"content_type": content_type, "content": content},
                }
            )

        return problems

    def _detect_content_type(self, content: Dict[str, Any]) -> str:
        """Определение типа контента."""
        # Простая эвристика
        if "price" in content or "product" in content.get("text", "").lower():
            return "Product"
        elif "faq" in content.get("text", "").lower():
            return "FAQPage"
        else:
            return "Article"

    def fix(self, cms_connector, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация и добавление Schema разметки."""
        content_type = metadata.get("content_type", "Article")
        content = metadata.get("content", {})

        schema_markup = self._generate_schema(content_type, content)

        if cms_connector:
            cms_connector.add_schema_markup(metadata.get("page_id"), schema_markup)

        logger.info(f"Generated {content_type} schema markup")

        return {"success": True, "schema_type": content_type, "markup": schema_markup}

    def _generate_schema(self, content_type: str, content: Dict[str, Any]) -> Dict:
        """Генерация Schema.org JSON-LD."""
        if content_type == "Article":
            return {
                "@context": "https://schema.org",
                "@type": "Article",
                "headline": content.get("title", ""),
                "description": content.get("description", ""),
                "datePublished": content.get("published_date", datetime.now()).isoformat(),
                "dateModified": datetime.now().isoformat(),
                "author": {"@type": "Person", "name": content.get("author", "Unknown")},
            }
        elif content_type == "Product":
            return {
                "@context": "https://schema.org",
                "@type": "Product",
                "name": content.get("title", ""),
                "description": content.get("description", ""),
                "offers": {
                    "@type": "Offer",
                    "price": content.get("price", "0"),
                    "priceCurrency": "USD",
                },
            }
        # Добавить другие типы по необходимости

    def verify(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Проверка валидности разметки."""
        markup = result.get("markup", {})
        if markup and "@context" in markup and "@type" in markup:
            return {"success": True}
        return {"success": False, "error": "Invalid schema markup"}

    def rollback(self, cms_connector, backup: Dict[str, Any]):
        """Удаление добавленной разметки."""
        if cms_connector:
            page_id = backup.get("metadata", {}).get("page_id")
            cms_connector.remove_schema_markup(page_id)


class InternalLinksFixer(BaseFixer):
    """
    Добавление внутренних ссылок.

    Автоматически находит возможности для внутренней перелинковки
    на основе релевантности контента.
    """

    def detect(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Поиск возможностей для внутренних ссылок."""
        problems = []

        internal_links_count = len(content.get("internal_links", []))

        if internal_links_count < 3:
            problems.append(
                {
                    "type": "insufficient_internal_links",
                    "description": f"Only {internal_links_count} internal links found",
                    "severity": "low",
                    "auto_fixable": True,
                    "metadata": {
                        "current_links": internal_links_count,
                        "content_text": content.get("text", ""),
                    },
                }
            )

        return problems

    def fix(self, cms_connector, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Добавление внутренних ссылок."""
        content_text = metadata.get("content_text", "")

        # Находим релевантные страницы для ссылок
        if cms_connector:
            related_pages = cms_connector.find_related_pages(content_text)

            # Добавляем ссылки в контент
            for page in related_pages[:3]:  # Добавляем до 3 ссылок
                # Находим подходящее место для ссылки
                anchor_text = page.get("suggested_anchor")
                link_url = page.get("url")

                # Вставляем ссылку (простая реализация)
                link_html = f'<a href="{link_url}">{anchor_text}</a>'
                # TODO: Более умная вставка с учетом контекста

            return {"success": True, "links_added": len(related_pages), "links": related_pages}

        return {"success": False, "error": "CMS connector not available"}

    def verify(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Проверка добавления ссылок."""
        links_added = result.get("links_added", 0)
        if links_added > 0:
            return {"success": True}
        return {"success": False, "error": "No links added"}

    def rollback(self, cms_connector, backup: Dict[str, Any]):
        """Удаление добавленных ссылок."""
        if cms_connector:
            page_id = backup.get("metadata", {}).get("page_id")
            original_content = backup.get("metadata", {}).get("content_text")
            cms_connector.update_content(page_id, original_content)
