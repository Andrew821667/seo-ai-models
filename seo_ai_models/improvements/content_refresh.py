"""
Content Refresh Automation - Автоматическое обновление устаревшего контента.

Функции:
- Поиск устаревшего контента (>12 месяцев)
- Обновление дат и статистики
- Добавление новой информации
- Модернизация терминологии
- АВТОМАТИЧЕСКОЕ ИСПРАВЛЕНИЕ через AutoFix
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import re

logger = logging.getLogger(__name__)


class ContentRefreshAutomation:
    """Автоматическое обновление устаревшего контента."""

    def __init__(self, cms_connector=None, llm_service=None, autofix_engine=None):
        self.cms = cms_connector
        self.llm = llm_service
        self.autofix = autofix_engine
        logger.info("ContentRefreshAutomation initialized")

    def identify_outdated_content(self, threshold_days: int = 365) -> List[Dict]:
        """Находит устаревший контент."""
        if not self.cms:
            return []

        pages = self.cms.get_all_pages()
        outdated = []

        for page in pages:
            age = (datetime.now() - page.get("published_date", datetime.now())).days

            if age > threshold_days:
                score = self._calculate_refresh_priority(page, age)
                outdated.append(
                    {
                        "page_id": page["id"],
                        "title": page["title"],
                        "age_days": age,
                        "priority_score": score,
                        "traffic": page.get("traffic", 0),
                        "needs_refresh": True,
                    }
                )

        # Сортируем по приоритету
        outdated.sort(key=lambda x: x["priority_score"], reverse=True)

        logger.info(f"Found {len(outdated)} outdated pages")
        return outdated

    def _calculate_refresh_priority(self, page: Dict, age_days: int) -> float:
        """Расчет приоритета обновления."""
        # Факторы: возраст, трафик, позиции в поиске
        traffic = page.get("traffic", 0)
        rankings = page.get("avg_ranking", 100)

        # Формула приоритета
        age_score = min(age_days / 365, 3.0)  # Max 3 года
        traffic_score = min(traffic / 1000, 5.0)  # Normalize traffic
        ranking_score = (100 - rankings) / 100  # Лучше позиции = выше приоритет

        priority = (age_score * 0.3) + (traffic_score * 0.5) + (ranking_score * 0.2)

        return round(priority, 2)

    def auto_refresh(self, page_id: str) -> Dict[str, Any]:
        """АВТОМАТИЧЕСКОЕ обновление страницы."""
        page = self.cms.get_page(page_id)

        changes = {
            "updated_date": True,
            "updated_statistics": False,
            "added_sections": [],
            "modernized_terms": [],
        }

        # 1. Обновление даты модификации
        self.cms.update_modified_date(page_id)

        # 2. Обновление статистики и дат в тексте
        text = page.get("content", "")
        updated_text = self._update_statistics(text)

        if updated_text != text:
            changes["updated_statistics"] = True

        # 3. Добавление актуальной информации через LLM
        if self.llm:
            new_section = self._generate_update_section(page)
            if new_section:
                updated_text += f"\n\n{new_section}"
                changes["added_sections"].append("Recent Updates")

        # 4. Модернизация устаревших терминов
        modernized_text, terms = self._modernize_terminology(updated_text)
        changes["modernized_terms"] = terms

        # 5. Применение изменений
        if self.cms:
            self.cms.update_content(page_id, modernized_text)

        logger.info(f"✅ Auto-refreshed page {page_id}: {changes}")

        return {"success": True, "page_id": page_id, "changes": changes}

    def _update_statistics(self, text: str) -> str:
        """Обновление дат и статистики."""
        current_year = str(datetime.now().year)

        # Находим и обновляем годы (20XX)
        years_pattern = r"\b(20\d{2})\b"
        years = re.findall(years_pattern, text)

        for old_year in set(years):
            if int(old_year) < int(current_year) - 2:  # Старше 2 лет
                text = text.replace(old_year, current_year)
                logger.debug(f"Updated year: {old_year} → {current_year}")

        return text

    def _generate_update_section(self, page: Dict) -> str:
        """Генерация секции с обновлениями."""
        if not self.llm:
            return ""

        prompt = f"""Generate a brief "Recent Updates" section for this article:

Title: {page.get('title')}
Topic: {page.get('description', '')}

Include:
- Latest developments in the topic
- Current statistics (use {datetime.now().year})
- 2-3 paragraphs max

Keep it factual and relevant."""

        update_section = self.llm.generate(prompt, max_tokens=300)
        return f"## Recent Updates\n\n{update_section}"

    def _modernize_terminology(self, text: str) -> tuple:
        """Обновление устаревших терминов."""
        # Словарь замен
        replacements = {
            "webmaster": "SEO specialist",
            "meta keywords": "meta tags",
            "PageRank": "Domain Authority",
            "Google+": "social media",
            # Добавить больше по мере необходимости
        }

        modernized = []
        updated_text = text

        for old_term, new_term in replacements.items():
            if old_term in updated_text:
                updated_text = updated_text.replace(old_term, new_term)
                modernized.append({old_term: new_term})

        return updated_text, modernized

    def schedule_refresh(self, page_ids: List[str], interval_days: int = 90):
        """Планирование периодического обновления."""
        # TODO: Интеграция с планировщиком задач
        for page_id in page_ids:
            logger.info(f"Scheduled refresh for {page_id} every {interval_days} days")

    def get_refresh_report(self) -> Dict[str, Any]:
        """Отчет об обновлениях."""
        outdated = self.identify_outdated_content()

        return {
            "total_outdated": len(outdated),
            "high_priority": len([p for p in outdated if p["priority_score"] > 2.0]),
            "pages": outdated[:10],  # Top 10
            "recommendation": "Auto-refresh high priority pages weekly",
        }
