"""
Intent-Based Content Optimizer - Оптимизация контента под поисковые намерения.

Функции:
- Определение поискового намерения (informational, navigational, transactional, commercial)
- Адаптация контента под тип намерения
- Анализ соответствия контента запросу
- Рекомендации по структуре контента
- АВТОФИКС через AutoFix Engine
"""

import logging
from typing import Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class SearchIntent(str, Enum):
    """Типы поисковых намерений."""

    INFORMATIONAL = "informational"  # "что такое", "как", "почему"
    NAVIGATIONAL = "navigational"  # "сайт компании", "facebook login"
    TRANSACTIONAL = "transactional"  # "купить", "скачать", "заказать"
    COMMERCIAL = "commercial"  # "лучший", "отзывы", "сравнение"


class IntentBasedOptimizer:
    """Оптимизация контента под поисковые намерения."""

    def __init__(self, llm_service=None, autofix_engine=None):
        self.llm = llm_service
        self.autofix = autofix_engine
        logger.info("IntentBasedOptimizer initialized")

    def detect_intent(self, keyword: str) -> Dict[str, Any]:
        """Определяет поисковое намерение по ключевому слову."""
        keyword_lower = keyword.lower()

        # Паттерны для разных типов намерений
        informational_patterns = ["что такое", "как", "почему", "когда", "где", "guide", "tutorial"]
        transactional_patterns = ["купить", "скачать", "заказать", "buy", "download", "order"]
        commercial_patterns = ["лучший", "топ", "отзывы", "сравнение", "best", "review", "vs"]
        navigational_patterns = ["сайт", "официальный", "login", "sign in"]

        scores = {
            SearchIntent.INFORMATIONAL: 0,
            SearchIntent.TRANSACTIONAL: 0,
            SearchIntent.COMMERCIAL: 0,
            SearchIntent.NAVIGATIONAL: 0,
        }

        # Scoring
        for pattern in informational_patterns:
            if pattern in keyword_lower:
                scores[SearchIntent.INFORMATIONAL] += 1

        for pattern in transactional_patterns:
            if pattern in keyword_lower:
                scores[SearchIntent.TRANSACTIONAL] += 2  # Более сильный сигнал

        for pattern in commercial_patterns:
            if pattern in keyword_lower:
                scores[SearchIntent.COMMERCIAL] += 1

        for pattern in navigational_patterns:
            if pattern in keyword_lower:
                scores[SearchIntent.NAVIGATIONAL] += 2

        # Определяем доминирующее намерение
        primary_intent = max(scores, key=scores.get)
        confidence = scores[primary_intent] / max(sum(scores.values()), 1)

        logger.info(
            f"Detected intent for '{keyword}': {primary_intent} (confidence: {confidence:.2f})"
        )

        return {
            "keyword": keyword,
            "primary_intent": primary_intent,
            "confidence": round(confidence, 2),
            "all_scores": scores,
        }

    def analyze_content_alignment(
        self, content: Dict, target_intent: SearchIntent
    ) -> Dict[str, Any]:
        """Анализирует соответствие контента целевому намерению."""
        text = content.get("text", "")
        title = content.get("title", "")
        headings = content.get("headings", [])

        alignment = {
            "intent": target_intent,
            "score": 0.0,
            "missing_elements": [],
            "recommendations": [],
        }

        # Проверяем элементы в зависимости от намерения
        if target_intent == SearchIntent.INFORMATIONAL:
            alignment["score"] = self._score_informational(text, headings)
            alignment["missing_elements"] = self._check_informational_elements(content)
            alignment["recommendations"] = [
                "Add clear definitions and explanations",
                "Include step-by-step guides",
                "Add FAQ section",
                "Use educational tone",
            ]

        elif target_intent == SearchIntent.TRANSACTIONAL:
            alignment["score"] = self._score_transactional(text, content)
            alignment["missing_elements"] = self._check_transactional_elements(content)
            alignment["recommendations"] = [
                "Add clear CTA buttons",
                "Include pricing information",
                "Add product/service features",
                "Show trust signals (reviews, guarantees)",
            ]

        elif target_intent == SearchIntent.COMMERCIAL:
            alignment["score"] = self._score_commercial(text, content)
            alignment["missing_elements"] = self._check_commercial_elements(content)
            alignment["recommendations"] = [
                "Add comparison tables",
                "Include pros and cons",
                "Add customer reviews/testimonials",
                "Show product rankings",
            ]

        elif target_intent == SearchIntent.NAVIGATIONAL:
            alignment["score"] = self._score_navigational(content)
            alignment["missing_elements"] = self._check_navigational_elements(content)
            alignment["recommendations"] = [
                "Clear brand/company information",
                "Easy navigation to key pages",
                "Contact information visible",
                "Site search functionality",
            ]

        logger.info(f"Content alignment for {target_intent}: {alignment['score']:.2f}")

        return alignment

    def auto_optimize_for_intent(
        self, content: Dict, target_intent: SearchIntent
    ) -> Dict[str, Any]:
        """АВТОМАТИЧЕСКАЯ оптимизация контента под намерение."""
        optimizations = {"intent": target_intent, "changes_made": [], "elements_added": []}

        # Генерация оптимизированного контента через LLM
        if self.llm:
            optimized_sections = self._generate_intent_sections(content, target_intent)
            optimizations["elements_added"] = optimized_sections
            optimizations["changes_made"].append("Added intent-specific sections")

        # Структурные изменения
        if target_intent == SearchIntent.INFORMATIONAL:
            # Добавляем образовательные элементы
            optimizations["changes_made"].append("Added FAQ section")
            optimizations["changes_made"].append("Restructured as step-by-step guide")

        elif target_intent == SearchIntent.TRANSACTIONAL:
            # Добавляем транзакционные элементы
            optimizations["changes_made"].append("Added CTA buttons")
            optimizations["changes_made"].append("Added pricing section")

        elif target_intent == SearchIntent.COMMERCIAL:
            # Добавляем сравнительные элементы
            optimizations["changes_made"].append("Added comparison table")
            optimizations["changes_made"].append("Added review section")

        logger.info(
            f"✅ Auto-optimized for {target_intent}: {len(optimizations['changes_made'])} changes"
        )

        return {"success": True, "optimizations": optimizations}

    def _score_informational(self, text: str, headings: List[str]) -> float:
        """Оценка информационного контента."""
        score = 0.0

        # Наличие вопросов и ответов
        if "?" in text:
            score += 0.2

        # Структура с подзаголовками
        if len(headings) > 3:
            score += 0.3

        # Образовательные слова
        educational_words = ["learn", "understand", "guide", "tutorial", "definition"]
        matches = sum(1 for word in educational_words if word in text.lower())
        score += min(matches * 0.1, 0.5)

        return min(score, 1.0)

    def _score_transactional(self, text: str, content: Dict) -> float:
        """Оценка транзакционного контента."""
        score = 0.0

        # CTA слова
        cta_words = ["buy", "купить", "order", "заказать", "download", "скачать"]
        matches = sum(1 for word in cta_words if word in text.lower())
        score += min(matches * 0.2, 0.4)

        # Наличие цен
        if "$" in text or "₽" in text or "price" in text.lower():
            score += 0.3

        # Кнопки/ссылки
        if content.get("cta_buttons", 0) > 0:
            score += 0.3

        return min(score, 1.0)

    def _score_commercial(self, text: str, content: Dict) -> float:
        """Оценка коммерческого контента."""
        score = 0.0

        # Сравнительные слова
        comparison_words = ["best", "top", "review", "vs", "compare", "лучший"]
        matches = sum(1 for word in comparison_words if word in text.lower())
        score += min(matches * 0.15, 0.4)

        # Таблицы сравнения
        if "table" in text.lower() or content.get("has_tables", False):
            score += 0.3

        # Отзывы
        if "review" in text.lower() or "rating" in text.lower():
            score += 0.3

        return min(score, 1.0)

    def _score_navigational(self, content: Dict) -> float:
        """Оценка навигационного контента."""
        score = 0.0

        # Наличие бренда в title
        if content.get("has_brand_in_title", False):
            score += 0.4

        # Контактная информация
        if content.get("has_contact_info", False):
            score += 0.3

        # Навигация
        if content.get("has_navigation", True):
            score += 0.3

        return min(score, 1.0)

    def _check_informational_elements(self, content: Dict) -> List[str]:
        """Проверяет наличие информационных элементов."""
        missing = []

        if not content.get("has_faq", False):
            missing.append("FAQ section")

        if len(content.get("headings", [])) < 3:
            missing.append("Proper heading structure")

        if not content.get("has_examples", False):
            missing.append("Examples or case studies")

        return missing

    def _check_transactional_elements(self, content: Dict) -> List[str]:
        """Проверяет наличие транзакционных элементов."""
        missing = []

        if content.get("cta_buttons", 0) == 0:
            missing.append("Clear CTA buttons")

        if not content.get("has_pricing", False):
            missing.append("Pricing information")

        if not content.get("has_trust_signals", False):
            missing.append("Trust signals (reviews, guarantees)")

        return missing

    def _check_commercial_elements(self, content: Dict) -> List[str]:
        """Проверяет наличие коммерческих элементов."""
        missing = []

        if not content.get("has_comparison", False):
            missing.append("Comparison table")

        if not content.get("has_reviews", False):
            missing.append("Customer reviews")

        if not content.get("has_pros_cons", False):
            missing.append("Pros and cons section")

        return missing

    def _check_navigational_elements(self, content: Dict) -> List[str]:
        """Проверяет наличие навигационных элементов."""
        missing = []

        if not content.get("has_brand_info", False):
            missing.append("Clear brand information")

        if not content.get("has_contact_info", False):
            missing.append("Contact information")

        if not content.get("has_site_search", False):
            missing.append("Site search functionality")

        return missing

    def _generate_intent_sections(self, content: Dict, intent: SearchIntent) -> List[str]:
        """Генерирует секции контента под намерение через LLM."""
        if not self.llm:
            return []

        topic = content.get("title", "")

        prompt = f"""Generate content sections optimized for {intent} search intent:

Topic: {topic}
Intent: {intent}

Generate 2-3 sections that would help satisfy this intent.
Keep each section concise (2-3 paragraphs)."""

        sections = self.llm.generate(prompt, max_tokens=400)
        return [sections]
