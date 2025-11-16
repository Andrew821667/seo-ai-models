"""
Competitor Monitor - Мониторинг и анализ конкурентов в реальном времени.

Функции:
- Отслеживание позиций конкурентов
- Анализ контента конкурентов
- Обнаружение новых backlinks конкурентов
- Gap-анализ (что есть у конкурентов, чего нет у нас)
- Alerts при изменениях у конкурентов
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class CompetitorMonitor:
    """Мониторинг конкурентов."""

    def __init__(self, crawler=None, llm_service=None):
        self.crawler = crawler
        self.llm = llm_service
        self.competitors_data = {}
        logger.info("CompetitorMonitor initialized")

    def add_competitor(self, name: str, domain: str, keywords: List[str]) -> Dict:
        """Добавляет конкурента для мониторинга."""
        competitor = {
            "name": name,
            "domain": domain,
            "keywords": keywords,
            "added_date": datetime.now().isoformat(),
            "last_checked": None,
            "baseline_data": None
        }

        self.competitors_data[domain] = competitor
        logger.info(f"Added competitor: {name} ({domain})")

        # Сразу собираем базовые данные
        baseline = self._collect_baseline_data(domain, keywords)
        competitor["baseline_data"] = baseline
        competitor["last_checked"] = datetime.now().isoformat()

        return {
            "success": True,
            "competitor": competitor
        }

    def monitor_rankings(self, keywords: List[str]) -> Dict[str, Any]:
        """Отслеживает позиции конкурентов по ключевым словам."""
        rankings = {
            "timestamp": datetime.now().isoformat(),
            "keywords": {},
            "changes_detected": []
        }

        for keyword in keywords:
            keyword_data = {
                "keyword": keyword,
                "competitors_positions": {}
            }

            for domain, competitor in self.competitors_data.items():
                if keyword in competitor["keywords"]:
                    # Здесь должен быть реальный парсинг SERP
                    # Для примера используем заглушку
                    current_position = self._get_serp_position(domain, keyword)
                    previous_position = self._get_previous_position(domain, keyword)

                    keyword_data["competitors_positions"][domain] = {
                        "current": current_position,
                        "previous": previous_position,
                        "change": previous_position - current_position if previous_position else 0
                    }

                    # Детект изменений
                    if previous_position and abs(current_position - previous_position) >= 3:
                        rankings["changes_detected"].append({
                            "competitor": competitor["name"],
                            "keyword": keyword,
                            "movement": "up" if current_position < previous_position else "down",
                            "positions": abs(current_position - previous_position)
                        })

            rankings["keywords"][keyword] = keyword_data

        logger.info(f"Monitored rankings for {len(keywords)} keywords. Changes: {len(rankings['changes_detected'])}")

        return rankings

    def analyze_competitor_content(self, competitor_domain: str, url: str) -> Dict[str, Any]:
        """Глубокий анализ контента конкурента."""
        if not self.crawler:
            return {"error": "Crawler not available"}

        # Парсим страницу
        content = self.crawler.fetch_page(url)

        analysis = {
            "url": url,
            "competitor": competitor_domain,
            "word_count": len(content.get("text", "").split()),
            "headings_count": len(content.get("headings", [])),
            "images_count": len(content.get("images", [])),
            "internal_links": len(content.get("internal_links", [])),
            "external_links": len(content.get("external_links", [])),
            "has_schema": bool(content.get("schema_markup")),
            "meta_quality": self._assess_meta_quality(content),
            "content_structure": self._analyze_structure(content),
            "key_topics": self._extract_topics(content.get("text", ""))
        }

        # LLM-анализ для более глубоких инсайтов
        if self.llm:
            insights = self._get_llm_insights(content)
            analysis["llm_insights"] = insights

        logger.info(f"Analyzed competitor content: {url}")

        return analysis

    def gap_analysis(self, our_domain: str, competitor_domain: str) -> Dict[str, Any]:
        """Gap-анализ: что есть у конкурента, чего нет у нас."""
        gaps = {
            "content_gaps": [],
            "keyword_gaps": [],
            "feature_gaps": [],
            "opportunities": []
        }

        # Сравниваем контент
        our_pages = self._get_domain_pages(our_domain)
        their_pages = self._get_domain_pages(competitor_domain)

        # Находим темы, которые есть у них, но нет у нас
        our_topics = set([self._extract_main_topic(p) for p in our_pages])
        their_topics = set([self._extract_main_topic(p) for p in their_pages])

        content_gaps = their_topics - our_topics
        gaps["content_gaps"] = list(content_gaps)

        # Keyword gaps (упрощенная версия)
        # В реальности нужна интеграция с SEO tools API (Ahrefs, SEMrush)
        gaps["keyword_gaps"] = self._find_keyword_gaps(our_domain, competitor_domain)

        # Feature gaps (технические возможности)
        our_features = self._analyze_site_features(our_domain)
        their_features = self._analyze_site_features(competitor_domain)

        feature_gaps = set(their_features) - set(our_features)
        gaps["feature_gaps"] = list(feature_gaps)

        # Opportunities - что можно сделать лучше
        gaps["opportunities"] = self._identify_opportunities(gaps)

        logger.info(f"Gap analysis complete. Found {len(gaps['content_gaps'])} content gaps")

        return gaps

    def track_backlinks(self, competitor_domain: str) -> Dict[str, Any]:
        """Отслеживает новые backlinks конкурентов."""
        # В реальности нужна интеграция с backlink API
        backlinks = {
            "domain": competitor_domain,
            "total_backlinks": 0,
            "new_backlinks": [],
            "lost_backlinks": [],
            "top_referring_domains": []
        }

        # Заглушка - в продакшене использовать API вроде Ahrefs/Majestic
        logger.info(f"Tracked backlinks for {competitor_domain}")

        return backlinks

    def get_alert_report(self) -> Dict[str, Any]:
        """Генерирует отчет об изменениях у конкурентов."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "competitors_monitored": len(self.competitors_data),
            "alerts": [],
            "summary": {}
        }

        for domain, competitor in self.competitors_data.items():
            # Проверяем различные метрики
            alerts = []

            # Проверка позиций
            ranking_changes = self._check_ranking_changes(competitor)
            if ranking_changes:
                alerts.extend(ranking_changes)

            # Проверка нового контента
            new_content = self._check_new_content(competitor)
            if new_content:
                alerts.extend(new_content)

            if alerts:
                report["alerts"].extend(alerts)

        # Summary
        report["summary"] = {
            "total_alerts": len(report["alerts"]),
            "ranking_changes": len([a for a in report["alerts"] if a["type"] == "ranking"]),
            "new_content": len([a for a in report["alerts"] if a["type"] == "content"])
        }

        logger.info(f"Alert report generated: {report['summary']['total_alerts']} alerts")

        return report

    # Helper methods

    def _collect_baseline_data(self, domain: str, keywords: List[str]) -> Dict:
        """Собирает базовые данные о конкуренте."""
        baseline = {
            "domain": domain,
            "keywords_positions": {},
            "content_snapshot": {},
            "collected_at": datetime.now().isoformat()
        }

        for keyword in keywords:
            position = self._get_serp_position(domain, keyword)
            baseline["keywords_positions"][keyword] = position

        return baseline

    def _get_serp_position(self, domain: str, keyword: str) -> int:
        """Получает позицию домена в SERP по ключевому слову."""
        # Заглушка - в реальности парсить Google/Yandex
        # Или использовать API вроде SerpAPI
        import random
        return random.randint(1, 100)

    def _get_previous_position(self, domain: str, keyword: str) -> int:
        """Получает предыдущую позицию из истории."""
        competitor = self.competitors_data.get(domain, {})
        baseline = competitor.get("baseline_data", {})
        return baseline.get("keywords_positions", {}).get(keyword, 0)

    def _assess_meta_quality(self, content: Dict) -> Dict:
        """Оценивает качество meta-тегов."""
        title = content.get("title", "")
        description = content.get("description", "")

        return {
            "title_length": len(title),
            "title_optimal": 30 <= len(title) <= 60,
            "description_length": len(description),
            "description_optimal": 120 <= len(description) <= 160,
            "has_keywords": bool(content.get("target_keywords"))
        }

    def _analyze_structure(self, content: Dict) -> Dict:
        """Анализирует структуру контента."""
        headings = content.get("headings", [])

        return {
            "has_h1": any(h.get("level") == 1 for h in headings),
            "h2_count": sum(1 for h in headings if h.get("level") == 2),
            "h3_count": sum(1 for h in headings if h.get("level") == 3),
            "well_structured": len(headings) >= 3
        }

    def _extract_topics(self, text: str) -> List[str]:
        """Извлекает ключевые топики из текста."""
        # Упрощенная версия - в реальности использовать NLP
        words = re.findall(r'\b[a-zA-Zа-яА-Я]{5,}\b', text.lower())
        from collections import Counter
        common = Counter(words).most_common(10)
        return [word for word, _ in common]

    def _get_llm_insights(self, content: Dict) -> str:
        """Получает инсайты от LLM."""
        if not self.llm:
            return ""

        prompt = f"""Analyze this competitor's content and provide insights:

Title: {content.get('title')}
Word count: {len(content.get('text', '').split())}
Headings: {len(content.get('headings', []))}

What makes this content effective? What can we learn from it?"""

        insights = self.llm.generate(prompt, max_tokens=200)
        return insights

    def _get_domain_pages(self, domain: str) -> List[Dict]:
        """Получает список страниц домена."""
        # Заглушка - в реальности использовать sitemap или crawler
        return []

    def _extract_main_topic(self, page: Dict) -> str:
        """Извлекает основную тему страницы."""
        return page.get("title", "").lower()

    def _find_keyword_gaps(self, our_domain: str, their_domain: str) -> List[str]:
        """Находит keyword gaps."""
        # Заглушка - нужен API доступ к keyword data
        return []

    def _analyze_site_features(self, domain: str) -> List[str]:
        """Анализирует технические возможности сайта."""
        features = []

        # Заглушка - в реальности проверять:
        # - HTTPS
        # - Mobile-friendly
        # - Page speed
        # - Schema markup
        # - AMP
        # - etc.

        return features

    def _identify_opportunities(self, gaps: Dict) -> List[str]:
        """Определяет возможности на основе gaps."""
        opportunities = []

        if gaps["content_gaps"]:
            opportunities.append(f"Create content on {len(gaps['content_gaps'])} missing topics")

        if gaps["keyword_gaps"]:
            opportunities.append(f"Target {len(gaps['keyword_gaps'])} untapped keywords")

        if gaps["feature_gaps"]:
            opportunities.append(f"Implement {len(gaps['feature_gaps'])} missing features")

        return opportunities

    def _check_ranking_changes(self, competitor: Dict) -> List[Dict]:
        """Проверяет изменения в позициях."""
        # Заглушка
        return []

    def _check_new_content(self, competitor: Dict) -> List[Dict]:
        """Проверяет новый контент у конкурента."""
        # Заглушка
        return []
