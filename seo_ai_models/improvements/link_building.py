"""
Link Building Assistant - Помощник в построении ссылочной массы.

Функции:
- Поиск возможностей для link building
- Анализ качества потенциальных доноров
- Outreach email генерация
- Broken link building
- Competitor backlink analysis
"""

import logging
from typing import Dict, Any, List
import re

logger = logging.getLogger(__name__)


class LinkBuildingAssistant:
    """Помощник в построении ссылочной массы."""

    def __init__(self, llm_service=None, crawler=None):
        self.llm = llm_service
        self.crawler = crawler
        logger.info("LinkBuildingAssistant initialized")

    def find_link_opportunities(self, domain: str, niche: str) -> List[Dict]:
        """Находит возможности для получения ссылок."""
        opportunities = []

        # 1. Guest posting opportunities
        guest_post_sites = self._find_guest_posting_sites(niche)
        for site in guest_post_sites:
            opportunities.append({
                "type": "guest_post",
                "domain": site["domain"],
                "da": site.get("domain_authority", 0),
                "relevance": site.get("relevance", 0),
                "difficulty": "medium",
                "potential_value": self._calculate_link_value(site)
            })

        # 2. Resource pages
        resource_pages = self._find_resource_pages(niche)
        for page in resource_pages:
            opportunities.append({
                "type": "resource_page",
                "url": page["url"],
                "da": page.get("domain_authority", 0),
                "difficulty": "easy",
                "potential_value": self._calculate_link_value(page)
            })

        # 3. Broken link opportunities
        broken_links = self._find_broken_link_opportunities(niche)
        for link in broken_links:
            opportunities.append({
                "type": "broken_link",
                "url": link["url"],
                "broken_url": link["broken_url"],
                "difficulty": "medium",
                "potential_value": "high"
            })

        # 4. Unlinked mentions
        mentions = self._find_unlinked_mentions(domain)
        for mention in mentions:
            opportunities.append({
                "type": "unlinked_mention",
                "url": mention["url"],
                "difficulty": "easy",
                "potential_value": "medium"
            })

        # Сортируем по ценности
        opportunities.sort(key=lambda x: self._opportunity_score(x), reverse=True)

        logger.info(f"Found {len(opportunities)} link building opportunities")

        return opportunities[:50]  # Top 50

    def analyze_link_quality(self, donor_url: str) -> Dict[str, Any]:
        """Анализирует качество потенциального донора."""
        if not self.crawler:
            return {"error": "Crawler not available"}

        # Парсим страницу донора
        page_data = self.crawler.fetch_page(donor_url)

        analysis = {
            "url": donor_url,
            "quality_score": 0,
            "metrics": {},
            "red_flags": [],
            "green_flags": [],
            "recommendation": ""
        }

        # Метрики
        metrics = {
            "domain_authority": self._estimate_domain_authority(donor_url),
            "page_authority": self._estimate_page_authority(page_data),
            "spam_score": self._calculate_spam_score(page_data),
            "outbound_links": len(page_data.get("external_links", [])),
            "content_quality": self._assess_content_quality(page_data),
            "relevance": self._calculate_relevance(page_data)
        }

        analysis["metrics"] = metrics

        # Red flags
        if metrics["spam_score"] > 50:
            analysis["red_flags"].append("High spam score")

        if metrics["outbound_links"] > 100:
            analysis["red_flags"].append("Too many outbound links (link farm)")

        if metrics["content_quality"] < 0.3:
            analysis["red_flags"].append("Low content quality")

        # Green flags
        if metrics["domain_authority"] > 40:
            analysis["green_flags"].append("High domain authority")

        if metrics["relevance"] > 0.7:
            analysis["green_flags"].append("Highly relevant content")

        if metrics["spam_score"] < 20:
            analysis["green_flags"].append("Low spam score")

        # Quality score (0-100)
        quality_score = (
            metrics["domain_authority"] * 0.3 +
            metrics["page_authority"] * 0.2 +
            (100 - metrics["spam_score"]) * 0.2 +
            metrics["content_quality"] * 100 * 0.15 +
            metrics["relevance"] * 100 * 0.15
        )

        analysis["quality_score"] = round(quality_score, 1)

        # Recommendation
        if quality_score > 70:
            analysis["recommendation"] = "Excellent link opportunity - pursue actively"
        elif quality_score > 50:
            analysis["recommendation"] = "Good link opportunity - worth pursuing"
        elif quality_score > 30:
            analysis["recommendation"] = "Moderate quality - consider if easy to obtain"
        else:
            analysis["recommendation"] = "Low quality - avoid or be cautious"

        logger.info(f"Link quality analysis: {donor_url} - Score: {quality_score}")

        return analysis

    def generate_outreach_email(self, opportunity: Dict, your_content_url: str) -> str:
        """Генерирует персонализированный outreach email."""
        if not self.llm:
            return self._get_template_email(opportunity)

        opportunity_type = opportunity.get("type", "guest_post")

        prompt = f"""Generate a personalized outreach email for link building:

Opportunity type: {opportunity_type}
Target website: {opportunity.get('domain', opportunity.get('url'))}
Our content: {your_content_url}

Requirements:
- Personalized and friendly tone
- Clear value proposition
- Not salesy or pushy
- Professional
- Include specific benefit for them
- Max 150 words

Email:"""

        email = self.llm.generate(prompt, max_tokens=250)

        logger.info(f"Generated outreach email for {opportunity_type}")

        return email.strip()

    def find_broken_links(self, competitor_url: str) -> List[Dict]:
        """Находит broken links на сайте конкурента."""
        if not self.crawler:
            return []

        broken = []

        # Парсим страницу
        page_data = self.crawler.fetch_page(competitor_url)
        all_links = page_data.get("external_links", [])

        for link in all_links:
            # Проверяем доступность
            status = self._check_link_status(link)

            if status >= 400:  # 404, 410, etc.
                broken.append({
                    "url": link,
                    "status_code": status,
                    "found_on": competitor_url,
                    "opportunity": "Suggest your content as replacement"
                })

        logger.info(f"Found {len(broken)} broken links on {competitor_url}")

        return broken

    def analyze_competitor_backlinks(self, competitor_domain: str, your_domain: str) -> Dict[str, Any]:
        """Анализирует backlinks конкурента для поиска возможностей."""
        # В реальности нужен API Ahrefs/SEMrush/Majestic
        analysis = {
            "competitor": competitor_domain,
            "your_domain": your_domain,
            "opportunities": [],
            "easy_wins": [],
            "gap_analysis": {}
        }

        # Заглушка - в продакшене использовать backlink API
        logger.info(f"Analyzed competitor backlinks: {competitor_domain}")

        return analysis

    def create_linkable_asset_ideas(self, niche: str, competitors: List[str]) -> List[Dict]:
        """Генерирует идеи для linkable assets."""
        if not self.llm:
            return []

        prompt = f"""Generate 5 linkable asset ideas for this niche: {niche}

Competitors doing well: {', '.join(competitors[:3])}

For each idea provide:
1. Asset type (infographic, tool, guide, etc.)
2. Title
3. Brief description
4. Why it would attract links

Format as numbered list."""

        ideas_text = self.llm.generate(prompt, max_tokens=500)

        # Парсим идеи
        ideas = self._parse_asset_ideas(ideas_text)

        logger.info(f"Generated {len(ideas)} linkable asset ideas")

        return ideas

    # Helper methods

    def _find_guest_posting_sites(self, niche: str) -> List[Dict]:
        """Находит сайты, принимающие guest posts."""
        # В реальности использовать поиск Google с footprints:
        # "niche" + "write for us"
        # "niche" + "guest post"
        # "niche" + "contribute"
        return []

    def _find_resource_pages(self, niche: str) -> List[Dict]:
        """Находит resource pages в нише."""
        # Google footprints:
        # "niche" + "resources"
        # "niche" + "useful links"
        # inurl:resources "niche"
        return []

    def _find_broken_link_opportunities(self, niche: str) -> List[Dict]:
        """Находит broken link opportunities."""
        return []

    def _find_unlinked_mentions(self, domain: str) -> List[Dict]:
        """Находит упоминания домена без ссылки."""
        # Использовать Google Alerts или API
        return []

    def _calculate_link_value(self, site: Dict) -> str:
        """Оценивает ценность ссылки."""
        da = site.get("domain_authority", 0)

        if da > 60:
            return "very_high"
        elif da > 40:
            return "high"
        elif da > 20:
            return "medium"
        else:
            return "low"

    def _opportunity_score(self, opportunity: Dict) -> float:
        """Рассчитывает score возможности."""
        value_scores = {
            "very_high": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }

        difficulty_scores = {
            "easy": 3,
            "medium": 2,
            "hard": 1
        }

        value = value_scores.get(opportunity.get("potential_value", "low"), 1)
        difficulty = difficulty_scores.get(opportunity.get("difficulty", "hard"), 1)

        return value * difficulty

    def _estimate_domain_authority(self, url: str) -> int:
        """Оценивает Domain Authority."""
        # В реальности использовать API Moz или аналоги
        import random
        return random.randint(20, 80)

    def _estimate_page_authority(self, page_data: Dict) -> int:
        """Оценивает Page Authority."""
        import random
        return random.randint(15, 70)

    def _calculate_spam_score(self, page_data: Dict) -> int:
        """Рассчитывает spam score."""
        spam_indicators = 0

        # Проверяем индикаторы спама
        text = page_data.get("text", "").lower()

        if len(page_data.get("external_links", [])) > 100:
            spam_indicators += 20

        spam_words = ["casino", "viagra", "porn", "xxx"]
        for word in spam_words:
            if word in text:
                spam_indicators += 30

        if len(text) < 300:
            spam_indicators += 10

        return min(spam_indicators, 100)

    def _assess_content_quality(self, page_data: Dict) -> float:
        """Оценивает качество контента."""
        text = page_data.get("text", "")
        word_count = len(text.split())

        quality = 0.0

        if word_count > 1000:
            quality += 0.3
        elif word_count > 500:
            quality += 0.2

        if len(page_data.get("images", [])) > 2:
            quality += 0.2

        if len(page_data.get("headings", [])) > 3:
            quality += 0.2

        if page_data.get("has_schema", False):
            quality += 0.3

        return min(quality, 1.0)

    def _calculate_relevance(self, page_data: Dict) -> float:
        """Рассчитывает релевантность."""
        # Упрощенная версия
        return 0.5

    def _check_link_status(self, url: str) -> int:
        """Проверяет HTTP status code ссылки."""
        try:
            import requests
            response = requests.head(url, timeout=5, allow_redirects=True)
            return response.status_code
        except:
            return 0

    def _get_template_email(self, opportunity: Dict) -> str:
        """Возвращает шаблонный email."""
        templates = {
            "guest_post": """Subject: Guest Post Contribution Idea

Hi [Name],

I came across your website and really enjoyed your content on [topic].

I'm reaching out because I'd love to contribute a guest post to your site. I have expertise in [niche] and think I could provide value to your readers.

Here are a few topic ideas:
- [Topic 1]
- [Topic 2]
- [Topic 3]

Would you be interested in a collaboration?

Best regards,
[Your Name]""",

            "broken_link": """Subject: Broken Link on Your Page

Hi [Name],

I was browsing your excellent resource page at [URL] and noticed that one of your links appears to be broken:

[Broken URL]

I recently published a comprehensive guide on this topic that might make a good replacement:
[Your URL]

Hope this helps!

Best,
[Your Name]""",

            "resource_page": """Subject: Resource Suggestion for Your Page

Hi [Name],

I really appreciate the resource list you've curated at [URL].

I recently created [type of content] that I think would be a valuable addition:
[Your URL]

It covers [brief description] and has been well-received by the community.

Would you consider adding it to your list?

Thank you,
[Your Name]"""
        }

        opportunity_type = opportunity.get("type", "guest_post")
        return templates.get(opportunity_type, templates["guest_post"])

    def _parse_asset_ideas(self, ideas_text: str) -> List[Dict]:
        """Парсит идеи из текста LLM."""
        ideas = []

        # Простой парсинг - можно улучшить
        sections = ideas_text.split("\n\n")

        for section in sections:
            if section.strip():
                ideas.append({
                    "description": section.strip(),
                    "estimated_value": "medium"
                })

        return ideas
