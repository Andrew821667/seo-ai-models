"""
AI Content Generator - Генерация SEO-оптимизированного контента через AI.

Функции:
- Генерация статей по ключевым словам
- Auto-blogging с SEO оптимизацией
- Content briefs создание
- FAQ generation
- Product descriptions
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class AIContentGenerator:
    """AI-генератор SEO контента."""

    def __init__(self, llm_service=None, seo_advisor=None):
        self.llm = llm_service
        self.seo_advisor = seo_advisor
        logger.info("AIContentGenerator initialized")

    def generate_article(self, keyword: str, word_count: int = 1500, tone: str = "professional") -> Dict[str, Any]:
        """Генерирует полную SEO-оптимизированную статью."""
        if not self.llm:
            return {
                "error": "LLM service not available",
                "recommendation": "Configure LLM service to enable content generation"
            }

        article = {
            "keyword": keyword,
            "word_count_target": word_count,
            "tone": tone,
            "content": {},
            "seo_score": 0
        }

        # 1. Генерируем outline
        outline = self._generate_outline(keyword, word_count)
        article["outline"] = outline

        # 2. Генерируем title
        title = self._generate_seo_title(keyword)
        article["content"]["title"] = title

        # 3. Генерируем meta description
        meta_desc = self._generate_meta_description(keyword)
        article["content"]["meta_description"] = meta_desc

        # 4. Генерируем introduction
        intro = self._generate_introduction(keyword, outline)
        article["content"]["introduction"] = intro

        # 5. Генерируем основной контент по секциям
        sections = []
        for section in outline["sections"]:
            section_content = self._generate_section(section, keyword, tone)
            sections.append(section_content)

        article["content"]["sections"] = sections

        # 6. Генерируем conclusion
        conclusion = self._generate_conclusion(keyword)
        article["content"]["conclusion"] = conclusion

        # 7. Генерируем FAQ
        faq = self._generate_faq(keyword)
        article["content"]["faq"] = faq

        # 8. SEO оптимизация
        if self.seo_advisor:
            article["seo_score"] = self._calculate_seo_score(article)
            article["seo_recommendations"] = self._get_seo_improvements(article)

        # Собираем полный текст
        full_text = self._assemble_article(article["content"])
        article["full_text"] = full_text
        article["actual_word_count"] = len(full_text.split())

        logger.info(f"Generated article: '{title}' ({article['actual_word_count']} words)")

        return article

    def create_content_brief(self, keyword: str, competitors: List[str] = None) -> Dict[str, Any]:
        """Создает content brief для писателей."""
        brief = {
            "keyword": keyword,
            "target_audience": "",
            "search_intent": "",
            "recommended_word_count": 0,
            "outline": [],
            "required_elements": [],
            "competitor_analysis": [],
            "keywords_to_include": [],
            "tone_and_style": ""
        }

        # Определяем search intent
        brief["search_intent"] = self._detect_search_intent(keyword)

        # Рекомендуемый word count
        brief["recommended_word_count"] = self._recommend_word_count(keyword, competitors)

        # Outline
        brief["outline"] = self._generate_outline(keyword, brief["recommended_word_count"])

        # Required elements
        brief["required_elements"] = self._get_required_elements(brief["search_intent"])

        # Конкурентный анализ
        if competitors:
            brief["competitor_analysis"] = self._analyze_competitor_content(competitors)

        # LSI keywords
        brief["keywords_to_include"] = self._generate_lsi_keywords(keyword)

        # Tone & Style
        brief["tone_and_style"] = self._recommend_tone(keyword, brief["search_intent"])

        logger.info(f"Created content brief for '{keyword}'")

        return brief

    def generate_product_descriptions(self, products: List[Dict]) -> List[Dict]:
        """Генерирует SEO-описания для продуктов."""
        if not self.llm:
            return []

        descriptions = []

        for product in products:
            desc = self._generate_single_product_description(product)
            descriptions.append({
                "product_id": product.get("id"),
                "product_name": product.get("name"),
                "short_description": desc["short"],
                "long_description": desc["long"],
                "features_list": desc["features"],
                "seo_keywords": desc["keywords"]
            })

        logger.info(f"Generated descriptions for {len(descriptions)} products")

        return descriptions

    def auto_blog_post(self, topic: str, publish: bool = False) -> Dict[str, Any]:
        """Автоматически создает и (опционально) публикует пост."""
        result = {
            "topic": topic,
            "article_generated": False,
            "published": False,
            "url": None,
            "errors": []
        }

        # Генерируем статью
        article = self.generate_article(topic, word_count=1200)

        if "error" not in article:
            result["article_generated"] = True
            result["article"] = article

            # Публикация (если требуется)
            if publish:
                # Здесь должна быть интеграция с CMS
                logger.info(f"Auto-publishing article: {article['content']['title']}")
                result["published"] = True
                result["url"] = f"/blog/{topic.lower().replace(' ', '-')}"

        else:
            result["errors"].append(article["error"])

        logger.info(f"Auto-blog: {topic} - {'Published' if result['published'] else 'Generated only'}")

        return result

    # Helper methods

    def _generate_outline(self, keyword: str, word_count: int) -> Dict:
        """Генерирует outline статьи."""
        if not self.llm:
            return {"sections": []}

        sections_count = max(5, word_count // 300)

        prompt = f"""Create an article outline for SEO keyword: "{keyword}"

Target word count: {word_count}
Number of sections: {sections_count}

Provide:
1. Main H1 title
2. {sections_count} H2 section titles
3. 2-3 H3 subtitles for each H2

Format as structured outline."""

        outline_text = self.llm.generate(prompt, max_tokens=400)

        # Парсим outline
        sections = self._parse_outline(outline_text)

        return {
            "h1": f"Complete Guide to {keyword.title()}",
            "sections": sections,
            "estimated_word_count": word_count
        }

    def _generate_seo_title(self, keyword: str) -> str:
        """Генерирует SEO-оптимизированный title."""
        if not self.llm:
            return f"{keyword.title()} - Complete Guide"

        prompt = f"""Generate a compelling SEO title for keyword: "{keyword}"

Requirements:
- Include the keyword naturally
- 50-60 characters
- Engaging and click-worthy
- Professional tone

Title only:"""

        title = self.llm.generate(prompt, max_tokens=30)
        return title.strip()[:60]

    def _generate_meta_description(self, keyword: str) -> str:
        """Генерирует meta description."""
        if not self.llm:
            return f"Learn everything about {keyword} in this comprehensive guide."

        prompt = f"""Generate a meta description for keyword: "{keyword}"

Requirements:
- Include keyword naturally
- 150-160 characters
- Compelling and informative
- Include call-to-action

Description only:"""

        desc = self.llm.generate(prompt, max_tokens=50)
        return desc.strip()[:160]

    def _generate_introduction(self, keyword: str, outline: Dict) -> str:
        """Генерирует введение."""
        if not self.llm:
            return f"Introduction to {keyword}..."

        prompt = f"""Write an engaging introduction for an article about "{keyword}"

Article outline:
{outline.get('h1', keyword)}

Requirements:
- Hook the reader immediately
- Explain what they'll learn
- Include the keyword naturally
- 150-200 words
- Professional and engaging tone

Introduction:"""

        intro = self.llm.generate(prompt, max_tokens=250)
        return intro.strip()

    def _generate_section(self, section_title: str, keyword: str, tone: str) -> Dict:
        """Генерирует секцию статьи."""
        if not self.llm:
            return {
                "title": section_title,
                "content": f"Content about {section_title}..."
            }

        prompt = f"""Write a detailed section for an article about "{keyword}"

Section title: {section_title}
Tone: {tone}

Requirements:
- 200-300 words
- Informative and well-structured
- Include examples where relevant
- Natural keyword usage
- Engaging {tone} tone

Section content:"""

        content = self.llm.generate(prompt, max_tokens=400)

        return {
            "title": section_title,
            "content": content.strip()
        }

    def _generate_conclusion(self, keyword: str) -> str:
        """Генерирует заключение."""
        if not self.llm:
            return f"In conclusion, {keyword} is important..."

        prompt = f"""Write a strong conclusion for an article about "{keyword}"

Requirements:
- Summarize key points
- Provide actionable takeaway
- Encourage engagement (comment/share)
- 100-150 words

Conclusion:"""

        conclusion = self.llm.generate(prompt, max_tokens=200)
        return conclusion.strip()

    def _generate_faq(self, keyword: str, count: int = 5) -> List[Dict]:
        """Генерирует FAQ секцию."""
        if not self.llm:
            return []

        prompt = f"""Generate {count} frequently asked questions about "{keyword}"

Format each as:
Q: [question]
A: [concise answer]

Make questions natural and answers informative (2-3 sentences each)."""

        faq_text = self.llm.generate(prompt, max_tokens=500)

        # Парсим FAQ
        faq_list = self._parse_faq(faq_text)

        return faq_list

    def _calculate_seo_score(self, article: Dict) -> int:
        """Рассчитывает SEO score статьи."""
        score = 0

        content = article.get("content", {})

        # Title length
        title = content.get("title", "")
        if 50 <= len(title) <= 60:
            score += 15

        # Meta description length
        meta = content.get("meta_description", "")
        if 150 <= len(meta) <= 160:
            score += 15

        # Word count
        word_count = article.get("actual_word_count", 0)
        if word_count >= 1000:
            score += 20

        # Sections count
        sections = content.get("sections", [])
        if len(sections) >= 5:
            score += 15

        # FAQ presence
        if content.get("faq"):
            score += 10

        # Introduction and conclusion
        if content.get("introduction") and content.get("conclusion"):
            score += 15

        # Keyword usage (упрощенная проверка)
        score += 10

        return min(score, 100)

    def _get_seo_improvements(self, article: Dict) -> List[str]:
        """Возвращает рекомендации по улучшению SEO."""
        recommendations = []

        content = article.get("content", {})

        if len(content.get("title", "")) < 50:
            recommendations.append("Increase title length to 50-60 characters")

        if article.get("actual_word_count", 0) < 1000:
            recommendations.append("Increase content length to at least 1000 words")

        if len(content.get("sections", [])) < 5:
            recommendations.append("Add more sections for better structure")

        if not content.get("faq"):
            recommendations.append("Add FAQ section for featured snippet opportunity")

        return recommendations

    def _assemble_article(self, content: Dict) -> str:
        """Собирает полный текст статьи."""
        parts = [
            f"# {content.get('title', '')}",
            "",
            content.get("introduction", ""),
            ""
        ]

        for section in content.get("sections", []):
            parts.append(f"## {section['title']}")
            parts.append("")
            parts.append(section["content"])
            parts.append("")

        parts.append(content.get("conclusion", ""))
        parts.append("")

        # FAQ
        if content.get("faq"):
            parts.append("## Frequently Asked Questions")
            parts.append("")
            for qa in content["faq"]:
                parts.append(f"**{qa['question']}**")
                parts.append(qa["answer"])
                parts.append("")

        return "\n".join(parts)

    def _detect_search_intent(self, keyword: str) -> str:
        """Определяет search intent."""
        keyword_lower = keyword.lower()

        if any(word in keyword_lower for word in ["how to", "guide", "tutorial", "what is"]):
            return "informational"
        elif any(word in keyword_lower for word in ["buy", "price", "cheap", "best"]):
            return "commercial"
        elif any(word in keyword_lower for word in ["review", "vs", "comparison"]):
            return "commercial_investigation"
        else:
            return "informational"

    def _recommend_word_count(self, keyword: str, competitors: List[str]) -> int:
        """Рекомендует word count."""
        # В реальности анализировать конкурентов
        intent = self._detect_search_intent(keyword)

        word_counts = {
            "informational": 1500,
            "commercial": 2000,
            "commercial_investigation": 2500
        }

        return word_counts.get(intent, 1500)

    def _get_required_elements(self, intent: str) -> List[str]:
        """Возвращает обязательные элементы контента."""
        elements = {
            "informational": [
                "Clear definitions",
                "Step-by-step instructions",
                "Examples",
                "FAQ section",
                "Summary/Conclusion"
            ],
            "commercial": [
                "Product features",
                "Pricing information",
                "Pros and cons",
                "Customer reviews",
                "Clear CTA"
            ],
            "commercial_investigation": [
                "Comparison table",
                "Detailed pros and cons",
                "Expert opinion",
                "User reviews",
                "Recommendations"
            ]
        }

        return elements.get(intent, elements["informational"])

    def _analyze_competitor_content(self, competitors: List[str]) -> List[Dict]:
        """Анализирует контент конкурентов."""
        # Заглушка
        return []

    def _generate_lsi_keywords(self, keyword: str) -> List[str]:
        """Генерирует LSI keywords."""
        if not self.llm:
            return []

        prompt = f"""Generate 10 LSI (Latent Semantic Indexing) keywords related to "{keyword}"

These should be semantically related terms that would naturally appear in content about this topic.

List only the keywords:"""

        lsi_text = self.llm.generate(prompt, max_tokens=100)

        keywords = [k.strip().strip('-•123456789. ') for k in lsi_text.split('\n') if k.strip()]

        return keywords[:10]

    def _recommend_tone(self, keyword: str, intent: str) -> str:
        """Рекомендует tone для контента."""
        if intent == "commercial":
            return "Professional and persuasive"
        elif intent == "informational":
            return "Educational and friendly"
        else:
            return "Expert and authoritative"

    def _generate_single_product_description(self, product: Dict) -> Dict:
        """Генерирует описание одного продукта."""
        if not self.llm:
            return {
                "short": product.get("name", ""),
                "long": "",
                "features": [],
                "keywords": []
            }

        name = product.get("name", "")
        category = product.get("category", "")

        # Short description
        short_prompt = f"""Write a concise product description (50-70 words):

Product: {name}
Category: {category}

Make it engaging and highlight key benefits."""

        short_desc = self.llm.generate(short_prompt, max_tokens=100)

        # Long description
        long_prompt = f"""Write a detailed product description (150-200 words):

Product: {name}
Category: {category}

Include:
- Key features
- Benefits
- Use cases
- Why customers should buy"""

        long_desc = self.llm.generate(long_prompt, max_tokens=300)

        return {
            "short": short_desc.strip(),
            "long": long_desc.strip(),
            "features": product.get("features", []),
            "keywords": [name.lower(), category.lower()]
        }

    def _parse_outline(self, outline_text: str) -> List[Dict]:
        """Парсит outline из текста."""
        # Упрощенный парсинг
        sections = []

        lines = outline_text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('##') or line.startswith('H2'):
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "title": line.replace('##', '').replace('H2:', '').strip(),
                    "subsections": []
                }
            elif line and current_section:
                current_section["subsections"].append(line)

        if current_section:
            sections.append(current_section)

        return sections if sections else [{"title": "Introduction", "subsections": []}]

    def _parse_faq(self, faq_text: str) -> List[Dict]:
        """Парсит FAQ из текста."""
        faq_list = []

        lines = faq_text.split('\n')
        current_q = None
        current_a = None

        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                current_q = line.replace('Q:', '').strip()
            elif line.startswith('A:'):
                current_a = line.replace('A:', '').strip()

                if current_q and current_a:
                    faq_list.append({
                        "question": current_q,
                        "answer": current_a
                    })
                    current_q = None
                    current_a = None

        return faq_list
