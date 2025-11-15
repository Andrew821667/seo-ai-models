"""
CRO Integration - Интеграция Conversion Rate Optimization с SEO.

Функции:
- Анализ конверсионных элементов
- A/B тестирование рекомендации
- Heatmap анализ
- User journey optimization
- CTA optimization
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class CROIntegration:
    """Интеграция CRO и SEO."""

    def __init__(self, analytics_connector=None, llm_service=None):
        self.analytics = analytics_connector
        self.llm = llm_service
        logger.info("CROIntegration initialized")

    def analyze_conversion_elements(self, page_content: Dict) -> Dict[str, Any]:
        """Анализирует конверсионные элементы на странице."""
        analysis = {
            "page_url": page_content.get("url", ""),
            "conversion_score": 0,
            "elements_found": {},
            "missing_elements": [],
            "recommendations": []
        }

        score = 0

        # CTA Buttons
        cta_count = page_content.get("cta_buttons", 0)
        analysis["elements_found"]["cta_buttons"] = cta_count

        if cta_count == 0:
            analysis["missing_elements"].append("Call-to-Action buttons")
            analysis["recommendations"].append({
                "element": "CTA",
                "priority": "high",
                "action": "Add clear, prominent CTA button above the fold"
            })
        elif cta_count > 0:
            score += 20

        # Forms
        has_form = page_content.get("has_form", False)
        analysis["elements_found"]["contact_form"] = has_form

        if has_form:
            score += 15
            # Проверяем количество полей
            form_fields = page_content.get("form_fields", [])
            if len(form_fields) > 5:
                analysis["recommendations"].append({
                    "element": "Form",
                    "priority": "medium",
                    "action": f"Reduce form fields from {len(form_fields)} to 3-5 to increase completion rate"
                })
        else:
            if page_content.get("page_type") == "landing":
                analysis["missing_elements"].append("Contact/Lead form")

        # Trust Signals
        trust_signals = {
            "testimonials": page_content.get("has_testimonials", False),
            "reviews": page_content.get("has_reviews", False),
            "security_badges": page_content.get("has_security_badges", False),
            "guarantees": page_content.get("has_guarantees", False)
        }

        analysis["elements_found"]["trust_signals"] = trust_signals

        trust_count = sum(1 for v in trust_signals.values() if v)
        score += trust_count * 10

        if trust_count < 2:
            analysis["recommendations"].append({
                "element": "Trust Signals",
                "priority": "high",
                "action": "Add at least 2 trust elements (testimonials, reviews, badges, guarantees)"
            })

        # Social Proof
        has_social_proof = page_content.get("has_social_proof", False)
        if has_social_proof:
            score += 15
        else:
            analysis["recommendations"].append({
                "element": "Social Proof",
                "priority": "medium",
                "action": "Add social proof (customer count, ratings, case studies)"
            })

        # Value Proposition
        has_clear_value_prop = self._check_value_proposition(page_content)
        if has_clear_value_prop:
            score += 20
        else:
            analysis["recommendations"].append({
                "element": "Value Proposition",
                "priority": "high",
                "action": "Add clear, compelling value proposition in headline"
            })

        # Urgency/Scarcity
        has_urgency = page_content.get("has_urgency_elements", False)
        if has_urgency:
            score += 10

        # Visual Hierarchy
        visual_score = self._assess_visual_hierarchy(page_content)
        score += visual_score

        analysis["conversion_score"] = min(score, 100)

        logger.info(f"Conversion analysis: {analysis['conversion_score']}/100")

        return analysis

    def suggest_ab_tests(self, page_content: Dict, current_conversion_rate: float) -> List[Dict]:
        """Предлагает A/B тесты для улучшения конверсии."""
        tests = []

        # Test 1: CTA variations
        tests.append({
            "test_name": "CTA Button Color & Text",
            "hypothesis": "Changing CTA color and text will increase clicks",
            "variants": [
                {"element": "button", "change": "Color: Green → Orange", "expected_lift": "15-25%"},
                {"element": "button", "change": "Text: 'Submit' → 'Get Started Free'", "expected_lift": "10-20%"}
            ],
            "priority": "high",
            "estimated_runtime": "2-3 weeks",
            "required_sample_size": self._calculate_sample_size(current_conversion_rate, 0.1)
        })

        # Test 2: Headline
        tests.append({
            "test_name": "Value Proposition Headline",
            "hypothesis": "Benefit-focused headline will increase engagement",
            "variants": [
                {"element": "h1", "change": "Feature-based → Benefit-based", "expected_lift": "8-15%"}
            ],
            "priority": "high",
            "estimated_runtime": "2 weeks",
            "required_sample_size": self._calculate_sample_size(current_conversion_rate, 0.08)
        })

        # Test 3: Form length
        if page_content.get("has_form"):
            tests.append({
                "test_name": "Form Field Reduction",
                "hypothesis": "Fewer form fields will increase completion rate",
                "variants": [
                    {"element": "form", "change": "Remove optional fields", "expected_lift": "20-30%"}
                ],
                "priority": "medium",
                "estimated_runtime": "2-3 weeks",
                "required_sample_size": self._calculate_sample_size(current_conversion_rate, 0.15)
            })

        # Test 4: Trust elements placement
        tests.append({
            "test_name": "Trust Signals Position",
            "hypothesis": "Trust elements above fold will increase conversion",
            "variants": [
                {"element": "testimonials", "change": "Move from footer to above CTA", "expected_lift": "10-18%"}
            ],
            "priority": "medium",
            "estimated_runtime": "2 weeks",
            "required_sample_size": self._calculate_sample_size(current_conversion_rate, 0.1)
        })

        # Test 5: Images
        tests.append({
            "test_name": "Hero Image Variation",
            "hypothesis": "People-focused images increase emotional connection",
            "variants": [
                {"element": "hero_image", "change": "Product image → People using product", "expected_lift": "5-12%"}
            ],
            "priority": "low",
            "estimated_runtime": "3 weeks",
            "required_sample_size": self._calculate_sample_size(current_conversion_rate, 0.05)
        })

        # Сортируем по приоритету
        priority_order = {"high": 3, "medium": 2, "low": 1}
        tests.sort(key=lambda x: priority_order[x["priority"]], reverse=True)

        logger.info(f"Generated {len(tests)} A/B test suggestions")

        return tests

    def optimize_user_journey(self, entry_page: str, goal_page: str) -> Dict[str, Any]:
        """Оптимизирует путь пользователя от входа до конверсии."""
        optimization = {
            "entry_page": entry_page,
            "goal_page": goal_page,
            "current_funnel": [],
            "drop_off_points": [],
            "recommendations": []
        }

        # Анализируем текущий путь
        funnel = self._get_user_funnel(entry_page, goal_page)
        optimization["current_funnel"] = funnel

        # Находим точки отсева
        for i, step in enumerate(funnel):
            if step.get("drop_off_rate", 0) > 30:
                optimization["drop_off_points"].append({
                    "step": step["page"],
                    "drop_off_rate": step["drop_off_rate"],
                    "severity": "high" if step["drop_off_rate"] > 50 else "medium"
                })

                # Рекомендации для этого шага
                optimization["recommendations"].append({
                    "page": step["page"],
                    "issue": f"{step['drop_off_rate']}% drop-off rate",
                    "actions": [
                        "Simplify page design",
                        "Add progress indicators",
                        "Reduce form fields",
                        "Add trust signals",
                        "Improve page load speed"
                    ]
                })

        # Общие рекомендации
        if len(funnel) > 4:
            optimization["recommendations"].append({
                "page": "Overall",
                "issue": "Funnel too long",
                "actions": [
                    f"Reduce funnel from {len(funnel)} to 3 steps",
                    "Combine similar pages",
                    "Remove unnecessary steps"
                ]
            })

        logger.info(f"User journey optimization: {len(optimization['drop_off_points'])} drop-off points")

        return optimization

    def generate_cta_variants(self, page_context: Dict, count: int = 5) -> List[str]:
        """Генерирует варианты CTA текстов."""
        if not self.llm:
            return self._get_default_cta_variants()

        page_type = page_context.get("page_type", "general")
        product = page_context.get("product_name", "product")

        prompt = f"""Generate {count} compelling CTA button texts for a {page_type} page:

Product/Service: {product}
Page context: {page_context.get('description', '')}

Requirements:
- Action-oriented
- Clear benefit
- Creates urgency
- Max 4 words
- Engaging and persuasive

CTA variants:"""

        variants_text = self.llm.generate(prompt, max_tokens=150)

        # Парсим варианты
        variants = [line.strip().strip('-•123456789. ') for line in variants_text.split('\n') if line.strip()]

        logger.info(f"Generated {len(variants)} CTA variants")

        return variants[:count]

    def analyze_heatmap_recommendations(self, heatmap_data: Dict) -> Dict[str, Any]:
        """Анализирует heatmap данные и дает рекомендации."""
        recommendations = {
            "hot_zones": [],
            "cold_zones": [],
            "actions": []
        }

        # Анализируем горячие зоны
        for zone in heatmap_data.get("hot_zones", []):
            recommendations["hot_zones"].append({
                "area": zone["area"],
                "clicks": zone["clicks"],
                "recommendation": "Optimize this high-engagement area with conversion elements"
            })

            if zone["area"] == "top_right" and not zone.get("has_cta"):
                recommendations["actions"].append({
                    "priority": "high",
                    "action": "Add CTA button in top-right hot zone",
                    "expected_impact": "15-20% increase in CTA clicks"
                })

        # Анализируем холодные зоны
        for zone in heatmap_data.get("cold_zones", []):
            recommendations["cold_zones"].append({
                "area": zone["area"],
                "clicks": zone["clicks"],
                "recommendation": f"Low engagement - consider removing or redesigning {zone['area']}"
            })

            if zone.get("has_important_content"):
                recommendations["actions"].append({
                    "priority": "medium",
                    "action": f"Move important content from {zone['area']} to higher-engagement area",
                    "expected_impact": "10-15% better content visibility"
                })

        logger.info(f"Heatmap analysis: {len(recommendations['actions'])} recommendations")

        return recommendations

    # Helper methods

    def _check_value_proposition(self, page_content: Dict) -> bool:
        """Проверяет наличие четкого value proposition."""
        title = page_content.get("title", "").lower()
        h1 = page_content.get("h1", "").lower()

        # Упрощенная проверка
        value_words = ["best", "free", "easy", "fast", "save", "guarantee", "professional"]

        return any(word in title or word in h1 for word in value_words)

    def _assess_visual_hierarchy(self, page_content: Dict) -> int:
        """Оценивает визуальную иерархию."""
        score = 0

        # Наличие H1
        if page_content.get("h1"):
            score += 5

        # Структура заголовков
        if len(page_content.get("headings", [])) >= 3:
            score += 5

        # Изображения
        if len(page_content.get("images", [])) >= 2:
            score += 5

        # Контраст (упрощенно)
        if page_content.get("has_high_contrast", False):
            score += 5

        return score

    def _calculate_sample_size(self, baseline_rate: float, minimum_detectable_effect: float) -> int:
        """Рассчитывает необходимый размер выборки для A/B теста."""
        # Упрощенная формула
        # В реальности использовать статистические калькуляторы

        if baseline_rate <= 0 or baseline_rate >= 1:
            return 1000

        # Примерная формула
        sample_per_variant = int(16 * (baseline_rate * (1 - baseline_rate)) / (minimum_detectable_effect ** 2))

        return sample_per_variant * 2  # Для обоих вариантов

    def _get_user_funnel(self, entry: str, goal: str) -> List[Dict]:
        """Получает воронку пользователя."""
        # Заглушка - в реальности получать из Analytics
        funnel = [
            {"page": entry, "visitors": 1000, "drop_off_rate": 0},
            {"page": "/step1", "visitors": 700, "drop_off_rate": 30},
            {"page": "/step2", "visitors": 500, "drop_off_rate": 29},
            {"page": goal, "visitors": 400, "drop_off_rate": 20}
        ]

        return funnel

    def _get_default_cta_variants(self) -> List[str]:
        """Возвращает дефолтные варианты CTA."""
        return [
            "Get Started Free",
            "Start Your Trial",
            "Download Now",
            "Sign Up Free",
            "Request a Demo"
        ]
