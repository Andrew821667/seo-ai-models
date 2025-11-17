"""
Enhanced SEO Advisor - Расширенный анализатор с автоматическими исправлениями.

Интегрирует:
- Базовый SEOAdvisor
- AutoFix Engine для автоматических исправлений
- Все 10 модулей улучшений

Workflow:
1. Analyze (базовый анализ)
2. Detect issues (выявление проблем)
3. Auto-fix (автоматическое исправление)
4. Verify (проверка результатов)
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Base advisor
from .seo_advisor.advisor import SEOAdvisor, SEOAnalysisReport

# AutoFix Engine
from ..autofix.engine import AutoFixEngine, FixComplexity
from ..autofix.fixers import (
    MetaTagsFixer,
    ImageAltTagsFixer,
    ContentRefreshFixer,
    SchemaMarkupFixer,
    InternalLinksFixer,
)

# Improvement modules (optional)
try:
    from ..improvements import (
        ContentRefreshAutomation,
        VisualContentAnalyzer,
        IntentBasedOptimizer,
        CompetitorMonitor,
        InternationalSEO,
        LinkBuildingAssistant,
        PredictiveAnalytics,
        CROIntegration,
        MobileOptimizer,
        AIContentGenerator,
    )

    IMPROVEMENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some improvement modules not available: {e}")
    IMPROVEMENTS_AVAILABLE = False
    # Create dummy classes
    ContentRefreshAutomation = None
    VisualContentAnalyzer = None
    IntentBasedOptimizer = None
    CompetitorMonitor = None
    InternationalSEO = None
    LinkBuildingAssistant = None
    PredictiveAnalytics = None
    CROIntegration = None
    MobileOptimizer = None
    AIContentGenerator = None

logger = logging.getLogger(__name__)


class EnhancedSEOAdvisor:
    """
    Расширенный SEO советник с автоматическими исправлениями.

    Комбинирует традиционный SEO анализ с AI-powered автоисправлениями
    и продвинутыми модулями оптимизации.
    """

    def __init__(
        self,
        industry: str = "default",
        auto_execute: bool = True,
        cms_connector=None,
        llm_service=None,
        crawler=None,
        analytics_connector=None,
    ):
        """
        Инициализация Enhanced SEO Advisor.

        Args:
            industry: Отрасль для специфичного анализа
            auto_execute: Автоматически выполнять простые исправления
            cms_connector: Коннектор к CMS
            llm_service: LLM сервис для AI генерации
            crawler: Web crawler
            analytics_connector: Аналитика
        """
        # Базовый advisor
        self.base_advisor = SEOAdvisor(industry=industry)

        # AutoFix Engine
        self.autofix_engine = AutoFixEngine(cms_connector=cms_connector, auto_execute=auto_execute)

        # Register fixers
        self._register_fixers(llm_service, cms_connector)

        # Improvement modules (только если доступны)
        if IMPROVEMENTS_AVAILABLE:
            self.content_refresh = ContentRefreshAutomation(
                cms_connector=cms_connector,
                llm_service=llm_service,
                autofix_engine=self.autofix_engine,
            )

            self.visual_analyzer = VisualContentAnalyzer(
                llm_service=llm_service, autofix_engine=self.autofix_engine
            )

            self.intent_optimizer = IntentBasedOptimizer(
                llm_service=llm_service, autofix_engine=self.autofix_engine
            )

            self.competitor_monitor = CompetitorMonitor(crawler=crawler, llm_service=llm_service)

            self.international_seo = InternationalSEO(
                llm_service=llm_service, cms_connector=cms_connector
            )

            self.link_building = LinkBuildingAssistant(llm_service=llm_service, crawler=crawler)

            self.predictive = PredictiveAnalytics(analytics_connector=analytics_connector)

            self.cro = CROIntegration(
                analytics_connector=analytics_connector, llm_service=llm_service
            )

            self.mobile_optimizer = MobileOptimizer(crawler=crawler)

            self.ai_content = AIContentGenerator(
                llm_service=llm_service, seo_advisor=self.base_advisor
            )
        else:
            # Stub implementations
            self.content_refresh = None
            self.visual_analyzer = None
            self.intent_optimizer = None
            self.competitor_monitor = None
            self.international_seo = None
            self.link_building = None
            self.predictive = None
            self.cro = None
            self.mobile_optimizer = None
            self.ai_content = None
            logger.warning("Improvement modules not available - running in basic mode")

        self.analysis_history = []

        logger.info(f"EnhancedSEOAdvisor initialized (auto_execute: {auto_execute})")

    def analyze_and_fix(
        self,
        url: str,
        content: str,
        keywords: List[str],
        auto_fix: bool = True,
        fix_complexity_limit: FixComplexity = FixComplexity.SIMPLE,
    ) -> Dict[str, Any]:
        """
        ПОЛНЫЙ АНАЛИЗ + АВТОМАТИЧЕСКИЕ ИСПРАВЛЕНИЯ.

        Workflow:
        1. Базовый SEO анализ
        2. Дополнительные проверки (visual, mobile, etc.)
        3. Автоматическое исправление найденных проблем
        4. Генерация финального отчета

        Args:
            url: URL страницы
            content: Текстовое содержимое
            keywords: Ключевые слова
            auto_fix: Автоматически исправлять проблемы
            fix_complexity_limit: Максимальная сложность автофиксов

        Returns:
            Dict с результатами анализа и исправлений
        """
        result = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "analysis": {},
            "issues_detected": [],
            "fixes_applied": [],
            "improvements_available": [],
            "overall_score": 0,
        }

        # 1. БАЗОВЫЙ SEO АНАЛИЗ
        logger.info(f"Running base SEO analysis for {url}")
        base_analysis = self.base_advisor.analyze_content(content, keywords)
        result["analysis"]["base_seo"] = self._serialize_analysis(base_analysis)

        # 2. ДОПОЛНИТЕЛЬНЫЕ АНАЛИЗЫ

        # Visual content
        page_content = self._extract_page_content(url, content)

        visual_analysis = self.visual_analyzer.analyze_images(page_content)
        result["analysis"]["visual"] = visual_analysis

        # Mobile optimization
        mobile_analysis = self.mobile_optimizer.analyze_mobile_friendliness(url)
        result["analysis"]["mobile"] = mobile_analysis

        # Core Web Vitals
        vitals = self.mobile_optimizer.analyze_core_web_vitals(url)
        result["analysis"]["core_web_vitals"] = vitals

        # Intent matching
        if keywords:
            intent_analysis = self.intent_optimizer.detect_intent(keywords[0])
            result["analysis"]["intent"] = intent_analysis

        # 3. СБОР ВСЕХ ПРОБЛЕМ
        issues = self._collect_issues(
            base_analysis, visual_analysis, mobile_analysis, vitals, page_content
        )
        result["issues_detected"] = issues

        # 4. АВТОМАТИЧЕСКИЕ ИСПРАВЛЕНИЯ (если включено)
        if auto_fix and issues:
            logger.info(f"Auto-fixing {len(issues)} issues...")

            # Создаем план исправлений
            fix_plan = self.autofix_engine.analyze_and_plan(
                self._convert_to_analysis_format(result["analysis"])
            )

            # Фильтруем по complexity limit
            allowed_fixes = [
                action
                for action in fix_plan
                if action.complexity.value <= fix_complexity_limit.value
            ]

            # Выполняем исправления
            if allowed_fixes:
                execution_result = self.autofix_engine.execute_plan(
                    allowed_fixes,
                    require_approval_for=[FixComplexity.COMPLEX, FixComplexity.CRITICAL],
                )

                result["fixes_applied"] = execution_result.get("executed", [])
                result["pending_approval"] = execution_result.get("pending_approval", [])

                logger.info(f"✅ Applied {len(result['fixes_applied'])} auto-fixes")

        # 5. ДОСТУПНЫЕ УЛУЧШЕНИЯ
        improvements = self._suggest_improvements(result["analysis"])
        result["improvements_available"] = improvements

        # 6. ОБЩИЙ SCORE
        result["overall_score"] = self._calculate_overall_score(result)

        # Сохраняем в историю
        self.analysis_history.append(result)

        logger.info(
            f"Analysis complete: {result['overall_score']}/100, {len(result['fixes_applied'])} fixes applied"
        )

        return result

    def get_content_refresh_candidates(self, threshold_days: int = 365) -> List[Dict]:
        """Находит контент, требующий обновления."""
        return self.content_refresh.identify_outdated_content(threshold_days)

    def monitor_competitors(self, domain: str, competitors: List[str], keywords: List[str]) -> Dict:
        """Мониторинг конкурентов."""
        results = {"domain": domain, "competitors": [], "keyword_gaps": [], "opportunities": []}

        for competitor in competitors:
            # Добавляем конкурента
            self.competitor_monitor.add_competitor(
                name=competitor, domain=competitor, keywords=keywords
            )

            # Gap analysis
            gaps = self.competitor_monitor.gap_analysis(domain, competitor)
            results["keyword_gaps"].extend(gaps.get("keyword_gaps", []))
            results["opportunities"].extend(gaps.get("opportunities", []))

        return results

    def generate_content(self, keyword: str, word_count: int = 1500) -> Dict:
        """Генерирует SEO-оптимизированный контент."""
        return self.ai_content.generate_article(keyword, word_count)

    def forecast_performance(self, domain: str, months_ahead: int = 3) -> Dict:
        """Прогнозирует SEO метрики."""
        return self.predictive.forecast_traffic(domain, months_ahead)

    def optimize_for_international(self, domain: str, target_countries: List[str]) -> Dict:
        """Оптимизация для международных рынков."""
        return self.international_seo.optimize_for_geo_targeting(domain, target_countries)

    def get_link_building_opportunities(self, domain: str, niche: str) -> List[Dict]:
        """Находит возможности для link building."""
        return self.link_building.find_link_opportunities(domain, niche)

    def analyze_conversion_optimization(self, page_content: Dict) -> Dict:
        """CRO анализ."""
        return self.cro.analyze_conversion_elements(page_content)

    # Helper methods

    def _register_fixers(self, llm_service, cms_connector):
        """Регистрирует все fixers в AutoFix Engine."""
        self.autofix_engine.register_action("missing_meta_tags", MetaTagsFixer(llm_service))

        self.autofix_engine.register_action("missing_alt_tags", ImageAltTagsFixer(llm_service))

        self.autofix_engine.register_action(
            "outdated_content", ContentRefreshFixer(cms_connector, llm_service)
        )

        self.autofix_engine.register_action("missing_schema", SchemaMarkupFixer())

        self.autofix_engine.register_action(
            "insufficient_internal_links", InternalLinksFixer(cms_connector)
        )

    def _serialize_analysis(self, analysis: SEOAnalysisReport) -> Dict:
        """Конвертирует SEOAnalysisReport в dict."""
        return {
            "timestamp": analysis.timestamp.isoformat(),
            "content_metrics": analysis.content_metrics,
            "keyword_analysis": analysis.keyword_analysis,
            "predicted_position": analysis.predicted_position,
            "feature_scores": analysis.feature_scores,
            "content_quality": {
                "scores": analysis.content_quality.content_scores,
                "strengths": analysis.content_quality.strengths,
                "weaknesses": analysis.content_quality.weaknesses,
                "improvements": analysis.content_quality.potential_improvements,
            },
            "recommendations": analysis.recommendations,
            "priorities": analysis.priorities,
        }

    def _extract_page_content(self, url: str, content: str) -> Dict:
        """Извлекает структурированные данные страницы."""
        # Упрощенная версия - в реальности парсить HTML
        return {
            "url": url,
            "text": content,
            "images": [],
            "headings": [],
            "title": "",
            "description": "",
        }

    def _collect_issues(
        self, base_analysis, visual_analysis, mobile_analysis, vitals, page_content
    ) -> List[Dict]:
        """Собирает все обнаруженные проблемы."""
        issues = []

        # Issues from base analysis
        if hasattr(base_analysis, "content_quality"):
            for weakness in base_analysis.content_quality.weaknesses:
                issues.append(
                    {
                        "type": "content_quality",
                        "severity": "medium",
                        "description": weakness,
                        "fixable": True,
                    }
                )

        # Visual issues
        if visual_analysis.get("missing_alt"):
            issues.append(
                {
                    "type": "missing_alt_tags",
                    "severity": "high",
                    "description": f"{len(visual_analysis['missing_alt'])} images without alt tags",
                    "fixable": True,
                    "auto_fixable": True,
                }
            )

        if visual_analysis.get("oversized"):
            issues.append(
                {
                    "type": "oversized_images",
                    "severity": "medium",
                    "description": f"{len(visual_analysis['oversized'])} oversized images",
                    "fixable": True,
                    "auto_fixable": True,
                }
            )

        # Mobile issues
        for issue in mobile_analysis.get("issues", []):
            issues.append(
                {
                    "type": "mobile_friendliness",
                    "severity": issue.get("priority", "medium"),
                    "description": issue.get("issue", ""),
                    "fixable": True,
                    "fix_instructions": issue.get("fix", ""),
                }
            )

        # Core Web Vitals issues
        for recommendation in vitals.get("recommendations", []):
            issues.append(
                {
                    "type": "core_web_vitals",
                    "severity": "high",
                    "description": recommendation.get("issue", ""),
                    "fixable": True,
                    "fixes": recommendation.get("fixes", []),
                }
            )

        return issues

    def _convert_to_analysis_format(self, analysis: Dict) -> Dict:
        """Конвертирует результаты анализа в формат для AutoFix."""
        # Упрощенная версия
        return {
            "missing_meta_tags": [],
            "missing_alt_tags": analysis.get("visual", {}).get("missing_alt", []),
            "oversized_images": analysis.get("visual", {}).get("oversized", []),
            "mobile_issues": analysis.get("mobile", {}).get("issues", []),
            "vitals_issues": analysis.get("core_web_vitals", {}).get("recommendations", []),
        }

    def _suggest_improvements(self, analysis: Dict) -> List[Dict]:
        """Предлагает доступные улучшения."""
        improvements = []

        # Content improvements
        base_analysis = analysis.get("base_seo", {})
        if base_analysis.get("recommendations"):
            for category, recs in base_analysis.get("recommendations", {}).items():
                for rec in recs:
                    improvements.append(
                        {
                            "category": category,
                            "improvement": rec,
                            "priority": "medium",
                            "module": "base_seo",
                        }
                    )

        # Visual improvements
        visual = analysis.get("visual", {})
        if visual.get("optimization_potential", 0) > 0:
            improvements.append(
                {
                    "category": "visual",
                    "improvement": f"Optimize images to save {visual['optimization_potential']}% size",
                    "priority": "medium",
                    "module": "visual_analyzer",
                }
            )

        # Mobile improvements
        mobile = analysis.get("mobile", {})
        if not mobile.get("mobile_friendly", True):
            improvements.append(
                {
                    "category": "mobile",
                    "improvement": "Improve mobile friendliness",
                    "priority": "high",
                    "module": "mobile_optimizer",
                }
            )

        return improvements

    def _calculate_overall_score(self, result: Dict) -> int:
        """Рассчитывает общий score."""
        scores = []

        # Base SEO score (примерная оценка)
        base = result["analysis"].get("base_seo", {})
        if base.get("predicted_position"):
            # Конвертируем позицию в score (позиция 1 = 100, позиция 100 = 0)
            position_score = max(0, 100 - result["analysis"]["base_seo"]["predicted_position"])
            scores.append(position_score)

        # Mobile score
        mobile = result["analysis"].get("mobile", {})
        if mobile.get("score") is not None:
            scores.append(mobile["score"])

        # Visual score (упрощенно)
        visual = result["analysis"].get("visual", {})
        if visual.get("missing_alt"):
            alt_score = max(0, 100 - len(visual["missing_alt"]) * 10)
            scores.append(alt_score)

        # Core Web Vitals
        vitals = result["analysis"].get("core_web_vitals", {})
        if vitals.get("passed"):
            scores.append(100)
        else:
            scores.append(50)

        # Средний score
        if scores:
            return round(sum(scores) / len(scores))
        else:
            return 70  # Default

    def get_full_report(self, url: str, content: str, keywords: List[str]) -> Dict:
        """Генерирует полный расширенный отчет."""
        # Запускаем полный анализ с исправлениями
        analysis = self.analyze_and_fix(url, content, keywords, auto_fix=True)

        # Добавляем дополнительные секции
        page_content = self._extract_page_content(url, content)

        report = {
            **analysis,
            "additional_insights": {
                "content_refresh": self.content_refresh.get_refresh_report(),
                "link_opportunities": self.link_building.find_link_opportunities(url, "general")[
                    :10
                ],
                "cro_analysis": self.cro.analyze_conversion_elements(page_content),
                "mobile_report": self.mobile_optimizer.generate_mobile_report(url),
            },
        }

        return report
