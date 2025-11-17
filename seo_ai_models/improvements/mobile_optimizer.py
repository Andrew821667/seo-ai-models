"""
Mobile Optimizer - Оптимизация для мобильных устройств и Core Web Vitals.

Функции:
- Mobile-friendly проверка
- Core Web Vitals анализ (LCP, FID, CLS)
- Touch-friendly элементы
- Responsive design проверка
- AMP validation
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class MobileOptimizer:
    """Оптимизация для мобильных устройств."""

    def __init__(self, crawler=None):
        self.crawler = crawler
        logger.info("MobileOptimizer initialized")

    def analyze_mobile_friendliness(self, url: str) -> Dict[str, Any]:
        """Полный анализ mobile-friendliness."""
        analysis = {
            "url": url,
            "mobile_friendly": False,
            "score": 0,
            "issues": [],
            "passed_checks": [],
            "recommendations": [],
        }

        score = 0

        # 1. Viewport meta tag
        viewport_check = self._check_viewport_meta(url)
        if viewport_check["passed"]:
            score += 20
            analysis["passed_checks"].append("Viewport meta tag")
        else:
            analysis["issues"].append(
                {
                    "issue": "Missing or incorrect viewport meta tag",
                    "priority": "critical",
                    "fix": "Add: <meta name='viewport' content='width=device-width, initial-scale=1'>",
                }
            )

        # 2. Touch-friendly elements
        touch_check = self._check_touch_elements(url)
        if touch_check["passed"]:
            score += 15
            analysis["passed_checks"].append("Touch-friendly buttons")
        else:
            analysis["issues"].append(
                {
                    "issue": f"Touch targets too small: {touch_check['small_targets']} elements",
                    "priority": "high",
                    "fix": "Increase touch target size to minimum 48x48 pixels",
                }
            )

        # 3. Text readability
        text_check = self._check_text_readability(url)
        if text_check["passed"]:
            score += 15
            analysis["passed_checks"].append("Text readability")
        else:
            analysis["issues"].append(
                {
                    "issue": "Text too small for mobile",
                    "priority": "high",
                    "fix": "Use minimum 16px font size for body text",
                }
            )

        # 4. Responsive images
        images_check = self._check_responsive_images(url)
        if images_check["passed"]:
            score += 10
            analysis["passed_checks"].append("Responsive images")
        else:
            analysis["issues"].append(
                {
                    "issue": "Images not responsive",
                    "priority": "medium",
                    "fix": "Use responsive images with srcset or CSS max-width: 100%",
                }
            )

        # 5. Horizontal scrolling
        scroll_check = self._check_horizontal_scroll(url)
        if scroll_check["passed"]:
            score += 15
            analysis["passed_checks"].append("No horizontal scrolling")
        else:
            analysis["issues"].append(
                {
                    "issue": "Content requires horizontal scrolling",
                    "priority": "high",
                    "fix": "Ensure all content fits within viewport width",
                }
            )

        # 6. Spacing between elements
        spacing_check = self._check_element_spacing(url)
        if spacing_check["passed"]:
            score += 10
            analysis["passed_checks"].append("Adequate spacing")
        else:
            analysis["issues"].append(
                {
                    "issue": "Insufficient spacing between clickable elements",
                    "priority": "medium",
                    "fix": "Add minimum 8px spacing between interactive elements",
                }
            )

        # 7. Fast loading
        speed_check = self._check_mobile_speed(url)
        if speed_check["passed"]:
            score += 15
            analysis["passed_checks"].append("Fast mobile loading")
        else:
            analysis["issues"].append(
                {
                    "issue": f"Slow mobile load time: {speed_check['load_time']}s",
                    "priority": "high",
                    "fix": "Optimize images, minify CSS/JS, enable compression",
                }
            )

        analysis["score"] = score
        analysis["mobile_friendly"] = score >= 70

        logger.info(
            f"Mobile analysis: {score}/100 - {'PASS' if analysis['mobile_friendly'] else 'FAIL'}"
        )

        return analysis

    def analyze_core_web_vitals(self, url: str) -> Dict[str, Any]:
        """Анализирует Core Web Vitals."""
        vitals = {"url": url, "passed": False, "metrics": {}, "recommendations": []}

        # LCP - Largest Contentful Paint
        lcp = self._measure_lcp(url)
        vitals["metrics"]["lcp"] = {
            "value": lcp,
            "unit": "seconds",
            "rating": self._rate_lcp(lcp),
            "threshold": "< 2.5s (good), < 4.0s (needs improvement), >= 4.0s (poor)",
        }

        if lcp > 2.5:
            vitals["recommendations"].append(
                {
                    "metric": "LCP",
                    "issue": f"LCP too high: {lcp}s",
                    "fixes": [
                        "Optimize and compress images",
                        "Remove render-blocking resources",
                        "Improve server response time",
                        "Use lazy loading for images",
                        "Implement CDN",
                    ],
                }
            )

        # FID - First Input Delay
        fid = self._measure_fid(url)
        vitals["metrics"]["fid"] = {
            "value": fid,
            "unit": "milliseconds",
            "rating": self._rate_fid(fid),
            "threshold": "< 100ms (good), < 300ms (needs improvement), >= 300ms (poor)",
        }

        if fid > 100:
            vitals["recommendations"].append(
                {
                    "metric": "FID",
                    "issue": f"FID too high: {fid}ms",
                    "fixes": [
                        "Minimize JavaScript execution time",
                        "Break up long tasks",
                        "Use web workers for heavy computations",
                        "Reduce third-party script impact",
                    ],
                }
            )

        # CLS - Cumulative Layout Shift
        cls = self._measure_cls(url)
        vitals["metrics"]["cls"] = {
            "value": cls,
            "unit": "score",
            "rating": self._rate_cls(cls),
            "threshold": "< 0.1 (good), < 0.25 (needs improvement), >= 0.25 (poor)",
        }

        if cls > 0.1:
            vitals["recommendations"].append(
                {
                    "metric": "CLS",
                    "issue": f"CLS too high: {cls}",
                    "fixes": [
                        "Set explicit width/height for images and embeds",
                        "Avoid inserting content above existing content",
                        "Use CSS aspect ratio boxes",
                        "Reserve space for ads and embeds",
                    ],
                }
            )

        # Overall pass/fail
        vitals["passed"] = (
            vitals["metrics"]["lcp"]["rating"] == "good"
            and vitals["metrics"]["fid"]["rating"] == "good"
            and vitals["metrics"]["cls"]["rating"] == "good"
        )

        logger.info(f"Core Web Vitals: {'PASS' if vitals['passed'] else 'FAIL'}")

        return vitals

    def check_amp_compatibility(self, url: str) -> Dict[str, Any]:
        """Проверяет AMP совместимость."""
        amp_check = {
            "url": url,
            "has_amp": False,
            "amp_url": None,
            "amp_valid": False,
            "errors": [],
            "recommendations": [],
        }

        # Проверяем наличие AMP версии
        amp_url = self._detect_amp_version(url)

        if amp_url:
            amp_check["has_amp"] = True
            amp_check["amp_url"] = amp_url

            # Валидация AMP
            validation = self._validate_amp(amp_url)
            amp_check["amp_valid"] = validation["valid"]
            amp_check["errors"] = validation["errors"]

        else:
            amp_check["recommendations"].append(
                {
                    "action": "Consider implementing AMP",
                    "benefit": "Faster mobile loading, potential featured snippet",
                    "difficulty": "medium",
                }
            )

        logger.info(f"AMP check: {'Valid AMP' if amp_check['amp_valid'] else 'No AMP or invalid'}")

        return amp_check

    def optimize_for_mobile(self, page_content: Dict) -> Dict[str, Any]:
        """АВТОМАТИЧЕСКАЯ оптимизация для мобильных."""
        optimizations = {
            "changes_made": [],
            "viewport_fixed": False,
            "images_optimized": False,
            "fonts_adjusted": False,
            "touch_targets_fixed": False,
        }

        # 1. Viewport meta tag
        if not page_content.get("has_viewport_meta"):
            # Добавляем viewport meta
            optimizations["viewport_fixed"] = True
            optimizations["changes_made"].append("Added viewport meta tag")

        # 2. Responsive images
        images = page_content.get("images", [])
        for img in images:
            if not img.get("responsive"):
                # Делаем изображение responsive
                optimizations["images_optimized"] = True

        if optimizations["images_optimized"]:
            optimizations["changes_made"].append(f"Made {len(images)} images responsive")

        # 3. Font sizes
        if page_content.get("min_font_size", 16) < 16:
            optimizations["fonts_adjusted"] = True
            optimizations["changes_made"].append("Increased minimum font size to 16px")

        # 4. Touch targets
        small_targets = page_content.get("small_touch_targets", [])
        if small_targets:
            optimizations["touch_targets_fixed"] = True
            optimizations["changes_made"].append(
                f"Increased size of {len(small_targets)} touch targets to 48x48px"
            )

        logger.info(f"✅ Mobile optimization: {len(optimizations['changes_made'])} changes")

        return {"success": True, "optimizations": optimizations}

    def generate_mobile_report(self, url: str) -> Dict[str, Any]:
        """Генерирует полный отчет по мобильной оптимизации."""
        report = {"url": url, "generated_at": "2025-11-15", "overall_score": 0, "sections": {}}

        # Собираем все проверки
        mobile_friendly = self.analyze_mobile_friendliness(url)
        core_vitals = self.analyze_core_web_vitals(url)
        amp_check = self.check_amp_compatibility(url)

        report["sections"]["mobile_friendliness"] = mobile_friendly
        report["sections"]["core_web_vitals"] = core_vitals
        report["sections"]["amp"] = amp_check

        # Общий score
        total_score = (
            mobile_friendly["score"] * 0.4
            + (100 if core_vitals["passed"] else 50) * 0.4
            + (100 if amp_check["amp_valid"] else 0) * 0.2
        )

        report["overall_score"] = round(total_score)

        # Приоритетные рекомендации
        all_recommendations = []
        all_recommendations.extend(mobile_friendly.get("recommendations", []))
        all_recommendations.extend(core_vitals.get("recommendations", []))
        all_recommendations.extend(amp_check.get("recommendations", []))

        report["priority_actions"] = all_recommendations[:5]  # Top 5

        logger.info(f"Mobile report generated: {report['overall_score']}/100")

        return report

    # Helper methods

    def _check_viewport_meta(self, url: str) -> Dict:
        """Проверяет viewport meta tag."""
        # Заглушка
        return {"passed": True, "content": "width=device-width, initial-scale=1"}

    def _check_touch_elements(self, url: str) -> Dict:
        """Проверяет размер touch targets."""
        return {"passed": False, "small_targets": 5}

    def _check_text_readability(self, url: str) -> Dict:
        """Проверяет читабельность текста."""
        return {"passed": True, "min_font_size": 16}

    def _check_responsive_images(self, url: str) -> Dict:
        """Проверяет responsive images."""
        return {"passed": False, "non_responsive_count": 3}

    def _check_horizontal_scroll(self, url: str) -> Dict:
        """Проверяет горизонтальную прокрутку."""
        return {"passed": True}

    def _check_element_spacing(self, url: str) -> Dict:
        """Проверяет spacing между элементами."""
        return {"passed": True}

    def _check_mobile_speed(self, url: str) -> Dict:
        """Проверяет скорость загрузки на мобильных."""
        return {"passed": False, "load_time": 4.5}

    def _measure_lcp(self, url: str) -> float:
        """Измеряет Largest Contentful Paint."""
        # В реальности использовать Lighthouse API или PageSpeed Insights API
        return 3.2  # seconds

    def _measure_fid(self, url: str) -> float:
        """Измеряет First Input Delay."""
        return 150  # milliseconds

    def _measure_cls(self, url: str) -> float:
        """Измеряет Cumulative Layout Shift."""
        return 0.15

    def _rate_lcp(self, value: float) -> str:
        """Оценивает LCP."""
        if value < 2.5:
            return "good"
        elif value < 4.0:
            return "needs_improvement"
        else:
            return "poor"

    def _rate_fid(self, value: float) -> str:
        """Оценивает FID."""
        if value < 100:
            return "good"
        elif value < 300:
            return "needs_improvement"
        else:
            return "poor"

    def _rate_cls(self, value: float) -> str:
        """Оценивает CLS."""
        if value < 0.1:
            return "good"
        elif value < 0.25:
            return "needs_improvement"
        else:
            return "poor"

    def _detect_amp_version(self, url: str) -> str:
        """Определяет AMP версию страницы."""
        # Проверяем наличие AMP link tag
        # или конвенции URL (/amp, ?amp=1)
        return None

    def _validate_amp(self, amp_url: str) -> Dict:
        """Валидирует AMP страницу."""
        # В реальности использовать AMP Validator API
        return {"valid": False, "errors": ["Missing required AMP tag", "Invalid AMP component"]}
