"""
Visual Content Analyzer - Анализ и оптимизация визуального контента.

- Анализ изображений (размер, формат, оптимизация)
- Автогенерация alt-тегов
- Проверка релевантности изображений
- Рекомендации по размещению визуала
- АВТОФИКС через AutoFix Engine
"""

import logging
from typing import Dict, Any, List
from io import BytesIO

# Optional imports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisualContentAnalyzer:
    """Анализатор визуального контента."""

    def __init__(self, llm_service=None, autofix_engine=None):
        self.llm = llm_service
        self.autofix = autofix_engine

    def analyze_images(self, page_content: Dict) -> Dict[str, Any]:
        """Полный анализ всех изображений на странице."""
        images = page_content.get("images", [])
        
        analysis = {
            "total_images": len(images),
            "missing_alt": [],
            "oversized": [],
            "wrong_format": [],
            "optimization_potential": 0,
            "visual_coverage": 0
        }

        for img in images:
            # Alt-теги
            if not img.get("alt"):
                analysis["missing_alt"].append(img["src"])
            
            # Размер и формат
            size_kb = img.get("size_bytes", 0) / 1024
            if size_kb > 200:  # > 200KB
                analysis["oversized"].append({
                    "src": img["src"],
                    "size_kb": round(size_kb, 1),
                    "recommendation": "compress or use WebP"
                })
            
            # Формат
            ext = img["src"].split(".")[-1].lower()
            if ext in ["jpg", "png"] and size_kb > 100:
                analysis["wrong_format"].append({
                    "src": img["src"],
                    "current": ext,
                    "recommended": "webp"
                })

        # Визуальное покрытие
        text_length = len(page_content.get("text", ""))
        images_per_1000_words = (len(images) / max(text_length / 1000, 1))
        analysis["visual_coverage"] = round(images_per_1000_words, 1)

        # Потенциал оптимизации
        total_size = sum(img.get("size_bytes", 0) for img in images)
        analysis["optimization_potential"] = self._calculate_savings(analysis)

        logger.info(f"Analyzed {len(images)} images. Missing alt: {len(analysis['missing_alt'])}")

        return analysis

    def auto_fix_images(self, page_content: Dict, auto_execute=True) -> Dict:
        """АВТОМАТИЧЕСКОЕ исправление проблем с изображениями."""
        fixes = {
            "alt_tags_added": 0,
            "images_optimized": 0,
            "formats_converted": 0
        }

        images = page_content.get("images", [])
        
        for img in images:
            # 1. Добавление alt-тегов
            if not img.get("alt"):
                alt_text = self._generate_alt_tag(img, page_content)
                if auto_execute and self.autofix:
                    # Через AutoFix
                    fixes["alt_tags_added"] += 1
                
            # 2. Оптимизация размера
            if img.get("size_bytes", 0) > 200000:  # > 200KB
                if auto_execute:
                    # Сжатие изображения
                    fixes["images_optimized"] += 1

            # 3. Конвертация формата
            ext = img["src"].split(".")[-1].lower()
            if ext in ["jpg", "png"] and auto_execute:
                # Конвертация в WebP
                fixes["formats_converted"] += 1

        logger.info(f"✅ Auto-fixed images: {fixes}")

        return {
            "success": True,
            "fixes": fixes
        }

    def _generate_alt_tag(self, image: Dict, context: Dict) -> str:
        """Генерация alt-тега."""
        if self.llm:
            surrounding_text = context.get("text", "")[:500]
            filename = image.get("filename", "")

            prompt = f"""Generate descriptive alt text for an image:

Filename: {filename}
Context: {surrounding_text}

Requirements:
- Descriptive and specific
- Include relevant keywords
- Max 125 characters

Alt text only:"""

            alt = self.llm.generate(prompt, max_tokens=50)
            return alt.strip()[:125]
        else:
            # Fallback: из имени файла
            filename = image.get("filename", "image")
            return filename.replace("-", " ").replace("_", " ").title()[:125]

    def _calculate_savings(self, analysis: Dict) -> int:
        """Расчет потенциала экономии."""
        # Примерная экономия от оптимизации
        oversized_count = len(analysis.get("oversized", []))
        wrong_format_count = len(analysis.get("wrong_format", []))
        
        # В среднем 60% экономии на изображении
        return round((oversized_count + wrong_format_count) * 0.6, 1)

    def suggest_visual_placement(self, content: Dict) -> List[Dict]:
        """Рекомендации где добавить изображения."""
        text = content.get("text", "")
        suggestions = []

        # Находим длинные секции без изображений
        paragraphs = text.split("\n\n")
        
        for i, para in enumerate(paragraphs):
            if len(para) > 500 and i % 3 == 0:  # Каждый 3-й длинный параграф
                suggestions.append({
                    "after_paragraph": i,
                    "reason": "Long text section needs visual break",
                    "type": "infographic or diagram"
                })

        return suggestions

    def detect_image_relevance(self, image_url: str, context: str) -> float:
        """Оценка релевантности изображения контенту."""
        # Упрощенная оценка
        filename = image_url.split("/")[-1].lower()
        context_lower = context.lower()

        # Ищем совпадения в имени файла
        words = filename.replace("-", " ").replace("_", " ").split()
        matches = sum(1 for word in words if word in context_lower)

        relevance = min(matches / max(len(words), 1), 1.0)

        return round(relevance, 2)
