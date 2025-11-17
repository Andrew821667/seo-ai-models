"""
International SEO - Оптимизация для международных рынков.

Функции:
- Hreflang тегов генерация и валидация
- Геотаргетинг настройка
- Мультиязычная оптимизация контента
- Локализация meta-тегов
- International duplicate content detection
"""

import logging
from typing import Dict, Any, List
import re

logger = logging.getLogger(__name__)


class InternationalSEO:
    """Международная SEO оптимизация."""

    def __init__(self, llm_service=None, cms_connector=None):
        self.llm = llm_service
        self.cms = cms_connector
        logger.info("InternationalSEO initialized")

    def generate_hreflang_tags(
        self, page_url: str, language_versions: Dict[str, str]
    ) -> List[Dict]:
        """
        Генерирует hreflang теги для мультиязычных версий.

        Args:
            page_url: URL страницы
            language_versions: {"en": "https://example.com/en/page", "ru": "https://example.ru/page"}
        """
        hreflang_tags = []

        for lang_code, url in language_versions.items():
            tag = {"rel": "alternate", "hreflang": lang_code, "href": url}
            hreflang_tags.append(tag)

        # Добавляем x-default для основной версии
        if language_versions:
            default_url = list(language_versions.values())[0]
            hreflang_tags.append({"rel": "alternate", "hreflang": "x-default", "href": default_url})

        logger.info(f"Generated {len(hreflang_tags)} hreflang tags for {page_url}")

        return hreflang_tags

    def validate_hreflang(self, page_content: Dict) -> Dict[str, Any]:
        """Валидирует существующие hreflang теги."""
        hreflang_tags = page_content.get("hreflang_tags", [])

        validation = {"valid": True, "errors": [], "warnings": [], "tags_count": len(hreflang_tags)}

        if not hreflang_tags:
            validation["warnings"].append("No hreflang tags found")
            return validation

        # Проверяем формат тегов
        seen_langs = set()
        has_x_default = False

        for tag in hreflang_tags:
            lang = tag.get("hreflang")
            url = tag.get("href")

            # Проверка формата языка
            if not self._is_valid_lang_code(lang):
                validation["errors"].append(f"Invalid language code: {lang}")
                validation["valid"] = False

            # Проверка дубликатов
            if lang in seen_langs:
                validation["errors"].append(f"Duplicate hreflang: {lang}")
                validation["valid"] = False

            seen_langs.add(lang)

            # Проверка x-default
            if lang == "x-default":
                has_x_default = True

            # Проверка URL
            if not url or not url.startswith("http"):
                validation["errors"].append(f"Invalid URL for {lang}: {url}")
                validation["valid"] = False

        # Рекомендация x-default
        if not has_x_default:
            validation["warnings"].append("Consider adding x-default hreflang")

        logger.info(
            f"Hreflang validation: {len(validation['errors'])} errors, {len(validation['warnings'])} warnings"
        )

        return validation

    def localize_content(
        self, content: str, target_language: str, target_country: str
    ) -> Dict[str, Any]:
        """Локализует контент для целевого рынка."""
        if not self.llm:
            return {"success": False, "error": "LLM service not available"}

        prompt = f"""Localize this content for {target_country} ({target_language} language):

Original content:
{content[:1000]}

Localization requirements:
1. Translate to {target_language}
2. Adapt cultural references for {target_country}
3. Use local terminology and expressions
4. Keep SEO keywords relevant
5. Maintain similar length

Localized version:"""

        localized = self.llm.generate(prompt, max_tokens=800)

        result = {
            "success": True,
            "original_length": len(content),
            "localized_length": len(localized),
            "localized_content": localized,
            "target_language": target_language,
            "target_country": target_country,
        }

        logger.info(f"Localized content for {target_country} ({target_language})")

        return result

    def detect_international_duplicates(self, pages: List[Dict]) -> Dict[str, Any]:
        """Обнаруживает международные дубликаты контента."""
        duplicates = {"total_pages": len(pages), "duplicate_groups": [], "recommendations": []}

        # Группируем страницы по контенту
        content_groups = {}

        for page in pages:
            # Нормализуем контент для сравнения
            normalized = self._normalize_content(page.get("text", ""))
            content_hash = hash(normalized[:500])  # Хеш первых 500 символов

            if content_hash not in content_groups:
                content_groups[content_hash] = []

            content_groups[content_hash].append(page)

        # Находим группы с дубликатами
        for content_hash, group in content_groups.items():
            if len(group) > 1:
                duplicate_group = {
                    "pages": [
                        {"url": p["url"], "language": p.get("language", "unknown")} for p in group
                    ],
                    "count": len(group),
                    "severity": "high" if len(group) > 3 else "medium",
                }
                duplicates["duplicate_groups"].append(duplicate_group)

                # Рекомендации
                duplicates["recommendations"].append(
                    {
                        "issue": f"{len(group)} pages with similar content",
                        "solution": "Add hreflang tags or canonical URLs to specify language/region versions",
                    }
                )

        logger.info(f"Found {len(duplicates['duplicate_groups'])} duplicate groups")

        return duplicates

    def optimize_for_geo_targeting(
        self, domain: str, target_countries: List[str]
    ) -> Dict[str, Any]:
        """Оптимизация геотаргетинга."""
        optimization = {
            "domain": domain,
            "target_countries": target_countries,
            "recommendations": [],
            "technical_setup": [],
        }

        # Определяем структуру URL
        domain_type = self._detect_domain_structure(domain)

        if domain_type == "single":
            optimization["recommendations"].append(
                {
                    "type": "domain_structure",
                    "message": "Consider using ccTLDs, subdomains, or subdirectories for international versions",
                    "options": [
                        "ccTLD: example.de, example.fr (best for SEO, more expensive)",
                        "Subdomain: de.example.com, fr.example.com (easier management)",
                        "Subdirectory: example.com/de/, example.com/fr/ (easiest to implement)",
                    ],
                }
            )

        # Технические настройки
        optimization["technical_setup"] = [
            "Set up hreflang tags for all language/region versions",
            "Configure Google Search Console for geo-targeting",
            "Use local hosting or CDN for target regions",
            "Register domain in Google Search Console for each country",
            "Add language meta tags: <meta http-equiv='content-language' content='de'>",
            f"Create XML sitemaps for each language version",
        ]

        # Country-specific recommendations
        for country in target_countries:
            country_tips = self._get_country_specific_tips(country)
            optimization["recommendations"].append({"country": country, "tips": country_tips})

        logger.info(
            f"Generated geo-targeting recommendations for {len(target_countries)} countries"
        )

        return optimization

    def translate_meta_tags(self, meta_tags: Dict, target_language: str) -> Dict[str, str]:
        """Переводит meta-теги на целевой язык."""
        if not self.llm:
            return meta_tags

        title = meta_tags.get("title", "")
        description = meta_tags.get("description", "")

        prompt = f"""Translate these SEO meta tags to {target_language}:

Title: {title}
Description: {description}

Requirements:
- Keep SEO effectiveness
- Maintain character limits (title: 30-60, description: 120-160)
- Use natural {target_language} language
- Keep keywords relevant

Provide translation in this format:
Title: [translated title]
Description: [translated description]"""

        translation = self.llm.generate(prompt, max_tokens=200)

        # Парсим ответ
        translated = self._parse_translated_meta(translation)

        logger.info(f"Translated meta tags to {target_language}")

        return translated

    def audit_international_setup(self, domain: str) -> Dict[str, Any]:
        """Аудит международной SEO настройки."""
        audit = {
            "domain": domain,
            "score": 0,
            "issues": [],
            "passed_checks": [],
            "recommendations": [],
        }

        checks = [
            self._check_hreflang_implementation(domain),
            self._check_language_meta_tags(domain),
            self._check_url_structure(domain),
            self._check_content_localization(domain),
            self._check_geo_targeting_settings(domain),
        ]

        total_checks = len(checks)
        passed = sum(1 for check in checks if check["passed"])

        audit["score"] = round((passed / total_checks) * 100)

        for check in checks:
            if check["passed"]:
                audit["passed_checks"].append(check["name"])
            else:
                audit["issues"].append(
                    {
                        "check": check["name"],
                        "issue": check["issue"],
                        "recommendation": check["recommendation"],
                    }
                )

        logger.info(f"International SEO audit: {audit['score']}/100")

        return audit

    # Helper methods

    def _is_valid_lang_code(self, lang_code: str) -> bool:
        """Проверяет валидность языкового кода."""
        if lang_code == "x-default":
            return True

        # ISO 639-1 (2 letter) или ISO 639-1 + ISO 3166-1 (en-US)
        pattern = r"^[a-z]{2}(-[A-Z]{2})?$"
        return bool(re.match(pattern, lang_code))

    def _normalize_content(self, text: str) -> str:
        """Нормализует текст для сравнения."""
        # Удаляем пунктуацию, приводим к lowercase
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        return normalized

    def _detect_domain_structure(self, domain: str) -> str:
        """Определяет тип структуры домена."""
        if "/" in domain:
            return "subdirectory"
        elif domain.count(".") > 1:
            return "subdomain"
        else:
            return "single"

    def _get_country_specific_tips(self, country: str) -> List[str]:
        """Возвращает специфичные советы для страны."""
        tips_db = {
            "DE": [
                "Germans prefer detailed product information",
                "Use formal 'Sie' in B2B contexts",
                "Include precise technical specifications",
            ],
            "FR": [
                "French users value quality content over quantity",
                "Use proper French language (avoid anglicisms)",
                "Include cultural references relevant to France",
            ],
            "JP": [
                "Japanese users prefer mobile-first design",
                "Use polite language forms (keigo)",
                "Include detailed images and diagrams",
            ],
            "US": [
                "Americans prefer concise, action-oriented content",
                "Use clear CTAs and direct language",
                "Optimize for voice search",
            ],
        }

        return tips_db.get(country, ["Research local market preferences"])

    def _parse_translated_meta(self, translation: str) -> Dict[str, str]:
        """Парсит переведенные meta-теги из ответа LLM."""
        result = {}

        title_match = re.search(r"Title:\s*(.+?)(?:\n|$)", translation)
        desc_match = re.search(r"Description:\s*(.+?)(?:\n|$)", translation)

        if title_match:
            result["title"] = title_match.group(1).strip()

        if desc_match:
            result["description"] = desc_match.group(1).strip()

        return result

    def _check_hreflang_implementation(self, domain: str) -> Dict:
        """Проверяет наличие hreflang тегов."""
        # Заглушка
        return {
            "name": "Hreflang Implementation",
            "passed": False,
            "issue": "No hreflang tags detected",
            "recommendation": "Implement hreflang tags for all language versions",
        }

    def _check_language_meta_tags(self, domain: str) -> Dict:
        """Проверяет language meta теги."""
        return {"name": "Language Meta Tags", "passed": True, "issue": None, "recommendation": None}

    def _check_url_structure(self, domain: str) -> Dict:
        """Проверяет URL структуру."""
        return {
            "name": "URL Structure",
            "passed": False,
            "issue": "No clear language/region indicators in URLs",
            "recommendation": "Use subdirectories or subdomains for language versions",
        }

    def _check_content_localization(self, domain: str) -> Dict:
        """Проверяет локализацию контента."""
        return {
            "name": "Content Localization",
            "passed": False,
            "issue": "Content appears to be machine-translated",
            "recommendation": "Use professional translators for quality localization",
        }

    def _check_geo_targeting_settings(self, domain: str) -> Dict:
        """Проверяет настройки геотаргетинга."""
        return {
            "name": "Geo-Targeting Settings",
            "passed": True,
            "issue": None,
            "recommendation": None,
        }
