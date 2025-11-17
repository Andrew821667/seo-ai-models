"""
Модуль для определения SPA-приложений в проекте SEO AI Models.
Предоставляет функции для обнаружения JavaScript фреймворков и SPA-паттернов.
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SPADetector:
    """
    Определяет, является ли веб-страница SPA-приложением по различным признакам.
    """

    def __init__(self):
        """
        Инициализация SPADetector.
        """
        # Список признаков для определения различных JS-фреймворков
        self.framework_patterns = {
            "React": [
                # Признаки React
                "reactroot",
                "react-root",
                "data-reactroot",
                "data-reactid",
                "react.development.js",
                "react.production.min.js",
                "react-dom",
            ],
            "Angular": [
                # Признаки Angular
                "ng-app",
                "ng-controller",
                "ng-model",
                "ng-bind",
                "ng-repeat",
                "angular.js",
                "angular.min.js",
                "_nghost",
                "_ngcontent",
                "ng-version",
                "[ng",
            ],
            "Vue": [
                # Признаки Vue
                "v-app",
                "v-bind",
                "v-model",
                "v-if",
                "v-for",
                "v-on",
                "vue.js",
                "vue.min.js",
                "data-v-",
                "[data-v-",
                "vue-router",
                "vuex",
            ],
            "Ember": [
                # Признаки Ember
                "ember-application",
                "data-ember",
                "ember-view",
                "ember-controller",
                "ember.js",
                "ember.min.js",
            ],
            "Next.js": [
                # Признаки Next.js
                "__NEXT_DATA__",
                "next/router",
                "_next/",
            ],
            "Nuxt.js": [
                # Признаки Nuxt.js
                "__NUXT__",
                "nuxt-link",
                "_nuxt/",
                "nuxt.js",
                "nuxt.min.js",
            ],
            "SvelteKit": [
                # Признаки SvelteKit
                "__SVELTEKIT",
                "svelte-",
                "svelte.js",
                "svelte.min.js",
            ],
        }

        # Другие признаки SPA
        self.spa_indicators = [
            # AJAX API клиенты
            "axios",
            "fetch(",
            ".fetch(",
            "XMLHttpRequest",
            "jquery.ajax",
            # Роутеры
            "router",
            "history.pushState",
            "window.history",
            # Шаблонизаторы
            "template",
            "mustache",
            "handlebars",
            # Другие признаки
            "spa",
            "single-page-application",
            "single-page-app",
        ]

        # Признаки динамической загрузки
        self.dynamic_loading_indicators = [
            "lazyload",
            "lazy-load",
            "infinite-scroll",
            "infinite-loading",
            "load-more",
            "loadMore",
            "pagination",
        ]

    def analyze_html(self, html_content: str) -> Dict[str, Any]:
        """
        Анализирует HTML на предмет признаков SPA.

        Args:
            html_content: HTML-контент для анализа

        Returns:
            Dict[str, Any]: Результаты анализа
        """
        if not html_content:
            return {
                "is_spa": False,
                "confidence": 0,
                "detected_frameworks": [],
                "dynamic_features": [],
            }

        soup = BeautifulSoup(html_content, "html.parser")

        # Получаем текст для анализа
        html_text = str(html_content).lower()

        # Проверяем на фреймворки
        detected_frameworks = []
        for framework, patterns in self.framework_patterns.items():
            for pattern in patterns:
                pattern_lower = pattern.lower()
                if pattern_lower in html_text:
                    if framework not in detected_frameworks:
                        detected_frameworks.append(framework)
                    break

        # Проверяем на другие признаки SPA
        spa_indicators_found = []
        for indicator in self.spa_indicators:
            if indicator.lower() in html_text:
                spa_indicators_found.append(indicator)

        # Проверяем на признаки динамической загрузки
        dynamic_features = []
        for indicator in self.dynamic_loading_indicators:
            if indicator.lower() in html_text:
                dynamic_features.append(indicator)

        # Анализируем скрипты
        script_count = len(soup.find_all("script"))
        large_scripts = 0

        for script in soup.find_all("script"):
            if script.string and len(script.string) > 1000:  # Большие скрипты
                large_scripts += 1

        # Вычисляем оценку уверенности
        confidence = 0

        # Если найден хотя бы один фреймворк, высокая уверенность
        if detected_frameworks:
            confidence += 0.7

        # Если найдены другие признаки SPA, средняя уверенность
        if spa_indicators_found:
            confidence += min(0.3, 0.05 * len(spa_indicators_found))

        # Если найдены признаки динамической загрузки, небольшая уверенность
        if dynamic_features:
            confidence += min(0.2, 0.04 * len(dynamic_features))

        # Если много скриптов, небольшая уверенность
        if script_count > 10:
            confidence += 0.1

        # Если есть большие скрипты, небольшая уверенность
        if large_scripts > 0:
            confidence += min(0.2, 0.05 * large_scripts)

        # Ограничиваем значение уверенности до 1.0
        confidence = min(1.0, confidence)

        # Определяем, является ли сайт SPA
        is_spa = confidence > 0.3 or len(detected_frameworks) > 0

        return {
            "is_spa": is_spa,
            "confidence": confidence,
            "detected_frameworks": detected_frameworks,
            "spa_indicators": spa_indicators_found,
            "dynamic_features": dynamic_features,
            "script_stats": {"total_scripts": script_count, "large_scripts": large_scripts},
        }

    def is_spa(self, html_content: str) -> bool:
        """
        Быстрая проверка, является ли страница SPA.

        Args:
            html_content: HTML-контент для анализа

        Returns:
            bool: True, если страница является SPA
        """
        result = self.analyze_html(html_content)
        return result["is_spa"]
