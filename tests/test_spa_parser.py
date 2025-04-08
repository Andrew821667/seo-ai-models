"""
Тесты для модулей парсинга SPA в проекте SEO AI Models.
"""

import unittest
import requests
from unittest.mock import patch, MagicMock

from seo_ai_models.parsers.utils.spa_detector import SPADetector
from seo_ai_models.parsers.adaptive_parsing_pipeline import AdaptiveParsingPipeline

class TestSPADetector(unittest.TestCase):
    """Тесты для детектора SPA-приложений."""
    
    def setUp(self):
        self.detector = SPADetector()
    
    def test_analyze_react_html(self):
        """Тест определения React-приложения."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>React App</title>
            <script src="react.development.js"></script>
            <script src="react-dom.development.js"></script>
        </head>
        <body>
            <div id="root" data-reactroot></div>
            <script src="app.js"></script>
        </body>
        </html>
        """
        
        result = self.detector.analyze_html(html)
        self.assertTrue(result["is_spa"])
        self.assertIn("React", result["detected_frameworks"])
    
    def test_analyze_angular_html(self):
        """Тест определения Angular-приложения."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Angular App</title>
            <script src="angular.js"></script>
        </head>
        <body ng-app="myApp">
            <div ng-controller="MainCtrl">
                <div ng-repeat="item in items">{{item.name}}</div>
            </div>
        </body>
        </html>
        """
        
        result = self.detector.analyze_html(html)
        self.assertTrue(result["is_spa"])
        self.assertIn("Angular", result["detected_frameworks"])
    
    def test_analyze_standard_html(self):
        """Тест определения обычного HTML-сайта."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Regular Website</title>
            <link rel="stylesheet" href="style.css">
        </head>
        <body>
            <header>
                <h1>Welcome to our site</h1>
            </header>
            <main>
                <article>
                    <h2>Article Title</h2>
                    <p>This is a regular website with no SPA frameworks</p>
                </article>
            </main>
            <footer>
                <p>&copy; 2025</p>
            </footer>
            <script src="analytics.js"></script>
        </body>
        </html>
        """
        
        result = self.detector.analyze_html(html)
        self.assertFalse(result["is_spa"])
        self.assertEqual(0, len(result["detected_frameworks"]))


class TestAdaptiveParsingPipeline(unittest.TestCase):
    """Тесты для адаптивного конвейера парсинга."""
    
    def setUp(self):
        self.pipeline = AdaptiveParsingPipeline(
            headless=True,
            max_pages=3
        )
    
    @patch('seo_ai_models.parsers.adaptive_parsing_pipeline.fetch_url')
    def test_detect_site_type_standard(self, mock_fetch_url):
        """Тест определения обычного сайта."""
        # Моделируем успешный ответ для стандартного сайта
        mock_fetch_url.return_value = (
            """
            <!DOCTYPE html>
            <html>
            <head><title>Standard Website</title></head>
            <body><h1>Hello World</h1></body>
            </html>
            """,
            {"Content-Type": "text/html"},
            None
        )
        
        # Мокаем fetch_url_with_javascript_sync, чтобы его не вызвали
        with patch('seo_ai_models.parsers.utils.request_utils.fetch_url_with_javascript_sync') as mock_js_fetch:
            result = self.pipeline.detect_site_type("http://example.com")
            
            # Проверяем, что fetch_url был вызван, а fetch_url_with_javascript_sync - нет
            mock_fetch_url.assert_called_once()
            mock_js_fetch.assert_not_called()
            
            # Проверяем результат
            self.assertFalse(result["is_spa"])
    
    @patch('seo_ai_models.parsers.adaptive_parsing_pipeline.fetch_url')
    @patch('seo_ai_models.parsers.adaptive_parsing_pipeline.fetch_url_with_javascript_sync')
    def test_detect_site_type_spa(self, mock_js_fetch, mock_fetch_url):
        """Тест определения SPA-сайта."""
        # Моделируем ответ для SPA-сайта
        mock_fetch_url.return_value = (
            """
            <!DOCTYPE html>
            <html>
            <head>
                <title>React App</title>
                <script src="react.development.js"></script>
            </head>
            <body>
                <div id="root"></div>
            </body>
            </html>
            """,
            {"Content-Type": "text/html"},
            None
        )
        
        # Моделируем JavaScript-рендеринг с дополнительным содержимым
        mock_js_fetch.return_value = (
            """
            <!DOCTYPE html>
            <html>
            <head>
                <title>React App</title>
                <script src="react.development.js"></script>
            </head>
            <body>
                <div id="root">
                    <div data-reactroot>
                        <h1>Welcome to React</h1>
                        <p>This content is rendered by JavaScript</p>
                        <div class="app-content">
                            <h2>Dynamic Content</h2>
                            <ul>
                                <li>Item 1</li>
                                <li>Item 2</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """,
            {"Content-Type": "text/html"},
            None
        )
        
        result = self.pipeline.detect_site_type("http://example.com/react-app")
        
        # Проверяем, что оба метода были вызваны
        mock_fetch_url.assert_called_once()
        mock_js_fetch.assert_called_once()
        
        # Проверяем результат
        self.assertTrue(result["is_spa"])
        self.assertGreater(result["confidence"], 0.5)
    
    def test_force_spa_mode(self):
        """Тест принудительного режима SPA."""
        # Создаем конвейер с принудительным режимом SPA
        pipeline = AdaptiveParsingPipeline(force_spa_mode=True)
        
        # Патчим fetch_url, чтобы не делать реальных запросов
        with patch('seo_ai_models.parsers.utils.request_utils.fetch_url'):
            result = pipeline.detect_site_type("http://example.com")
            
            # Проверяем, что сайт определен как SPA без анализа
            self.assertTrue(result["is_spa"])
            self.assertEqual(1.0, result["confidence"])
            self.assertEqual("forced", result["detection_method"])

if __name__ == '__main__':
    unittest.main()
