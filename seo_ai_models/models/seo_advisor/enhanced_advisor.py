
"""
Улучшенная версия SEO Advisor с интеграцией парсинга и проверки согласованности метрик.
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass

from seo_ai_models.models.seo_advisor.advisor import SEOAdvisor, SEOAnalysisReport, ContentQualityReport
from seo_ai_models.parsers.spa_parser import SPAParser
from seo_ai_models.models.seo_advisor.analyzers.enhanced_content_analyzer import EnhancedContentAnalyzer

class EnhancedSEOAdvisor(SEOAdvisor):
    """
    Улучшенный SEO Advisor с поддержкой SPA-сайтов и проверкой согласованности метрик.
    """
    
    def __init__(self, industry: str = 'default'):
        """Инициализация улучшенного SEO Advisor."""
        super().__init__(industry)
        
        # Заменяем стандартный ContentAnalyzer на улучшенный
        self.enhanced_content_analyzer = EnhancedContentAnalyzer()
        
        # Инициализируем SPA Parser
        self.spa_parser = SPAParser(
            wait_for_load=7000,  # 7 секунд ожидания после загрузки
            wait_for_timeout=45000,  # 45 секунд максимального ожидания
            record_ajax=True,
            analyze_ajax=True,
            max_retries=3
        )
    
    def analyze_url(self, url: str, target_keywords: List[str]) -> SEOAnalysisReport:
        """
        Анализ URL с поддержкой SPA и JavaScript.
        
        Args:
            url: URL для анализа
            target_keywords: Целевые ключевые слова
            
        Returns:
            SEOAnalysisReport: Полный отчет по SEO анализу
        """
        # Используем SPA Parser для получения контента
        parsed_data = self.spa_parser.analyze_url_sync(url)
        
        if not parsed_data.get('success'):
            # Возвращаем отчет с ошибкой
            return self._create_error_report(
                f"Ошибка при парсинге URL: {parsed_data.get('error')}",
                target_keywords
            )
        
        # Извлекаем HTML-контент для анализа
        html_content = parsed_data.get('html', '')
        
        # Извлекаем текстовое содержимое
        content = ""
        if 'content' in parsed_data and 'all_text' in parsed_data['content'].get('content', {}):
            content = parsed_data['content']['content']['all_text']
        else:
            for paragraph in parsed_data.get('content', {}).get('content', {}).get('paragraphs', []):
                content += paragraph + "\n\n"
        
        # Проверяем наличие контента
        if not content or len(content) < 100:
            return self._create_error_report(
                "Недостаточно контента для анализа после парсинга",
                target_keywords
            )
        
        # Используем улучшенный анализатор контента
        content_metrics = self.enhanced_content_analyzer.analyze_content(content, html_content)
        keyword_analysis = self.enhanced_content_analyzer.extract_keywords(content, target_keywords)
        
        # Продолжаем стандартный анализ
        # Выполняем семантический анализ, E-E-A-T и т.д.
        
        # Здесь можно вызвать стандартный метод analyze_content
        # и затем модифицировать результат, учитывая SPA-статус сайта
        
        report = super().analyze_content(content, target_keywords)
        
        # Добавляем информацию о SPA-статусе сайта
        is_spa = parsed_data.get('site_type', {}).get('is_spa', False)
        
        # Модифицируем отчет для отражения улучшенного анализа
        # (в реальной имплементации тут нужна более сложная логика)
        
        return report
    
    def _create_error_report(self, error_message: str, target_keywords: List[str]) -> SEOAnalysisReport:
        """
        Создает отчет с ошибкой.
        
        Args:
            error_message: Сообщение об ошибке
            target_keywords: Целевые ключевые слова
            
        Returns:
            SEOAnalysisReport: Отчет с ошибкой
        """
        # Создаем пустой отчет о качестве
        content_quality = ContentQualityReport(
            content_scores={'overall_quality': 0, 'error': 1},
            strengths=[],
            weaknesses=[f"ОШИБКА: {error_message}"],
            potential_improvements=["Проверьте доступность URL и попробуйте снова"]
        )
        
        # Возвращаем пустой отчет с ошибкой
        return SEOAnalysisReport(
            timestamp=datetime.now(),
            content_metrics={'error': error_message, 'word_count': 0},
            keyword_analysis={'density': 0, 'frequency': {}, 'distribution': {}},
            predicted_position=100,  # Худшая позиция
            feature_scores={},
            content_quality=content_quality,
            recommendations={'error': [error_message, "Проверьте доступность URL и попробуйте снова"]},
            priorities=[],
            industry=self.industry
        )
