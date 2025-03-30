"""Интеграционные тесты для взаимодействия анализаторов и предикторов."""

import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from seo_ai_models.models.seo_advisor.analyzers.content_analyzer import ContentAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.semantic_analyzer import SemanticAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.eeat.eeat_analyzer import EEATAnalyzer
from seo_ai_models.models.seo_advisor.predictors.calibrated_rank_predictor import CalibratedRankPredictor
from seo_ai_models.models.seo_advisor.suggester.suggester import Suggester


class TestAnalyzerPredictorIntegration(unittest.TestCase):
    """Интеграционные тесты для проверки взаимодействия анализаторов и предикторов."""
    
    def setUp(self):
        """Подготовка к тестам."""
        self.content_analyzer = ContentAnalyzer()
        self.semantic_analyzer = SemanticAnalyzer()
        self.eeat_analyzer = EEATAnalyzer()
        self.predictor = CalibratedRankPredictor()
        self.suggester = Suggester()
        
        self.test_content = """
        # Руководство по SEO-оптимизации
        
        SEO-оптимизация является важным аспектом продвижения сайта в поисковых системах.
        В этой статье мы рассмотрим основные принципы и методы оптимизации контента.
        
        ## Ключевые слова
        
        Правильный подбор и использование ключевых слов - основа SEO-оптимизации.
        Ключевые слова должны естественно вписываться в текст и соответствовать
        тематике страницы.
        
        ## Структура контента
        
        Хорошо структурированный контент улучшает пользовательский опыт и помогает
        поисковым системам лучше понять содержание страницы.
        
        * Используйте заголовки разных уровней
        * Разбивайте текст на абзацы
        * Применяйте списки для перечислений
        
        ## Мета-теги
        
        Оптимизация мета-тегов повышает релевантность страницы в выдаче.
        
        ## Заключение
        
        SEO-оптимизация - это непрерывный процесс, требующий постоянного анализа
        и улучшения контента.
        
        Источник: Справочник по SEO, 2024
        Автор: Эксперт по SEO-оптимизации
        """
        
        self.test_keywords = ["SEO", "оптимизация", "ключевые слова", "контент"]
    
    def test_full_analysis_pipeline(self):
        """Тест полного цикла анализа и предсказания."""
        # Анализ контента
        content_metrics = self.content_analyzer.analyze_text(self.test_content)
        keyword_analysis = self.content_analyzer.extract_keywords(
            self.test_content, self.test_keywords
        )
        
        # Семантический анализ
        semantic_analysis = self.semantic_analyzer.analyze_text(
            self.test_content, self.test_keywords
        )
        
        # E-E-A-T анализ
        eeat_analysis = self.eeat_analyzer.analyze(self.test_content)
        
        # Объединяем метрики
        combined_metrics = {**content_metrics}
        combined_metrics.update({
            "semantic_density": semantic_analysis["semantic_density"],
            "semantic_coverage": semantic_analysis["semantic_coverage"],
            "topical_coherence": semantic_analysis["topical_coherence"],
            "contextual_relevance": semantic_analysis["contextual_relevance"],
            "expertise_score": eeat_analysis["expertise_score"],
            "authority_score": eeat_analysis["authority_score"],
            "trust_score": eeat_analysis["trust_score"],
            "overall_eeat_score": eeat_analysis["overall_eeat_score"]
        })
        
        # Создаем данные для предсказания
        prediction_features = {
            'keyword_density': keyword_analysis['density'],
            'content_length': content_metrics['word_count'],
            'readability_score': content_metrics.get('readability', 0),
            'meta_tags_score': content_metrics.get('meta_score', 0),
            'header_structure_score': content_metrics.get('header_score', 0),
            'multimedia_score': content_metrics.get('multimedia_score', 0),
            'internal_linking_score': content_metrics.get('linking_score', 0),
            'topic_relevance': content_metrics.get('topic_relevance', 0),
            'semantic_depth': content_metrics.get('semantic_depth', 0),
            'engagement_potential': content_metrics.get('engagement_potential', 0),
            "expertise_score": combined_metrics.get("expertise_score", 0),
            "authority_score": combined_metrics.get("authority_score", 0),
            "trust_score": combined_metrics.get("trust_score", 0),
            "overall_eeat_score": combined_metrics.get("overall_eeat_score", 0)
        }
        
        # Получаем предсказание и рекомендации
        prediction = self.predictor.predict_position(prediction_features)
        base_recommendations = self.predictor.generate_recommendations(prediction_features)
        
        # Получаем улучшенные рекомендации
        enhanced_recommendations = self.suggester.generate_suggestions(
            base_recommendations,
            prediction['feature_scores'],
            'default'
        )
        
        # Приоритизация задач
        priorities = self.suggester.prioritize_tasks(
            enhanced_recommendations,
            prediction['feature_scores'],
            prediction['weighted_scores']
        )
        
        # Проверяем, что все этапы успешно выполнены и дали ожидаемый результат
        self.assertIsNotNone(prediction['position'])
        self.assertIsInstance(prediction['position'], (int, float))
        self.assertGreater(len(enhanced_recommendations), 0)
        self.assertGreater(len(priorities), 0)
    
    def test_quality_impact_on_position(self):
        """Тест влияния качества контента на предсказанную позицию."""
        # Высококачественный контент
        high_quality_content = self.test_content + """
        
        ## Дополнительные советы
        
        В дополнение к основным принципам, рекомендуется:
        
        1. Регулярно обновлять контент
        2. Улучшать скорость загрузки страниц
        3. Оптимизировать для мобильных устройств
        4. Использовать структурированные данные
        
        ### Технические аспекты
        
        Техническая оптимизация включает работу с:
        
        * robots.txt
        * sitemap.xml
        * HTTPS протоколом
        * Канонические URL
        
        Исследования показывают, что комплексный подход к SEO дает наилучшие результаты.
        
        Источник: Международная SEO ассоциация, исследование за 2023 год
        Автор: Профессор SEO-технологий, эксперт с 15-летним опытом
        """
        
        # Низкокачественный контент
        low_quality_content = "SEO оптимизация важна. Используйте ключевые слова. Создавайте контент."
        
        # Анализируем оба типа контента
        high_quality_metrics = self._analyze_content(high_quality_content)
        low_quality_metrics = self._analyze_content(low_quality_content)
        
        # Предсказываем позиции
        high_quality_prediction = self.predictor.predict_position(high_quality_metrics)
        low_quality_prediction = self.predictor.predict_position(low_quality_metrics)
        
        # Проверяем, что позиции прогнозируются
        self.assertIsNotNone(high_quality_prediction['position'])
        self.assertIsNotNone(low_quality_prediction['position'])
        
        # Высококачественный контент должен иметь не худшую позицию
        self.assertLessEqual(
            high_quality_prediction['position'],
            low_quality_prediction['position']
        )
    
    def _analyze_content(self, content):
        """Вспомогательный метод для полного анализа контента."""
        content_metrics = self.content_analyzer.analyze_text(content)
        keyword_analysis = self.content_analyzer.extract_keywords(content, self.test_keywords)
        semantic_analysis = self.semantic_analyzer.analyze_text(content, self.test_keywords)
        eeat_analysis = self.eeat_analyzer.analyze(content)
        
        # Создаем данные для предсказания
        return {
            'keyword_density': keyword_analysis['density'],
            'content_length': content_metrics['word_count'],
            'readability_score': content_metrics.get('readability', 0),
            'meta_tags_score': content_metrics.get('meta_score', 0),
            'header_structure_score': content_metrics.get('header_score', 0),
            'multimedia_score': content_metrics.get('multimedia_score', 0),
            'internal_linking_score': content_metrics.get('linking_score', 0),
            'topic_relevance': content_metrics.get('topic_relevance', 0),
            'semantic_depth': content_metrics.get('semantic_depth', 0),
            'engagement_potential': content_metrics.get('engagement_potential', 0),
            "expertise_score": eeat_analysis.get("expertise_score", 0),
            "authority_score": eeat_analysis.get("authority_score", 0),
            "trust_score": eeat_analysis.get("trust_score", 0),
            "overall_eeat_score": eeat_analysis.get("overall_eeat_score", 0)
        }


if __name__ == '__main__':
    unittest.main()
