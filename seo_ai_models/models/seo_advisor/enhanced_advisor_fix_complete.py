
"""
Полное исправление для EnhancedSEOAdvisor, решающее проблемы с инициализацией и методами.
"""

import logging
from seo_ai_models.models.seo_advisor.enhanced_advisor import EnhancedSEOAdvisor

# Сохраняем оригинальный метод инициализации
original_init = EnhancedSEOAdvisor.__init__

# Определяем исправленный метод инициализации
def fixed_init(self, llm_api_key=None, industry='default'):
    """
    Исправленная инициализация, совместимая с базовым классом.
    
    Args:
        llm_api_key: API ключ для LLM-сервисов (опционально)
        industry: Отрасль для анализа
    """
    # Создаем улучшенные компоненты
    from seo_ai_models.models.seo_advisor.analyzers.enhanced_content_analyzer import EnhancedContentAnalyzer
    from seo_ai_models.models.seo_advisor.analyzers.semantic_analyzer import SemanticAnalyzer
    from seo_ai_models.models.seo_advisor.analyzers.eeat.enhanced_eeat_analyzer import EnhancedEEATAnalyzer
    from seo_ai_models.models.seo_advisor.predictors.calibrated_rank_predictor import CalibratedRankPredictor
    from seo_ai_models.models.seo_advisor.suggester.suggester import Suggester
    
    # Вызываем конструктор базового класса с правильными параметрами
    super(EnhancedSEOAdvisor, self).__init__(industry=industry)
    
    # Заменяем компоненты на улучшенные версии
    self.content_analyzer = EnhancedContentAnalyzer()
    self.semantic_analyzer = SemanticAnalyzer()
    self.eeat_analyzer = EnhancedEEATAnalyzer()
    self.rank_predictor = CalibratedRankPredictor(industry=industry)
    self.suggester = Suggester()
    
    # Настройка логгирования
    self.logger = logging.getLogger(__name__)
    
    # Инициализируем LLM-компоненты, если указан API ключ
    self.llm_factory = None
    self.llm_components = {}
    self.llm_api_key = llm_api_key
    
    if llm_api_key:
        try:
            # Пробуем импортировать LLM-компоненты
            from seo_ai_models.models.seo_advisor.llm_integration_adapter import LLMAdvisorFactory
            
            self.llm_factory = LLMAdvisorFactory(llm_api_key)
            self._initialize_llm_components()
            self.logger.info("LLM-компоненты успешно инициализированы")
        except ImportError as e:
            self.logger.warning(f"Не удалось импортировать LLM-компоненты: {e}")
            self.llm_factory = None
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации LLM-компонентов: {e}")
            self.llm_factory = None
    
    # Флаг доступности LLM-функций
    self.llm_enabled = bool(self.llm_factory and self.llm_components)
    
    if not self.llm_enabled:
        self.logger.warning("LLM-компоненты недоступны")

# Переопределяем метод analyze_content для совместимости с базовым классом
def fixed_analyze_content(self, content, target_keywords=None, use_llm=True, llm_budget=None, url=None, title=None, html_content=None):
    """
    Исправленный метод analyze_content, совместимый с базовым классом.
    
    Args:
        content: Текст для анализа
        target_keywords: Целевые ключевые слова (опционально)
        use_llm: Использовать LLM-компоненты
        llm_budget: Бюджет для LLM-анализа
        url: URL страницы (опционально)
        title: Заголовок страницы (опционально)
        html_content: HTML-контент (опционально)
        
    Returns:
        Dict: Результат анализа
    """
    # Если ключевые слова не указаны, создаем пустой список
    if target_keywords is None:
        target_keywords = []
    
    # Используем базовый анализ из родительского класса
    try:
        # Получаем базовый результат от оригинального метода
        base_result = super(EnhancedSEOAdvisor, self).analyze_content(content, target_keywords)
        
        # Преобразуем отчет в словарь для удобства работы
        result = {
            'timestamp': base_result.timestamp,
            'content': base_result.content_metrics,
            'keywords': base_result.keyword_analysis,
            'predicted_position': base_result.predicted_position,
            'feature_scores': base_result.feature_scores,
            'content_quality': {
                'scores': base_result.content_quality.content_scores,
                'strengths': base_result.content_quality.strengths,
                'weaknesses': base_result.content_quality.weaknesses,
                'improvements': base_result.content_quality.potential_improvements,
            },
            'recommendations': base_result.recommendations,
            'priorities': base_result.priorities,
            'industry': base_result.industry,
            'eeat': {
                'experience_score': base_result.content_metrics.get('expertise_score', 0),
                'expertise_score': base_result.content_metrics.get('expertise_score', 0),
                'authority_score': base_result.content_metrics.get('authority_score', 0),
                'trust_score': base_result.content_metrics.get('trust_score', 0),
                'overall_score': base_result.content_metrics.get('overall_eeat_score', 0)
            },
            'overall_score': base_result.content_quality.content_scores.get('overall_quality', 0)
        }
    except Exception as e:
        # Если возникла ошибка с базовым анализом, создаем базовый результат
        self.logger.error(f"Ошибка при выполнении базового анализа: {e}")
        result = {
            'content': {},
            'keywords': {},
            'eeat': {},
            'recommendations': {},
            'priorities': [],
            'industry': self.industry,
            'overall_score': 0.5,
            'error': str(e)
        }
    
    # Если HTML контент доступен, выполняем дополнительный анализ
    if html_content:
        try:
            html_analysis = self.content_analyzer.analyze_html(html_content)
            result['html_analysis'] = html_analysis
            
            # Обновляем оценку страницы с учетом HTML-анализа
            if 'meta_score' in html_analysis:
                result['content']['meta_score'] = html_analysis['meta_score']
            
            if 'tech_seo_score' in html_analysis:
                result['content']['tech_seo_score'] = html_analysis['tech_seo_score']
        except Exception as e:
            self.logger.error(f"Ошибка при анализе HTML: {e}")
            result['html_analysis_error'] = str(e)
    
    # Если LLM-компоненты недоступны или отключены, возвращаем базовый результат
    if not use_llm or not self.llm_enabled:
        return result
    
    # Расширяем анализ с помощью LLM-компонентов
    try:
        llm_result = self._analyze_with_llm(content, llm_budget)
        
        # Объединяем результаты
        result['llm_analysis'] = llm_result
        
        # Обновляем оценки на основе LLM-анализа
        self._update_scores_with_llm(result, llm_result)
    except Exception as e:
        self.logger.error(f"Ошибка при LLM-анализе: {e}")
        result['llm_analysis_error'] = str(e)
    
    return result

# Заменяем методы
EnhancedSEOAdvisor.__init__ = fixed_init
EnhancedSEOAdvisor.analyze_content = fixed_analyze_content

# Добавляем метод analyze_content_for_llm, если его нет в классе
if not hasattr(EnhancedSEOAdvisor, '_update_scores_with_llm'):
    def _update_scores_with_llm(self, result, llm_result):
        """
        Обновляет оценки на основе LLM-анализа.
        
        Args:
            result: Текущий результат анализа
            llm_result: Результат LLM-анализа
        """
        # Обновляем общую оценку с учетом LLM-анализа
        if 'overall_score' in result:
            # Получаем оценки LLM
            llm_compatibility = 0
            llm_citability = 0
            llm_eeat = 0
            
            # Извлекаем оценки совместимости
            if 'compatibility' in llm_result:
                comp_scores = llm_result['compatibility'].get('compatibility_scores', {})
                if 'overall' in comp_scores:
                    llm_compatibility = comp_scores['overall']
            
            # Извлекаем оценку цитируемости
            if 'citability' in llm_result:
                llm_citability = llm_result['citability'].get('citability_score', 0)
            
            # Извлекаем оценку E-E-A-T
            if 'eeat' in llm_result:
                eeat_scores = llm_result['eeat'].get('eeat_scores', {})
                if 'overall' in eeat_scores:
                    llm_eeat = eeat_scores['overall']
            
            # Вычисляем среднюю оценку LLM
            llm_scores = [score for score in [llm_compatibility, llm_citability, llm_eeat] if score > 0]
            if llm_scores:
                llm_score = sum(llm_scores) / len(llm_scores)
                
                # Обновляем общую оценку (70% базовая, 30% LLM)
                result['overall_score'] = result['overall_score'] * 0.7 + llm_score * 0.3
                result['llm_score_contribution'] = llm_score
    
    # Добавляем метод в класс
    EnhancedSEOAdvisor._update_scores_with_llm = _update_scores_with_llm
