"""
Адаптер для интеграции LLM-компонентов с SEO Advisor.

Модуль предоставляет адаптеры и фабрики для интеграции компонентов LLM
с существующей системой SEO Advisor.
"""

import logging
from typing import Dict, List, Any, Optional, Union

# Импорты из SEO Advisor
from seo_ai_models.models.seo_advisor.advisor import SEOAdvisor
from seo_ai_models.models.seo_advisor.analyzers.content_analyzer import ContentAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.semantic_analyzer import SemanticAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.eeat.eeat_analyzer import EEATAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.eeat.enhanced_eeat_analyzer import EnhancedEEATAnalyzer
from seo_ai_models.models.seo_advisor.predictors.calibrated_rank_predictor import CalibratedRankPredictor
from seo_ai_models.models.seo_advisor.suggester.suggester import Suggester

# Импорты из LLM-интеграции
from seo_ai_models.models.llm_integration.service.llm_service import LLMService
from seo_ai_models.models.llm_integration.service.prompt_generator import PromptGenerator
from seo_ai_models.models.llm_integration.service.multi_model_agent import MultiModelAgent
from seo_ai_models.models.llm_integration.service.cost_estimator import CostEstimator

from seo_ai_models.models.llm_integration.analyzers.llm_compatibility_analyzer import LLMCompatibilityAnalyzer
from seo_ai_models.models.llm_integration.analyzers.citability_scorer import CitabilityScorer
from seo_ai_models.models.llm_integration.analyzers.content_structure_enhancer import ContentStructureEnhancer
from seo_ai_models.models.llm_integration.analyzers.llm_eeat_analyzer import LLMEEATAnalyzer

from seo_ai_models.models.llm_integration.dimension_map.feature_importance_analyzer import FeatureImportanceAnalyzer
from seo_ai_models.models.llm_integration.dimension_map.semantic_structure_extractor import SemanticStructureExtractor


class LLMAdvisorFactory:
    """
    Фабрика для создания компонентов LLM-интеграции.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализирует фабрику LLM-компонентов.
        
        Args:
            api_key: API ключ для LLM-провайдера (опционально)
        """
        self.logger = logging.getLogger(__name__)
        self.services = {}
        
        try:
            # Создаем основные сервисы
            llm_service = LLMService()
            prompt_generator = PromptGenerator()
            
            # Добавляем провайдера, если указан API ключ
            if api_key:
                llm_service.add_provider("openai", api_key=api_key, model="gpt-4o-mini")
                
                # Создаем вспомогательные сервисы
                multi_model_agent = MultiModelAgent(llm_service, prompt_generator)
                cost_estimator = CostEstimator()
                
                # Сохраняем сервисы
                self.services = {
                    "llm_service": llm_service,
                    "prompt_generator": prompt_generator,
                    "multi_model_agent": multi_model_agent,
                    "cost_estimator": cost_estimator
                }
                
                self.logger.info("LLM-сервисы успешно инициализированы")
            else:
                self.logger.warning("API ключ не указан. LLM-функции будут недоступны")
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации LLM-сервисов: {e}")
    
    def create_compatibility_analyzer(self) -> Optional[LLMCompatibilityAnalyzer]:
        """
        Создает анализатор совместимости с LLM.
        
        Returns:
            Optional[LLMCompatibilityAnalyzer]: Анализатор совместимости или None
        """
        if not self.services:
            return None
        
        try:
            return LLMCompatibilityAnalyzer(
                self.services["llm_service"],
                self.services["prompt_generator"]
            )
        except Exception as e:
            self.logger.error(f"Ошибка при создании анализатора совместимости: {e}")
            return None
    
    def create_citability_scorer(self) -> Optional[CitabilityScorer]:
        """
        Создает оценщик цитируемости.
        
        Returns:
            Optional[CitabilityScorer]: Оценщик цитируемости или None
        """
        if not self.services:
            return None
        
        try:
            return CitabilityScorer(
                self.services["llm_service"],
                self.services["prompt_generator"]
            )
        except Exception as e:
            self.logger.error(f"Ошибка при создании оценщика цитируемости: {e}")
            return None
    
    def create_structure_enhancer(self) -> Optional[ContentStructureEnhancer]:
        """
        Создает улучшитель структуры контента.
        
        Returns:
            Optional[ContentStructureEnhancer]: Улучшитель структуры или None
        """
        if not self.services:
            return None
        
        try:
            return ContentStructureEnhancer(
                self.services["llm_service"],
                self.services["prompt_generator"]
            )
        except Exception as e:
            self.logger.error(f"Ошибка при создании улучшителя структуры: {e}")
            return None
    
    def create_llm_eeat_analyzer(self) -> Optional[LLMEEATAnalyzer]:
        """
        Создает анализатор E-E-A-T для LLM.
        
        Returns:
            Optional[LLMEEATAnalyzer]: Анализатор E-E-A-T или None
        """
        if not self.services:
            return None
        
        try:
            return LLMEEATAnalyzer(
                self.services["llm_service"],
                self.services["prompt_generator"]
            )
        except Exception as e:
            self.logger.error(f"Ошибка при создании анализатора E-E-A-T: {e}")
            return None
    
    def create_semantic_extractor(self) -> Optional[SemanticStructureExtractor]:
        """
        Создает экстрактор семантической структуры.
        
        Returns:
            Optional[SemanticStructureExtractor]: Экстрактор структуры или None
        """
        if not self.services:
            return None
        
        try:
            return SemanticStructureExtractor(
                self.services["llm_service"],
                self.services["prompt_generator"]
            )
        except Exception as e:
            self.logger.error(f"Ошибка при создании экстрактора структуры: {e}")
            return None


class LLMEnhancedSEOAdvisor(SEOAdvisor):
    """
    Расширенная версия SEO Advisor с поддержкой LLM-компонентов.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Инициализирует расширенный SEO Advisor.
        
        Args:
            api_key: API ключ для LLM-провайдера (опционально)
            **kwargs: Дополнительные параметры для базового SEO Advisor
        """
        # Инициализируем базовый SEO Advisor
        super().__init__(**kwargs)
        
        # Инициализируем фабрику LLM-компонентов
        self.llm_factory = LLMAdvisorFactory(api_key)
        
        # Создаем LLM-компоненты
        self.llm_compatibility_analyzer = self.llm_factory.create_compatibility_analyzer()
        self.citability_scorer = self.llm_factory.create_citability_scorer()
        self.structure_enhancer = self.llm_factory.create_structure_enhancer()
        self.llm_eeat_analyzer = self.llm_factory.create_llm_eeat_analyzer()
        self.semantic_extractor = self.llm_factory.create_semantic_extractor()
        
        # Флаг доступности LLM-функций
        self.llm_enabled = all([
            self.llm_compatibility_analyzer,
            self.citability_scorer,
            self.structure_enhancer,
            self.llm_eeat_analyzer,
            self.semantic_extractor
        ])
        
        self.logger = logging.getLogger(__name__)
        
        if self.llm_enabled:
            self.logger.info("LLM-функции успешно активированы")
        else:
            self.logger.warning("LLM-функции недоступны. Будут использованы только базовые функции")
    
    def analyze_content(self, content: str, url: Optional[str] = None, 
                       title: Optional[str] = None, 
                       use_llm: bool = True, llm_budget: Optional[float] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Анализирует контент с использованием базовых и LLM-компонентов.
        
        Args:
            content: Текст для анализа
            url: URL страницы (опционально)
            title: Заголовок страницы (опционально)
            use_llm: Использовать LLM-компоненты (если доступны)
            llm_budget: Бюджет для LLM-анализа в рублях (опционально)
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict[str, Any]: Результат анализа
        """
        # Получаем базовый анализ от SEO Advisor
        base_result = super().analyze_content(content, url, title, **kwargs)
        
        # Если LLM-функции отключены или недоступны, возвращаем только базовый анализ
        if not use_llm or not self.llm_enabled:
            return base_result
        
        # Выполняем LLM-анализ
        llm_result = self._perform_llm_analysis(content, llm_budget, **kwargs)
        
        # Объединяем результаты
        combined_result = self._combine_analysis_results(base_result, llm_result)
        
        return combined_result
    
    def enhance_content(self, content: str, enhancement_type: str = "structure",
                      llm_budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Улучшает контент с использованием LLM-компонентов.
        
        Args:
            content: Текст для улучшения
            enhancement_type: Тип улучшения (structure, semantic, etc.)
            llm_budget: Бюджет для LLM-улучшения в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Результат улучшения
        """
        if not self.llm_enabled:
            self.logger.warning("LLM-функции недоступны. Невозможно улучшить контент")
            return {
                "success": False,
                "error": "LLM-функции недоступны",
                "original_content": content
            }
        
        try:
            # Выполняем улучшение в зависимости от типа
            if enhancement_type == "structure":
                # Улучшаем структуру контента
                result = self.structure_enhancer.enhance_structure(
                    content, content_type="article", budget=llm_budget
                )
                
                return {
                    "success": True,
                    "enhanced_content": result.get("enhanced_content", ""),
                    "changes": result.get("changes", {}),
                    "original_content": content,
                    "enhancement_type": enhancement_type
                }
            
            else:
                self.logger.warning(f"Неизвестный тип улучшения: {enhancement_type}")
                return {
                    "success": False,
                    "error": f"Неизвестный тип улучшения: {enhancement_type}",
                    "original_content": content
                }
        
        except Exception as e:
            self.logger.error(f"Ошибка при улучшении контента: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_content": content
            }
    
    def _perform_llm_analysis(self, content: str, llm_budget: Optional[float] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Выполняет анализ контента с использованием LLM-компонентов.
        
        Args:
            content: Текст для анализа
            llm_budget: Бюджет для LLM-анализа в рублях (опционально)
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict[str, Any]: Результат LLM-анализа
        """
        # Распределяем бюджет на компоненты анализа
        component_budget = None
        if llm_budget is not None:
            # Распределяем бюджет на 4 компонента
            component_budget = llm_budget / 4
        
        result = {}
        
        try:
            # Анализ совместимости с LLM
            compatibility_result = self.llm_compatibility_analyzer.analyze_compatibility(
                content, budget=component_budget
            )
            result["llm_compatibility"] = compatibility_result
            
            # Оценка цитируемости
            citability_result = self.citability_scorer.score_citability(
                content, budget=component_budget
            )
            result["llm_citability"] = citability_result
            
            # Анализ E-E-A-T для LLM
            eeat_result = self.llm_eeat_analyzer.analyze_eeat(
                content, budget=component_budget
            )
            result["llm_eeat"] = eeat_result
            
            # Извлечение семантической структуры
            semantic_result = self.semantic_extractor.extract_semantic_structure(
                content, method="llm", budget=component_budget
            )
            result["llm_semantic"] = semantic_result
            
            # Добавляем общие метрики и статистику
            result["llm_metrics"] = {
                "overall_compatibility": compatibility_result.get("compatibility_scores", {}).get("overall", 0),
                "citability_score": citability_result.get("citability_score", 0),
                "eeat_score": eeat_result.get("eeat_scores", {}).get("overall", 0),
                "total_cost": (
                    compatibility_result.get("cost", 0) +
                    citability_result.get("cost", 0) +
                    eeat_result.get("cost", 0) +
                    semantic_result.get("cost", 0)
                )
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении LLM-анализа: {e}")
            result["error"] = str(e)
        
        return result
    
    def _combine_analysis_results(self, base_result: Dict[str, Any], 
                                llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Объединяет результаты базового и LLM-анализа.
        
        Args:
            base_result: Результат базового анализа
            llm_result: Результат LLM-анализа
            
        Returns:
            Dict[str, Any]: Объединенный результат анализа
        """
        # Создаем копию базового результата
        combined = base_result.copy()
        
        # Добавляем разделы LLM-анализа
        combined["llm_analysis"] = llm_result
        
        # Обновляем общую оценку с учетом LLM-метрик
        if "llm_metrics" in llm_result and "overall_score" in combined:
            llm_score = (
                llm_result["llm_metrics"].get("overall_compatibility", 0) +
                llm_result["llm_metrics"].get("citability_score", 0) +
                llm_result["llm_metrics"].get("eeat_score", 0)
            ) / 3
            
            # Объединяем базовую оценку и LLM-оценку (с весами 0.6 и 0.4)
            combined["overall_score"] = combined["overall_score"] * 0.6 + llm_score * 0.4
        
        # Обновляем предложения по улучшению
        if "suggestions" in combined:
            llm_suggestions = self._extract_llm_suggestions(llm_result)
            combined["suggestions"].extend(llm_suggestions)
        
        # Добавляем флаг использования LLM
        combined["llm_enabled"] = True
        
        return combined
    
    def _extract_llm_suggestions(self, llm_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Извлекает предложения по улучшению из результатов LLM-анализа.
        
        Args:
            llm_result: Результат LLM-анализа
            
        Returns:
            List[Dict[str, Any]]: Список предложений по улучшению
        """
        suggestions = []
        
        # Извлекаем предложения из анализа совместимости
        if "llm_compatibility" in llm_result:
            compatibility = llm_result["llm_compatibility"]
            for factor, improvements in compatibility.get("suggested_improvements", {}).items():
                for improvement in improvements:
                    suggestions.append({
                        "type": "llm_compatibility",
                        "category": factor,
                        "description": improvement,
                        "importance": 7  # Средняя важность
                    })
        
        # Извлекаем предложения из оценки цитируемости
        if "llm_citability" in llm_result:
            citability = llm_result["llm_citability"]
            for factor, improvements in citability.get("suggested_improvements", {}).items():
                for improvement in improvements:
                    suggestions.append({
                        "type": "llm_citability",
                        "category": factor,
                        "description": improvement,
                        "importance": 8  # Высокая важность
                    })
        
        # Извлекаем предложения из анализа E-E-A-T
        if "llm_eeat" in llm_result:
            eeat = llm_result["llm_eeat"]
            for component, improvements in eeat.get("suggested_improvements", {}).items():
                for improvement in improvements:
                    suggestions.append({
                        "type": "llm_eeat",
                        "category": component,
                        "description": improvement,
                        "importance": 9  # Очень высокая важность
                    })
        
        return suggestions
