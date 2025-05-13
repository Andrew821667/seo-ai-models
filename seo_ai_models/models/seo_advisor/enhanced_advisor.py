"""
Расширенная версия SEO Advisor с поддержкой LLM-компонентов.

Модуль предоставляет расширенную версию SEO Advisor,
которая использует не только традиционные методы анализа,
но и компоненты на основе LLM для более глубокого анализа контента.
"""

import logging
from typing import Dict, List, Any, Optional, Union

from seo_ai_models.models.seo_advisor.advisor import SEOAdvisor
from seo_ai_models.models.seo_advisor.analyzers.enhanced_content_analyzer import EnhancedContentAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.semantic_analyzer import SemanticAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.eeat.enhanced_eeat_analyzer import EnhancedEEATAnalyzer
from seo_ai_models.models.seo_advisor.predictors.calibrated_rank_predictor import CalibratedRankPredictor
from seo_ai_models.models.seo_advisor.suggester.suggester import Suggester

# Импортируем адаптер LLM-интеграции
from seo_ai_models.models.seo_advisor.llm_integration_adapter import LLMAdvisorFactory


class EnhancedSEOAdvisor(SEOAdvisor):
    """
    Расширенная версия SEO Advisor с поддержкой LLM-компонентов.
    """
    
    def __init__(self, llm_api_key: Optional[str] = None, **kwargs):
        """
        Инициализирует расширенный SEO Advisor.
        
        Args:
            llm_api_key: API ключ для LLM-сервисов (опционально)
            **kwargs: Дополнительные параметры для базового SEO Advisor
        """
        # Инициализируем улучшенные компоненты
        content_analyzer = kwargs.pop('content_analyzer', EnhancedContentAnalyzer())
        semantic_analyzer = kwargs.pop('semantic_analyzer', SemanticAnalyzer())
        eeat_analyzer = kwargs.pop('eeat_analyzer', EnhancedEEATAnalyzer())
        predictor = kwargs.pop('predictor', CalibratedRankPredictor())
        suggester = kwargs.pop('suggester', Suggester())
        
        # Инициализируем базовый SEO Advisor с улучшенными компонентами
        super().__init__(
            content_analyzer=content_analyzer,
            semantic_analyzer=semantic_analyzer,
            eeat_analyzer=eeat_analyzer,
            predictor=predictor,
            suggester=suggester,
            **kwargs
        )
        
        # Инициализируем фабрику LLM-компонентов, если указан API ключ
        self.llm_factory = None
        self.llm_components = {}
        
        if llm_api_key:
            self.llm_factory = LLMAdvisorFactory(llm_api_key)
            self._initialize_llm_components()
        
        # Флаг доступности LLM-функций
        self.llm_enabled = llm_api_key is not None and bool(self.llm_components)
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
    
    def _initialize_llm_components(self) -> None:
        """
        Инициализирует компоненты LLM-интеграции.
        """
        if not self.llm_factory:
            return
        
        # Создаем компоненты LLM-интеграции
        self.llm_components = {
            "compatibility_analyzer": self.llm_factory.create_compatibility_analyzer(),
            "citability_scorer": self.llm_factory.create_citability_scorer(),
            "structure_enhancer": self.llm_factory.create_structure_enhancer(),
            "eeat_analyzer": self.llm_factory.create_llm_eeat_analyzer(),
            "semantic_extractor": self.llm_factory.create_semantic_extractor()
        }
        
        # Проверяем, все ли компоненты созданы успешно
        self.llm_enabled = all(self.llm_components.values())
        
        if self.llm_enabled:
            self.logger.info("LLM-компоненты успешно инициализированы")
        else:
            self.logger.warning("Не все LLM-компоненты удалось инициализировать")
    
    def analyze_content(self, content: str, url: Optional[str] = None, 
                       title: Optional[str] = None, 
                       use_llm: bool = True, llm_budget: Optional[float] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Анализирует контент с использованием традиционных и LLM-компонентов.
        
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
        # Получаем результат анализа от базового SEO Advisor
        result = super().analyze_content(content, url, title, **kwargs)
        
        # Если LLM-компоненты недоступны или отключены, возвращаем только базовый результат
        if not use_llm or not self.llm_enabled:
            return result
        
        # Расширяем анализ с помощью LLM-компонентов
        llm_result = self._analyze_with_llm(content, llm_budget)
        
        # Объединяем результаты
        result = self._merge_results(result, llm_result)
        
        # Обновляем оценки и предложения
        self._update_scores_and_suggestions(result)
        
        return result
    
    def _analyze_with_llm(self, content: str, budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Выполняет анализ контента с использованием LLM-компонентов.
        
        Args:
            content: Текст для анализа
            budget: Бюджет для LLM-анализа в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Результат LLM-анализа
        """
        # Результат LLM-анализа
        llm_result = {}
        
        # Распределяем бюджет между компонентами
        component_budget = None
        if budget is not None:
            component_budget = budget / len(self.llm_components)
        
        # Анализ совместимости с LLM
        compatibility_analyzer = self.llm_components.get("compatibility_analyzer")
        if compatibility_analyzer:
            try:
                compatibility_result = compatibility_analyzer.analyze_compatibility(
                    content, budget=component_budget
                )
                llm_result["compatibility"] = compatibility_result
            except Exception as e:
                self.logger.error(f"Ошибка при анализе совместимости: {e}")
        
        # Оценка цитируемости
        citability_scorer = self.llm_components.get("citability_scorer")
        if citability_scorer:
            try:
                citability_result = citability_scorer.score_citability(
                    content, budget=component_budget
                )
                llm_result["citability"] = citability_result
            except Exception as e:
                self.logger.error(f"Ошибка при оценке цитируемости: {e}")
        
        # Анализ E-E-A-T для LLM
        eeat_analyzer = self.llm_components.get("eeat_analyzer")
        if eeat_analyzer:
            try:
                eeat_result = eeat_analyzer.analyze_eeat(
                    content, budget=component_budget
                )
                llm_result["eeat"] = eeat_result
            except Exception as e:
                self.logger.error(f"Ошибка при анализе E-E-A-T: {e}")
        
        # Извлечение семантической структуры
        semantic_extractor = self.llm_components.get("semantic_extractor")
        if semantic_extractor:
            try:
                semantic_result = semantic_extractor.extract_semantic_structure(
                    content, method="llm", budget=component_budget
                )
                llm_result["semantic"] = semantic_result
            except Exception as e:
                self.logger.error(f"Ошибка при извлечении семантической структуры: {e}")
        
        # Добавляем суммарную стоимость анализа
        total_cost = sum(
            result.get("cost", 0) 
            for result in llm_result.values() 
            if isinstance(result, dict)
        )
        llm_result["total_cost"] = total_cost
        
        return llm_result
    
    def _merge_results(self, base_result: Dict[str, Any], 
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
        result = base_result.copy()
        
        # Добавляем результаты LLM-анализа
        result["llm_analysis"] = llm_result
        
        # Объединяем семантическую информацию
        if "semantic" in llm_result and "semantic" in result:
            result["semantic"] = self._merge_semantic_data(
                result["semantic"], llm_result["semantic"]
            )
        
        # Продолжение enhanced_advisor.py
        # Объединяем информацию о E-E-A-T
        if "eeat" in llm_result and "eeat" in result:
            result["eeat"] = self._merge_eeat_data(
                result["eeat"], llm_result["eeat"]
            )
        
        # Добавляем метрики LLM
        result["llm_metrics"] = self._extract_llm_metrics(llm_result)
        
        return result
    
    def _merge_semantic_data(self, base_semantic: Dict[str, Any], 
                           llm_semantic: Dict[str, Any]) -> Dict[str, Any]:
        """
        Объединяет семантические данные из базового и LLM-анализа.
        
        Args:
            base_semantic: Семантические данные из базового анализа
            llm_semantic: Семантические данные из LLM-анализа
            
        Returns:
            Dict[str, Any]: Объединенные семантические данные
        """
        # Создаем копию базовых данных
        result = base_semantic.copy()
        
        # Добавляем данные из LLM-анализа
        result["llm_topics"] = llm_semantic.get("topics", [])
        result["llm_keywords"] = llm_semantic.get("keywords", [])
        result["llm_semantic_clusters"] = llm_semantic.get("semantic_clusters", [])
        
        # Объединяем ключевые слова
        all_keywords = set(result.get("keywords", []))
        all_keywords.update(llm_semantic.get("keywords", []))
        result["keywords"] = list(all_keywords)
        
        return result
    
    def _merge_eeat_data(self, base_eeat: Dict[str, Any], 
                        llm_eeat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Объединяет данные E-E-A-T из базового и LLM-анализа.
        
        Args:
            base_eeat: Данные E-E-A-T из базового анализа
            llm_eeat: Данные E-E-A-T из LLM-анализа
            
        Returns:
            Dict[str, Any]: Объединенные данные E-E-A-T
        """
        # Создаем копию базовых данных
        result = base_eeat.copy()
        
        # Добавляем данные из LLM-анализа
        result["llm_scores"] = llm_eeat.get("eeat_scores", {})
        result["llm_analysis"] = llm_eeat.get("eeat_analysis", {})
        result["llm_improvements"] = llm_eeat.get("suggested_improvements", {})
        
        # Обновляем общую оценку E-E-A-T
        base_score = result.get("overall_score", 0)
        llm_score = llm_eeat.get("eeat_scores", {}).get("overall", 0)
        
        # Объединяем оценки с весами (60% базовая, 40% LLM)
        if base_score and llm_score:
            result["overall_score"] = base_score * 0.6 + llm_score * 0.4
        
        return result
    
    def _extract_llm_metrics(self, llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлекает метрики из результата LLM-анализа.
        
        Args:
            llm_result: Результат LLM-анализа
            
        Returns:
            Dict[str, Any]: Метрики LLM
        """
        metrics = {
            "total_cost": llm_result.get("total_cost", 0)
        }
        
        # Добавляем оценки совместимости
        if "compatibility" in llm_result:
            for factor, score in llm_result["compatibility"].get("compatibility_scores", {}).items():
                metrics[f"compatibility_{factor}"] = score
        
        # Добавляем оценку цитируемости
        if "citability" in llm_result:
            metrics["citability_score"] = llm_result["citability"].get("citability_score", 0)
            
            # Добавляем оценки по факторам
            for factor, score in llm_result["citability"].get("factor_scores", {}).items():
                metrics[f"citability_{factor}"] = score
        
        # Добавляем оценки E-E-A-T
        if "eeat" in llm_result:
            for component, score in llm_result["eeat"].get("eeat_scores", {}).items():
                metrics[f"eeat_{component}"] = score
        
        return metrics
    
    def _update_scores_and_suggestions(self, result: Dict[str, Any]) -> None:
        """
        Обновляет оценки и предложения с учетом LLM-анализа.
        
        Args:
            result: Результат анализа для обновления
        """
        # Обновляем общую оценку
        if "overall_score" in result and "llm_metrics" in result:
            # Получаем оценки LLM
            llm_compatibility = result["llm_metrics"].get("compatibility_overall", 0)
            llm_citability = result["llm_metrics"].get("citability_score", 0)
            llm_eeat = result["llm_metrics"].get("eeat_overall", 0)
            
            # Вычисляем среднюю оценку LLM
            llm_scores = [score for score in [llm_compatibility, llm_citability, llm_eeat] if score > 0]
            if llm_scores:
                llm_score = sum(llm_scores) / len(llm_scores)
                
                # Обновляем общую оценку (70% базовая, 30% LLM)
                result["overall_score"] = result["overall_score"] * 0.7 + llm_score * 0.3
        
        # Добавляем предложения из LLM-анализа
        if "suggestions" in result and "llm_analysis" in result:
            llm_suggestions = self._create_llm_suggestions(result["llm_analysis"])
            result["suggestions"].extend(llm_suggestions)
            
            # Сортируем предложения по важности
            result["suggestions"] = sorted(
                result["suggestions"], 
                key=lambda x: x.get("importance", 0),
                reverse=True
            )
    
    def _create_llm_suggestions(self, llm_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Создает предложения по улучшению на основе LLM-анализа.
        
        Args:
            llm_analysis: Результат LLM-анализа
            
        Returns:
            List[Dict[str, Any]]: Список предложений по улучшению
        """
        suggestions = []
        
        # Добавляем предложения из анализа совместимости
        if "compatibility" in llm_analysis:
            for category, improvements in llm_analysis["compatibility"].get("suggested_improvements", {}).items():
                for improvement in improvements:
                    suggestions.append({
                        "type": "llm_compatibility",
                        "category": category,
                        "description": improvement,
                        "importance": 7  # Средняя важность
                    })
        
        # Добавляем предложения из оценки цитируемости
        if "citability" in llm_analysis:
            for category, improvements in llm_analysis["citability"].get("suggested_improvements", {}).items():
                for improvement in improvements:
                    suggestions.append({
                        "type": "llm_citability",
                        "category": category,
                        "description": improvement,
                        "importance": 8  # Высокая важность
                    })
        
        # Добавляем предложения из анализа E-E-A-T
        if "eeat" in llm_analysis:
            for category, improvements in llm_analysis["eeat"].get("suggested_improvements", {}).items():
                for improvement in improvements:
                    suggestions.append({
                        "type": "llm_eeat",
                        "category": category,
                        "description": improvement,
                        "importance": 9  # Очень высокая важность
                    })
        
        # Добавляем предложения из семантического анализа
        if "semantic" in llm_analysis:
            for gap in llm_analysis["semantic"].get("semantic_gaps", []):
                suggestions.append({
                    "type": "llm_semantic",
                    "category": "semantic_gap",
                    "description": f"Заполнить семантический пробел: {gap}",
                    "importance": 8  # Высокая важность
                })
        
        return suggestions
    
    def enhance_content(self, content: str, enhancement_type: str = "structure",
                      budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Улучшает контент с использованием LLM-компонентов.
        
        Args:
            content: Текст для улучшения
            enhancement_type: Тип улучшения (structure, semantic, etc.)
            budget: Бюджет для улучшения в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Результат улучшения
        """
        # Проверяем доступность LLM-компонентов
        if not self.llm_enabled:
            return {
                "success": False,
                "error": "LLM-компоненты недоступны",
                "original_content": content
            }
        
        # Выполняем улучшение в зависимости от типа
        if enhancement_type == "structure":
            # Улучшаем структуру контента
            structure_enhancer = self.llm_components.get("structure_enhancer")
            if structure_enhancer:
                try:
                    result = structure_enhancer.enhance_structure(
                        content, content_type="article", budget=budget
                    )
                    
                    return {
                        "success": True,
                        "enhancement_type": enhancement_type,
                        "enhanced_content": result.get("enhanced_content", ""),
                        "original_content": content,
                        "changes": result.get("changes", {}),
                        "cost": result.get("cost", 0)
                    }
                except Exception as e:
                    self.logger.error(f"Ошибка при улучшении структуры: {e}")
                    return {
                        "success": False,
                        "error": str(e),
                        "original_content": content
                    }
            else:
                return {
                    "success": False,
                    "error": "Компонент улучшения структуры недоступен",
                    "original_content": content
                }
        
        # Если тип улучшения не поддерживается
        return {
            "success": False,
            "error": f"Неподдерживаемый тип улучшения: {enhancement_type}",
            "original_content": content
        }
    
    def analyze_content_for_llm(self, content: str, 
                              llm_type: str = "generic",
                              budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Анализирует контент специально для оптимизации под LLM.
        
        Args:
            content: Текст для анализа
            llm_type: Тип LLM (generic, chat, search)
            budget: Бюджет для анализа в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Результат анализа для LLM
        """
        # Проверяем доступность LLM-компонентов
        if not self.llm_enabled:
            return {
                "success": False,
                "error": "LLM-компоненты недоступны"
            }
        
        # Распределяем бюджет между компонентами
        component_budget = None
        if budget is not None:
            component_budget = budget / 3  # Для трех компонентов
        
        result = {
            "success": True,
            "llm_type": llm_type,
            "content_length": len(content)
        }
        
        # Анализ совместимости с LLM
        compatibility_analyzer = self.llm_components.get("compatibility_analyzer")
        if compatibility_analyzer:
            try:
                compatibility_result = compatibility_analyzer.analyze_compatibility(
                    content, llm_types=[llm_type], budget=component_budget
                )
                result["compatibility"] = compatibility_result
            except Exception as e:
                self.logger.error(f"Ошибка при анализе совместимости: {e}")
                result["compatibility_error"] = str(e)
        
        # Оценка цитируемости
        citability_scorer = self.llm_components.get("citability_scorer")
        if citability_scorer:
            try:
                citability_result = citability_scorer.score_citability(
                    content, budget=component_budget
                )
                result["citability"] = citability_result
            except Exception as e:
                self.logger.error(f"Ошибка при оценке цитируемости: {e}")
                result["citability_error"] = str(e)
        
        # Анализ E-E-A-T для LLM
        eeat_analyzer = self.llm_components.get("eeat_analyzer")
        if eeat_analyzer:
            try:
                eeat_result = eeat_analyzer.analyze_eeat(
                    content, budget=component_budget
                )
                result["eeat"] = eeat_result
            except Exception as e:
                self.logger.error(f"Ошибка при анализе E-E-A-T: {e}")
                result["eeat_error"] = str(e)
        
        # Добавляем предложения по улучшению для LLM
        suggestions = []
        
        # Из совместимости
        if "compatibility" in result:
            for category, improvements in result["compatibility"].get("suggested_improvements", {}).items():
                for improvement in improvements:
                    suggestions.append({
                        "type": "compatibility",
                        "category": category,
                        "description": improvement
                    })
        
        # Из цитируемости
        if "citability" in result:
            for category, improvements in result["citability"].get("suggested_improvements", {}).items():
                for improvement in improvements:
                    suggestions.append({
                        "type": "citability",
                        "category": category,
                        "description": improvement
                    })
        
        # Из E-E-A-T
        if "eeat" in result:
            for category, improvements in result["eeat"].get("suggested_improvements", {}).items():
                for improvement in improvements:
                    suggestions.append({
                        "type": "eeat",
                        "category": category,
                        "description": improvement
                    })
        
        result["suggestions"] = suggestions
        
        # Добавляем общую оценку для LLM
        overall_score = 0
        score_count = 0
        
        if "compatibility" in result:
            compatibility_score = result["compatibility"].get("compatibility_scores", {}).get("overall", 0)
            if compatibility_score:
                overall_score += compatibility_score
                score_count += 1
        
        if "citability" in result:
            citability_score = result["citability"].get("citability_score", 0)
            if citability_score:
                overall_score += citability_score
                score_count += 1
        
        if "eeat" in result:
            eeat_score = result["eeat"].get("eeat_scores", {}).get("overall", 0)
            if eeat_score:
                overall_score += eeat_score
                score_count += 1
        
        if score_count > 0:
            result["overall_score"] = overall_score / score_count
        
        # Добавляем общую стоимость анализа
        total_cost = 0
        
        if "compatibility" in result:
            total_cost += result["compatibility"].get("cost", 0)
        
        if "citability" in result:
            total_cost += result["citability"].get("cost", 0)
        
        if "eeat" in result:
            total_cost += result["eeat"].get("cost", 0)
        
        result["total_cost"] = total_cost
        
        return result
