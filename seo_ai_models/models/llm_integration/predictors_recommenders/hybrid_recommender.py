"""
Рекомендации, оптимизированные для обоих типов поиска.

Модуль предоставляет функционал для генерации рекомендаций,
оптимизированных как для традиционных поисковых систем,
так и для LLM-поисковиков.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from ..analyzers.citability_scorer import CitabilityScorer
from ..analyzers.llm_eeat_analyzer import LLMEEATAnalyzer
from ..serp_analysis.llm_serp_analyzer import LLMSerpAnalyzer
from .llm_rank_predictor import LLMRankPredictor
from ...seo_advisor.enhanced_advisor import EnhancedSEOAdvisor
from ..common.utils import parse_json_response


class HybridRecommender:
    """
    Рекомендации, оптимизированные для обоих типов поиска.
    """
    
    def __init__(self, llm_service: LLMService, prompt_generator: PromptGenerator,
               enhanced_seo_advisor: Optional[EnhancedSEOAdvisor] = None):
        """
        Инициализирует систему рекомендаций, оптимизированных для обоих типов поиска.
        
        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM
            prompt_generator: Экземпляр PromptGenerator для генерации промптов
            enhanced_seo_advisor: Экземпляр EnhancedSEOAdvisor для традиционных рекомендаций (опционально)
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        self.enhanced_seo_advisor = enhanced_seo_advisor
        self.citability_scorer = CitabilityScorer(llm_service, prompt_generator)
        self.eeat_analyzer = LLMEEATAnalyzer(llm_service, prompt_generator)
        self.serp_analyzer = LLMSerpAnalyzer(llm_service, prompt_generator)
        self.rank_predictor = LLMRankPredictor(llm_service, prompt_generator)
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
        
        # Типы рекомендаций
        self.recommendation_types = {
            "traditional_seo": {
                "weight": 0.5,  # Вес рекомендаций для традиционных поисковиков
                "categories": [
                    "on_page", "technical", "links", "content", "meta_tags", "user_experience"
                ]
            },
            "llm_optimization": {
                "weight": 0.5,  # Вес рекомендаций для LLM-поисковиков
                "categories": [
                    "citability", "eeat", "content_structure", "information_quality", "authority"
                ]
            }
        }
    
    def generate_recommendations(self, content: str, query: str,
                               industry: Optional[str] = None,
                               balance_mode: str = "balanced",
                               max_recommendations: int = 10) -> Dict[str, Any]:
        """
        Генерирует рекомендации, оптимизированные для обоих типов поиска.
        
        Args:
            content: Контент для анализа
            query: Ключевой запрос
            industry: Отрасль (опционально)
            balance_mode: Режим баланса рекомендаций ("traditional", "llm", "balanced")
            max_recommendations: Максимальное количество рекомендаций
            
        Returns:
            Dict[str, Any]: Рекомендации для оптимизации
        """
        # Логируем начало генерации рекомендаций
        self.logger.info(f"Начало генерации рекомендаций для запроса: {query}")
        
        # Настраиваем баланс рекомендаций
        balance_weights = self._get_balance_weights(balance_mode)
        
        # Получаем рекомендации для традиционных поисковиков
        traditional_recommendations = self._get_traditional_recommendations(
            content, query, industry
        )
        
        # Получаем рекомендации для LLM-поисковиков
        llm_recommendations = self._get_llm_recommendations(
            content, query, industry
        )
        
        # Анализируем потенциальные конфликты рекомендаций
        conflicts = self._analyze_recommendation_conflicts(
            traditional_recommendations, llm_recommendations
        )
        
        # Объединяем и приоритизируем рекомендации
        combined_recommendations = self._combine_and_prioritize_recommendations(
            traditional_recommendations, llm_recommendations, 
            conflicts, balance_weights, max_recommendations
        )
        
        # Формируем итоговый результат
        result = {
            "query": query,
            "industry": industry,
            "balance_mode": balance_mode,
            "balance_weights": balance_weights,
            "recommendations": combined_recommendations,
            "recommendation_summary": self._generate_recommendation_summary(combined_recommendations),
            "conflicts": conflicts,
            "traditional_recommendations_count": len(traditional_recommendations),
            "llm_recommendations_count": len(llm_recommendations),
            "combined_recommendations_count": len(combined_recommendations),
            "timestamp": datetime.now().isoformat(),
            "tokens": {
                "traditional": sum(rec.get("tokens", {}).get("total", 0) for rec in traditional_recommendations),
                "llm": sum(rec.get("tokens", {}).get("total", 0) for rec in llm_recommendations)
            },
            "cost": (sum(rec.get("cost", 0) for rec in traditional_recommendations) + 
                    sum(rec.get("cost", 0) for rec in llm_recommendations))
        }
        
        return result
    
    def _get_balance_weights(self, balance_mode: str) -> Dict[str, float]:
        """
        Возвращает веса баланса рекомендаций.
        
        Args:
            balance_mode: Режим баланса рекомендаций ("traditional", "llm", "balanced")
            
        Returns:
            Dict[str, float]: Веса баланса рекомендаций
        """
        if balance_mode == "traditional":
            return {"traditional_seo": 0.8, "llm_optimization": 0.2}
        elif balance_mode == "llm":
            return {"traditional_seo": 0.2, "llm_optimization": 0.8}
        else:  # balanced
            return {"traditional_seo": 0.5, "llm_optimization": 0.5}
    
    def _get_traditional_recommendations(self, content: str, query: str,
                                       industry: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получает рекомендации для традиционных поисковиков.
        
        Args:
            content: Контент для анализа
            query: Ключевой запрос
            industry: Отрасль (опционально)
            
        Returns:
            List[Dict[str, Any]]: Рекомендации для традиционных поисковиков
        """
        # Если доступен EnhancedSEOAdvisor, используем его
        if self.enhanced_seo_advisor:
            # Анализируем контент и получаем рекомендации
            seo_analysis = self.enhanced_seo_advisor.analyze_content(
                content=content,
                target_keywords=[query],
                industry=industry
            )
            
            # Извлекаем рекомендации из результатов анализа
            traditional_recommendations = []
            
            for suggestion in seo_analysis.get("suggestions", []):
                traditional_recommendations.append({
                    "type": "traditional_seo",
                    "category": suggestion.get("category", "content"),
                    "recommendation": suggestion.get("description", ""),
                    "priority": suggestion.get("priority", 0),
                    "impact": suggestion.get("impact", 0),
                    "implementation_difficulty": suggestion.get("implementation_difficulty", 0),
                    "tokens": {"total": 0},
                    "cost": 0
                })
            
            return traditional_recommendations
        
        # Если EnhancedSEOAdvisor не доступен, используем LLM для генерации рекомендаций
        prompt = f"""
        Ты эксперт по SEO-оптимизации для традиционных поисковых систем (Google, Bing).
        
        Проанализируй следующий контент и предложи рекомендации по его оптимизации
        для традиционных поисковых систем для запроса: "{query}".
        
        Предложи рекомендации по следующим категориям:
        - on_page (оптимизация страницы)
        - technical (техническая оптимизация)
        - links (внутренние и внешние ссылки)
        - content (улучшение контента)
        - meta_tags (мета-теги)
        - user_experience (пользовательский опыт)
        
        Для каждой рекомендации укажи:
        - категорию
        - описание рекомендации
        - приоритет (от 1 до 5, где 5 - самый высокий)
        - ожидаемое влияние на ранжирование (от 1 до 5, где 5 - наибольшее влияние)
        - сложность внедрения (от 1 до 5, где 5 - самая сложная)
        
        Представь результат в формате JSON, например:
        [
            {
                "category": "content",
                "recommendation": "Добавить больше ключевых слов в заголовок",
                "priority": 4,
                "impact": 3,
                "implementation_difficulty": 1
            },
            ...
        ]
        
        Контент для анализа:
        {content}
        """
        
        # Выполняем запрос к LLM
        response = self.llm_service.query(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.5
        )
        
        # Извлекаем рекомендации из ответа
        recommendations_text = response.get("text", "")
        recommendations_data = parse_json_response(recommendations_text)
        
        # Если не удалось извлечь JSON, пытаемся структурировать ответ сами
        if not recommendations_data:
            recommendations_data = []
            
            # Пытаемся найти рекомендации в тексте
            recommendation_matches = re.findall(r'(\d+\.\s+)?(.*?):\s+(.*?)(?:\n|$)', recommendations_text)
            
            for match in recommendation_matches:
                recommendations_data.append({
                    "category": "content",  # Категория по умолчанию
                    "recommendation": match[2],
                    "priority": 3,  # Приоритет по умолчанию
                    "impact": 3,  # Влияние по умолчанию
                    "implementation_difficulty": 2  # Сложность по умолчанию
                })
        
        # Преобразуем данные в нужный формат
        traditional_recommendations = []
        
        for rec in recommendations_data:
            traditional_recommendations.append({
                "type": "traditional_seo",
                "category": rec.get("category", "content"),
                "recommendation": rec.get("recommendation", ""),
                "priority": rec.get("priority", 3),
                "impact": rec.get("impact", 3),
                "implementation_difficulty": rec.get("implementation_difficulty", 2),
                "tokens": response.get("tokens", {"total": 0}),
                "cost": response.get("cost", 0) / len(recommendations_data) if recommendations_data else 0
            })
        
        return traditional_recommendations
    
    def _get_llm_recommendations(self, content: str, query: str,
                               industry: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получает рекомендации для LLM-поисковиков.
        
        Args:
            content: Контент для анализа
            query: Ключевой запрос
            industry: Отрасль (опционально)
            
        Returns:
            List[Dict[str, Any]]: Рекомендации для LLM-поисковиков
        """
        # Получаем оценку цитируемости
        citability_result = self.citability_scorer.score_citability(
            content=content,
            queries=[query]
        )
        
        # Получаем оценку E-E-A-T
        eeat_result = self.eeat_analyzer.analyze_llm_eeat(
            content=content,
            industry=industry or "general"
        )
        
        # Инициализируем список рекомендаций
        llm_recommendations = []
        
        # Добавляем рекомендации по цитируемости
        for factor, improvements in citability_result.get("suggested_improvements", {}).items():
            for improvement in improvements:
                llm_recommendations.append({
                    "type": "llm_optimization",
                    "category": "citability",
                    "recommendation": improvement,
                    "source_factor": factor,
                    "priority": 4,  # Приоритет по умолчанию для цитируемости
                    "impact": 4,  # Влияние по умолчанию для цитируемости
                    "implementation_difficulty": 2,  # Сложность по умолчанию
                    "tokens": citability_result.get("tokens", {"total": 0}),
                    "cost": citability_result.get("cost", 0) / (len(improvements) or 1)
                })
        
        # Добавляем рекомендации по E-E-A-T
        for component, improvements in eeat_result.get("suggested_improvements", {}).items():
            for improvement in improvements:
                llm_recommendations.append({
                    "type": "llm_optimization",
                    "category": "eeat",
                    "recommendation": improvement,
                    "source_factor": component,
                    "priority": 5,  # Приоритет по умолчанию для E-E-A-T
                    "impact": 5,  # Влияние по умолчанию для E-E-A-T
                    "implementation_difficulty": 3,  # Сложность по умолчанию
                    "tokens": eeat_result.get("tokens", {"total": 0}),
                    "cost": eeat_result.get("cost", 0) / (len(improvements) or 1)
                })
        
        # Если список рекомендаций пуст или нужны дополнительные рекомендации,
        # генерируем их с помощью LLM
        if len(llm_recommendations) < 5:
            # Формируем промпт для генерации рекомендаций
            prompt = f"""
            Ты эксперт по оптимизации контента для LLM-поисковиков (языковых моделей).
            
            Проанализируй следующий контент и предложи рекомендации по его оптимизации
            для LLM-поисковиков для запроса: "{query}".
            
            Текущие показатели:
            - Цитируемость: {citability_result.get('citability_score', 0)}/10
            - E-E-A-T: {eeat_result.get('eeat_score', 0)}/10
            
            Предложи рекомендации по следующим категориям:
            - citability (цитируемость)
            - eeat (опыт, экспертиза, авторитетность, достоверность)
            - content_structure (структура контента)
            - information_quality (качество информации)
            - authority (авторитетность)
            
            Для каждой рекомендации укажи:
            - категорию
            - описание рекомендации
            - приоритет (от 1 до 5, где 5 - самый высокий)
            - ожидаемое влияние на ранжирование (от 1 до 5, где 5 - наибольшее влияние)
            - сложность внедрения (от 1 до 5, где 5 - самая сложная)
            
            Представь результат в формате JSON, например:
            [
                {{
                    "category": "citability",
                    "recommendation": "Добавить больше фактических данных и ссылки на источники",
                    "priority": 5,
                    "impact": 4,
                    "implementation_difficulty": 3
                }},
                ...
            ]
            
            Контент для анализа:
            {content}
            """
            
            # Выполняем запрос к LLM
            response = self.llm_service.query(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.5
            )
            
            # Извлекаем рекомендации из ответа
            recommendations_text = response.get("text", "")
            recommendations_data = parse_json_response(recommendations_text)
            
            # Если не удалось извлечь JSON, пытаемся структурировать ответ сами
            if not recommendations_data:
                recommendations_data = []
                
                # Пытаемся найти рекомендации в тексте
                recommendation_matches = re.findall(r'(\d+\.\s+)?(.*?):\s+(.*?)(?:\n|$)', recommendations_text)
                
                for match in recommendation_matches:
                    recommendations_data.append({
                        "category": "citability",  # Категория по умолчанию
                        "recommendation": match[2],
                        "priority": 4,  # Приоритет по умолчанию
                        "impact": 4,  # Влияние по умолчанию
                        "implementation_difficulty": 3  # Сложность по умолчанию
                    })
            
            # Добавляем сгенерированные рекомендации
            for rec in recommendations_data:
                llm_recommendations.append({
                    "type": "llm_optimization",
                    "category": rec.get("category", "citability"),
                    "recommendation": rec.get("recommendation", ""),
                    "priority": rec.get("priority", 4),
                    "impact": rec.get("impact", 4),
                    "implementation_difficulty": rec.get("implementation_difficulty", 3),
                    "tokens": response.get("tokens", {"total": 0}),
                    "cost": response.get("cost", 0) / len(recommendations_data) if recommendations_data else 0
                })
        
        return llm_recommendations
    
    def _analyze_recommendation_conflicts(self, traditional_recommendations: List[Dict[str, Any]],
                                        llm_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Анализирует потенциальные конфликты между рекомендациями.
        
        Args:
            traditional_recommendations: Рекомендации для традиционных поисковиков
            llm_recommendations: Рекомендации для LLM-поисковиков
            
        Returns:
            List[Dict[str, Any]]: Список выявленных конфликтов
        """
        # В реальном проекте здесь был бы более сложный алгоритм
        # Для примера используем простой алгоритм на основе ключевых слов
        
        # Ключевые слова, которые могут указывать на конфликты
        conflict_patterns = [
            ("keyword stuffing", "информативность"),
            ("тег h1", "структура заголовков"),
            ("ключевые слова", "естественность"),
            ("сокращение", "подробное объяснение"),
            ("краткость", "полнота"),
            ("технические детали", "понятность")
        ]
        
        # Инициализируем список конфликтов
        conflicts = []
        
        # Анализируем рекомендации на предмет конфликтов
        for trad_rec in traditional_recommendations:
            trad_text = trad_rec.get("recommendation", "").lower()
            
            for llm_rec in llm_recommendations:
                llm_text = llm_rec.get("recommendation", "").lower()
                
                # Проверяем наличие конфликтных паттернов
                for trad_pattern, llm_pattern in conflict_patterns:
                    if trad_pattern.lower() in trad_text and llm_pattern.lower() in llm_text:
                        # Найден потенциальный конфликт
                        conflicts.append({
                            "conflict_type": f"{trad_pattern} vs {llm_pattern}",
                            "traditional_recommendation": trad_rec.get("recommendation", ""),
                            "llm_recommendation": llm_rec.get("recommendation", ""),
                            "resolution_suggestion": self._generate_conflict_resolution(
                                trad_rec.get("recommendation", ""),
                                llm_rec.get("recommendation", "")
                            )
                        })
        
        return conflicts
    
    def _generate_conflict_resolution(self, trad_recommendation: str, 
                                    llm_recommendation: str) -> str:
        """
        Генерирует предложение по разрешению конфликта между рекомендациями.
        
        Args:
            trad_recommendation: Рекомендация для традиционных поисковиков
            llm_recommendation: Рекомендация для LLM-поисковиков
            
        Returns:
            str: Предложение по разрешению конфликта
        """
        # В реальном проекте здесь был бы более сложный алгоритм,
        # возможно с использованием LLM для генерации компромиссного решения
        
        # Для примера используем шаблон
        resolution = f"Найдите баланс между '{trad_recommendation}' и '{llm_recommendation}', "
        resolution += "сфокусировавшись на качестве контента, его структурировании и информативности, "
        resolution += "что будет полезно для обоих типов поисковых систем."
        
        return resolution
    
    def _combine_and_prioritize_recommendations(self, traditional_recommendations: List[Dict[str, Any]],
                                             llm_recommendations: List[Dict[str, Any]],
                                             conflicts: List[Dict[str, Any]],
                                             balance_weights: Dict[str, float],
                                             max_recommendations: int) -> List[Dict[str, Any]]:
        """
        Объединяет и приоритизирует рекомендации.
        
        Args:
            traditional_recommendations: Рекомендации для традиционных поисковиков
            llm_recommendations: Рекомендации для LLM-поисковиков
            conflicts: Список выявленных конфликтов
            balance_weights: Веса баланса рекомендаций
            max_recommendations: Максимальное количество рекомендаций
            
        Returns:
            List[Dict[str, Any]]: Объединенные и приоритизированные рекомендации
        """
        # Объединяем все рекомендации в один список
        all_recommendations = []
        
        # Добавляем рекомендации для традиционных поисковиков
        for rec in traditional_recommendations:
            # Рассчитываем взвешенный приоритет
            weighted_priority = rec.get("priority", 0) * rec.get("impact", 0) * balance_weights["traditional_seo"]
            
            # Добавляем рекомендацию с взвешенным приоритетом
            all_recommendations.append({
                **rec,
                "weighted_priority": weighted_priority
            })
        
        # Добавляем рекомендации для LLM-поисковиков
        for rec in llm_recommendations:
            # Рассчитываем взвешенный приоритет
            weighted_priority = rec.get("priority", 0) * rec.get("impact", 0) * balance_weights["llm_optimization"]
            
            # Добавляем рекомендацию с взвешенным приоритетом
            all_recommendations.append({
                **rec,
                "weighted_priority": weighted_priority
            })
        
        # Фильтруем конфликтующие рекомендации
        filtered_recommendations = []
        conflict_resolutions = []
        
        # Собираем конфликтующие рекомендации
        conflict_texts = []
        for conflict in conflicts:
            conflict_texts.append(conflict.get("traditional_recommendation", ""))
            conflict_texts.append(conflict.get("llm_recommendation", ""))
            
            # Добавляем резолюцию конфликта как отдельную рекомендацию
            conflict_resolutions.append({
                "type": "hybrid",
                "category": "conflict_resolution",
                "recommendation": conflict.get("resolution_suggestion", ""),
                "priority": 5,  # Высокий приоритет для резолюций конфликтов
                "impact": 5,  # Высокое влияние для резолюций конфликтов
                "implementation_difficulty": 3,  # Средняя сложность
                "weighted_priority": 5 * 5 * (balance_weights["traditional_seo"] + balance_weights["llm_optimization"]) / 2,
                "tokens": {"total": 0},
                "cost": 0
            })
        
        # Добавляем неконфликтующие рекомендации
        for rec in all_recommendations:
            if rec.get("recommendation", "") not in conflict_texts:
                filtered_recommendations.append(rec)
        
        # Добавляем резолюции конфликтов
        filtered_recommendations.extend(conflict_resolutions)
        
        # Сортируем рекомендации по взвешенному приоритету
        filtered_recommendations.sort(key=lambda x: x.get("weighted_priority", 0), reverse=True)
        
        # Ограничиваем количество рекомендаций
        limited_recommendations = filtered_recommendations[:max_recommendations]
        
        # Удаляем служебное поле weighted_priority из итоговых рекомендаций
        for rec in limited_recommendations:
            if "weighted_priority" in rec:
                del rec["weighted_priority"]
        
        return limited_recommendations
    
    def _generate_recommendation_summary(self, recommendations: List[Dict[str, Any]]) -> str:
        """
        Генерирует краткое резюме рекомендаций.
        
        Args:
            recommendations: Список рекомендаций
            
        Returns:
            str: Краткое резюме рекомендаций
        """
        # Считаем количество рекомендаций по типам
        traditional_count = sum(1 for rec in recommendations if rec.get("type") == "traditional_seo")
        llm_count = sum(1 for rec in recommendations if rec.get("type") == "llm_optimization")
        hybrid_count = sum(1 for rec in recommendations if rec.get("type") == "hybrid")
        
        # Определяем наиболее приоритетные рекомендации
        high_priority_recs = [rec for rec in recommendations if rec.get("priority", 0) >= 4]
        
        # Генерируем резюме
        summary = f"Сгенерировано {len(recommendations)} рекомендаций: "
        summary += f"{traditional_count} для традиционных поисковиков, "
        summary += f"{llm_count} для LLM-поисковиков, "
        summary += f"{hybrid_count} гибридных. "
        
        if high_priority_recs:
            summary += f"Приоритетные задачи: "
            summary += ", ".join([rec.get("recommendation", "")[:50] + "..." for rec in high_priority_recs[:3]])
        
        return summary
    
    def simulate_recommendation_impact(self, content: str, query: str,
                                     recommendation: str,
                                     industry: Optional[str] = None) -> Dict[str, Any]:
        """
        Симулирует влияние рекомендации на контент.
        
        Args:
            content: Исходный контент
            query: Поисковый запрос
            recommendation: Рекомендация для внедрения
            industry: Отрасль (опционально)
            
        Returns:
            Dict[str, Any]: Результаты симуляции влияния рекомендации
        """
        # Формируем промпт для симуляции внедрения рекомендации
        prompt = f"""
        Ты эксперт по оптимизации контента для поисковых систем и LLM-поисковиков.
        
        Внедри следующую рекомендацию в контент, соблюдая его изначальный стиль и структуру:
        "{recommendation}"
        
        Контент оптимизируется для запроса: "{query}"
        
        Оригинальный контент:
        {content}
        
        Верни модифицированный контент с внедренной рекомендацией, а также краткое описание внесенных изменений.
        Используй формат JSON:
        {{
            "modified_content": "Модифицированный контент...",
            "changes_description": "Описание внесенных изменений..."
        }}
        """
        
        # Выполняем запрос к LLM
        response = self.llm_service.query(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.7
        )
        
        # Извлекаем результаты из ответа
        simulation_text = response.get("text", "")
        simulation_data = parse_json_response(simulation_text)
        
        # Если не удалось извлечь JSON, пытаемся структурировать ответ сами
        if not simulation_data:
            # Пытаемся найти модифицированный контент в ответе
            content_match = re.search(r'Модифицированный контент:(.*?)(?:Описание изменений:|$)', 
                                    simulation_text, re.DOTALL)
            changes_match = re.search(r'Описание (внесенных )?изменений:(.*?)$', 
                                    simulation_text, re.DOTALL)
            
            modified_content = content_match.group(1).strip() if content_match else content
            changes_description = changes_match.group(2).strip() if changes_match else "Нет описания изменений."
            
            simulation_data = {
                "modified_content": modified_content,
                "changes_description": changes_description
            }
        
        # Получаем модифицированный контент
        modified_content = simulation_data.get("modified_content", content)
        
        # Предсказываем влияние изменений на ранжирование
        ranking_impact = self.rank_predictor.predict_impact_of_changes(
            current_content=content,
            improved_content=modified_content,
            query=query,
            industry=industry
        )
        
        # Формируем результат симуляции
        result = {
            "query": query,
            "recommendation": recommendation,
            "modified_content": modified_content,
            "changes_description": simulation_data.get("changes_description", ""),
            "ranking_impact": {
                "score_change": ranking_impact.get("score_change", 0),
                "score_percent_change": ranking_impact.get("score_percent_change", 0),
                "position_change": ranking_impact.get("position_change", {})
            },
            "tokens": {
                "simulation": response.get("tokens", {}).get("total", 0),
                "ranking": ranking_impact.get("tokens", {}).get("total", 0),
                "total": (response.get("tokens", {}).get("total", 0) + 
                         ranking_impact.get("tokens", {}).get("total", 0))
            },
            "cost": (response.get("cost", 0) + ranking_impact.get("cost", 0))
        }
        
        return result
