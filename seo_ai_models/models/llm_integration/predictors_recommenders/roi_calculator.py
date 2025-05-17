"""
Калькулятор ROI от внедрения рекомендаций.

Модуль предоставляет функционал для расчета потенциального
возврата инвестиций от внедрения рекомендаций по оптимизации
контента для традиционных и LLM-поисковиков.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from .llm_rank_predictor import LLMRankPredictor
from ..common.utils import parse_json_response


class ROICalculator:
    """
    Калькулятор ROI от внедрения рекомендаций.
    """
    
    def __init__(self, llm_service: LLMService, prompt_generator: PromptGenerator,
               rank_predictor: LLMRankPredictor):
        """
        Инициализирует калькулятор ROI.
        
        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM
            prompt_generator: Экземпляр PromptGenerator для генерации промптов
            rank_predictor: Экземпляр LLMRankPredictor для предсказания ранжирования
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        self.rank_predictor = rank_predictor
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
        
        # Константы для расчета ROI
        self.default_traffic_increase_rates = {
            "position_1": 1.5,  # 150% увеличение трафика при достижении 1 позиции
            "position_2_3": 1.2,  # 120% увеличение трафика при достижении позиций 2-3
            "position_4_10": 0.8,  # 80% увеличение трафика при достижении позиций 4-10
            "below_10": 0.3  # 30% увеличение трафика при позиции ниже 10
        }
        
        # Стандартные ставки конверсии по отраслям
        self.default_conversion_rates = {
            "ecommerce": 0.03,  # 3% конверсия для E-commerce
            "b2b": 0.01,  # 1% конверсия для B2B
            "finance": 0.04,  # 4% конверсия для финансов
            "health": 0.025,  # 2.5% конверсия для здравоохранения
            "technology": 0.02,  # 2% конверсия для технологий
            "travel": 0.035,  # 3.5% конверсия для путешествий
            "education": 0.02,  # 2% конверсия для образования
            "other": 0.02  # 2% конверсия по умолчанию
        }
        
        # Стандартная ценность конверсии по отраслям (в рублях)
        self.default_conversion_values = {
            "ecommerce": 5000,  # 5000 руб. средний чек для E-commerce
            "b2b": 50000,  # 50000 руб. средняя ценность лида для B2B
            "finance": 15000,  # 15000 руб. средняя ценность для финансов
            "health": 10000,  # 10000 руб. средняя ценность для здравоохранения
            "technology": 20000,  # 20000 руб. средняя ценность для технологий
            "travel": 30000,  # 30000 руб. средняя ценность для путешествий
            "education": 12000,  # 12000 руб. средняя ценность для образования
            "other": 15000  # 15000 руб. средняя ценность по умолчанию
        }
        
        # Стандартная стоимость внедрения рекомендаций по сложности (в рублях)
        self.default_implementation_costs = {
            "1": 5000,  # 5000 руб. для простых рекомендаций
            "2": 10000,  # 10000 руб. для рекомендаций средней сложности
            "3": 20000,  # 20000 руб. для сложных рекомендаций
            "4": 40000,  # 40000 руб. для очень сложных рекомендаций
            "5": 80000   # 80000 руб. для крайне сложных рекомендаций
        }
    
    def calculate_roi(self, current_content: str, recommendations: List[Dict[str, Any]],
                    query: str, industry: Optional[str] = None,
                    current_traffic: Optional[int] = None,
                    conversion_rate: Optional[float] = None,
                    conversion_value: Optional[float] = None,
                    timeframe_months: int = 12) -> Dict[str, Any]:
        """
        Рассчитывает ROI от внедрения рекомендаций.
        
        Args:
            current_content: Текущий контент
            recommendations: Список рекомендаций для внедрения
            query: Поисковый запрос
            industry: Отрасль (опционально)
            current_traffic: Текущий трафик (опционально)
            conversion_rate: Ставка конверсии (опционально)
            conversion_value: Ценность конверсии (опционально)
            timeframe_months: Период расчета ROI в месяцах
            
        Returns:
            Dict[str, Any]: Результаты расчета ROI
        """
        # Определяем значения по умолчанию, если не указаны
        industry = industry or "other"
        conversion_rate = conversion_rate or self.default_conversion_rates.get(industry, 0.02)
        conversion_value = conversion_value or self.default_conversion_values.get(industry, 15000)
        
        # Если текущий трафик не указан, оцениваем его
        if current_traffic is None:
            # Предсказываем ранжирование для текущего контента
            current_ranking = self.rank_predictor.predict_ranking(
                content=current_content,
                query=query,
                industry=industry
            )
            
            # Оцениваем текущий трафик в зависимости от позиции
            position = 10  # Позиция по умолчанию, если нет данных
            if "position_estimate" in current_ranking["ranking_prediction"]:
                position = current_ranking["ranking_prediction"]["position_estimate"]["estimated_position"]
            
            # Оцениваем базовый трафик в зависимости от позиции
            if position <= 1:
                current_traffic = 1000  # Примерный трафик для позиции 1
            elif position <= 3:
                current_traffic = 500   # Примерный трафик для позиций 2-3
            elif position <= 10:
                current_traffic = 200   # Примерный трафик для позиций 4-10
            else:
                current_traffic = 50    # Примерный трафик для позиций ниже 10
        
        # Инициализируем данные для расчета ROI
        implementation_costs = 0
        traffic_increase = 0
        position_improvements = []
        
        # Рассчитываем стоимость внедрения и оцениваем влияние на трафик
        for i, recommendation in enumerate(recommendations):
            self.logger.info(f"Оценка влияния рекомендации {i+1} из {len(recommendations)}")
            
            # Симулируем внедрение рекомендации
            modified_content = self._simulate_recommendation_implementation(
                content=current_content,
                recommendation=recommendation.get("recommendation", ""),
                query=query
            )
            
            # Если не удалось симулировать внедрение, пропускаем рекомендацию
            if not modified_content:
                continue
            
            # Предсказываем влияние на ранжирование
            ranking_impact = self.rank_predictor.predict_impact_of_changes(
                current_content=current_content,
                improved_content=modified_content,
                query=query,
                industry=industry
            )
            
            # Получаем изменение позиции
            position_change = 0
            if "position_change" in ranking_impact:
                position_change = ranking_impact["position_change"].get("position_change", 0)
            
            # Если позиция улучшилась, учитываем в расчете ROI
            if position_change > 0:
                # Оцениваем стоимость внедрения рекомендации
                difficulty = str(recommendation.get("implementation_difficulty", 3))
                implementation_cost = self.default_implementation_costs.get(difficulty, 20000)
                
                # Оцениваем влияние на трафик
                traffic_increase_rate = self._estimate_traffic_increase_rate(position_change)
                
                # Добавляем данные о рекомендации
                position_improvements.append({
                    "recommendation": recommendation.get("recommendation", ""),
                    "implementation_cost": implementation_cost,
                    "position_change": position_change,
                    "traffic_increase_rate": traffic_increase_rate,
                    "estimated_traffic_increase": current_traffic * traffic_increase_rate
                })
                
                # Суммируем стоимость и влияние на трафик
                implementation_costs += implementation_cost
                traffic_increase += current_traffic * traffic_increase_rate
        
        # Расчет ROI
        additional_revenue = traffic_increase * conversion_rate * conversion_value * timeframe_months
        roi = (additional_revenue - implementation_costs) / implementation_costs if implementation_costs > 0 else 0
        roi_percent = roi * 100
        
        # Расчет срока окупаемости (в месяцах)
        monthly_additional_revenue = traffic_increase * conversion_rate * conversion_value
        payback_period = implementation_costs / monthly_additional_revenue if monthly_additional_revenue > 0 else float('inf')
        
        # Формируем итоговый результат
        result = {
            "query": query,
            "industry": industry,
            "current_traffic": current_traffic,
            "potential_traffic_increase": traffic_increase,
            "new_total_traffic": current_traffic + traffic_increase,
            "conversion_rate": conversion_rate,
            "conversion_value": conversion_value,
            "implementation_costs": implementation_costs,
            "additional_revenue": additional_revenue,
            "roi": roi,
            "roi_percent": roi_percent,
            "payback_period_months": payback_period,
            "timeframe_months": timeframe_months,
            "position_improvements": position_improvements,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _simulate_recommendation_implementation(self, content: str, recommendation: str,
                                             query: str) -> Optional[str]:
        """
        Симулирует внедрение рекомендации.
        
        Args:
            content: Исходный контент
            recommendation: Рекомендация для внедрения
            query: Поисковый запрос
            
        Returns:
            Optional[str]: Модифицированный контент или None в случае ошибки
        """
        # Формируем промпт для симуляции внедрения рекомендации
        prompt = f"""
        Ты эксперт по оптимизации контента для поисковых систем и LLM-поисковиков.
        
        Внедри следующую рекомендацию в контент, соблюдая его изначальный стиль и структуру:
        "{recommendation}"
        
        Контент оптимизируется для запроса: "{query}"
        
        Оригинальный контент:
        {content}
        
        Верни ТОЛЬКО модифицированный контент с внедренной рекомендацией, без пояснений и комментариев.
        """
        
        # Выполняем запрос к LLM
        try:
            response = self.llm_service.query(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7
            )
            
            # Извлекаем модифицированный контент
            modified_content = response.get("text", "")
            
            if not modified_content:
                return None
                
            return modified_content
            
        except Exception as e:
            self.logger.error(f"Ошибка при симуляции внедрения рекомендации: {str(e)}")
            return None
    
    def _estimate_traffic_increase_rate(self, position_change: int) -> float:
        """
        Оценивает коэффициент увеличения трафика в зависимости от изменения позиции.
        
        Args:
            position_change: Изменение позиции
            
        Returns:
            float: Коэффициент увеличения трафика
        """
        # Если позиция не улучшилась, нет увеличения трафика
        if position_change <= 0:
            return 0.0
        
        # Если позиция значительно улучшилась, используем более высокий коэффициент
        if position_change >= 10:
            return self.default_traffic_increase_rates["position_1"]
        elif position_change >= 5:
            return self.default_traffic_increase_rates["position_2_3"]
        elif position_change >= 2:
            return self.default_traffic_increase_rates["position_4_10"]
        else:
            return self.default_traffic_increase_rates["below_10"]
    
    def estimate_baseline_traffic(self, query: str, position: int,
                                industry: Optional[str] = None) -> int:
        """
        Оценивает базовый трафик для запроса и позиции.
        
        Args:
            query: Поисковый запрос
            position: Позиция в поисковой выдаче
            industry: Отрасль (опционально)
            
        Returns:
            int: Оценка базового трафика
        """
        # Формируем промпт для оценки трафика
        prompt = f"""
        Ты эксперт по SEO и анализу трафика.
        
        Оцени примерный месячный поисковый трафик для сайта на позиции {position} по запросу:
        "{query}"
        
        {f"Отрасль: {industry}" if industry else ""}
        
        Верни только числовое значение месячного трафика. Не включай в ответ никаких объяснений или дополнительного текста.
        """
        
        # Выполняем запрос к LLM
        try:
            response = self.llm_service.query(
                prompt=prompt,
                max_tokens=50,
                temperature=0.3
            )
            
            # Извлекаем оценку трафика
            traffic_text = response.get("text", "")
            
            # Пытаемся извлечь число
            traffic_match = re.search(r'\d+', traffic_text)
            if traffic_match:
                return int(traffic_match.group(0))
            
            # Если не удалось извлечь число, возвращаем значение по умолчанию
            return self._get_default_traffic(position)
            
        except Exception as e:
            self.logger.error(f"Ошибка при оценке базового трафика: {str(e)}")
            return self._get_default_traffic(position)
    
    def _get_default_traffic(self, position: int) -> int:
        """
        Возвращает значение трафика по умолчанию в зависимости от позиции.
        
        Args:
            position: Позиция в поисковой выдаче
            
        Returns:
            int: Значение трафика по умолчанию
        """
        if position <= 1:
            return 1000  # Примерный трафик для позиции 1
        elif position <= 3:
            return 500   # Примерный трафик для позиций 2-3
        elif position <= 10:
            return 200   # Примерный трафик для позиций 4-10
        else:
            return 50    # Примерный трафик для позиций ниже 10
    
    def calculate_detailed_roi(self, current_content: str, recommendations: List[Dict[str, Any]],
                            query: str, industry: Optional[str] = None,
                            business_data: Optional[Dict[str, Any]] = None,
                            timeframe_months: int = 12) -> Dict[str, Any]:
        """
        Рассчитывает детальный ROI от внедрения рекомендаций, включая раздельные
        показатели для традиционных и LLM-поисковиков.
        
        Args:
            current_content: Текущий контент
            recommendations: Список рекомендаций для внедрения
            query: Поисковый запрос
            industry: Отрасль (опционально)
            business_data: Бизнес-данные для расчета (опционально)
            timeframe_months: Период расчета ROI в месяцах
            
        Returns:
            Dict[str, Any]: Результаты детального расчета ROI
        """
        # Определяем отрасль
        industry = industry or "other"
        
        # Определяем бизнес-данные
        if business_data is None:
            business_data = {
                "traditional_traffic": 500,  # Трафик с традиционных поисковиков
                "llm_traffic": 200,  # Трафик с LLM-поисковиков
                "traditional_conversion_rate": self.default_conversion_rates.get(industry, 0.02),
                "llm_conversion_rate": self.default_conversion_rates.get(industry, 0.02) * 1.2,  # Обычно выше на 20%
                "traditional_conversion_value": self.default_conversion_values.get(industry, 15000),
                "llm_conversion_value": self.default_conversion_values.get(industry, 15000) * 1.1  # Обычно выше на 10%
            }
        
        # Инициализируем данные для расчета ROI
        implementation_costs = 0
        traditional_traffic_increase = 0
        llm_traffic_increase = 0
        recommendation_details = []
        
        # Рассчитываем стоимость внедрения и оцениваем влияние на трафик
        for i, recommendation in enumerate(recommendations):
            self.logger.info(f"Оценка влияния рекомендации {i+1} из {len(recommendations)}")
            
            # Определяем тип рекомендации
            rec_type = recommendation.get("type", "traditional_seo")
            
            # Симулируем внедрение рекомендации
            modified_content = self._simulate_recommendation_implementation(
                content=current_content,
                recommendation=recommendation.get("recommendation", ""),
                query=query
            )
            
            # Если не удалось симулировать внедрение, пропускаем рекомендацию
            if not modified_content:
                continue
            
            # Предсказываем влияние на ранжирование
            ranking_impact = self.rank_predictor.predict_impact_of_changes(
                current_content=current_content,
                improved_content=modified_content,
                query=query,
                industry=industry
            )
            
            # Получаем изменение позиции
            position_change = 0
            if "position_change" in ranking_impact:
                position_change = ranking_impact["position_change"].get("position_change", 0)
            
            # Оцениваем стоимость внедрения рекомендации
            difficulty = str(recommendation.get("implementation_difficulty", 3))
            implementation_cost = self.default_implementation_costs.get(difficulty, 20000)
            
            # Оцениваем влияние на трафик для разных типов поисковиков
            traditional_increase_rate = self._estimate_traffic_increase_rate(position_change) * 0.8
            llm_increase_rate = self._estimate_traffic_increase_rate(position_change)
            
            # Корректируем коэффициенты в зависимости от типа рекомендации
            if rec_type == "traditional_seo":
                traditional_increase_rate *= 1.5
                llm_increase_rate *= 0.5
            elif rec_type == "llm_optimization":
                traditional_increase_rate *= 0.5
                llm_increase_rate *= 1.5
            
            # Рассчитываем увеличение трафика
            traditional_increase = business_data.get("traditional_traffic", 0) * traditional_increase_rate
            llm_increase = business_data.get("llm_traffic", 0) * llm_increase_rate
            
            # Добавляем данные о рекомендации
            recommendation_details.append({
                "recommendation": recommendation.get("recommendation", ""),
                "type": rec_type,
                "implementation_cost": implementation_cost,
                "position_change": position_change,
                "traditional_traffic_increase": traditional_increase,
                "llm_traffic_increase": llm_increase,
                "total_traffic_increase": traditional_increase + llm_increase
            })
            
            # Суммируем стоимость и влияние на трафик
            implementation_costs += implementation_cost
            traditional_traffic_increase += traditional_increase
            llm_traffic_increase += llm_increase
        
        # Расчет ROI для традиционных поисковиков
        traditional_additional_revenue = (
            traditional_traffic_increase * 
            business_data.get("traditional_conversion_rate", 0.02) * 
            business_data.get("traditional_conversion_value", 15000) * 
            timeframe_months
        )
        
        # Расчет ROI для LLM-поисковиков
        llm_additional_revenue = (
            llm_traffic_increase * 
            business_data.get("llm_conversion_rate", 0.024) * 
            business_data.get("llm_conversion_value", 16500) * 
            timeframe_months
        )
        
        # Общая дополнительная выручка
        total_additional_revenue = traditional_additional_revenue + llm_additional_revenue
        
        # Общий ROI
        roi = (total_additional_revenue - implementation_costs) / implementation_costs if implementation_costs > 0 else 0
        roi_percent = roi * 100
        
        # Расчет срока окупаемости (в месяцах)
        monthly_additional_revenue = total_additional_revenue / timeframe_months
        payback_period = implementation_costs / monthly_additional_revenue if monthly_additional_revenue > 0 else float('inf')
        
        # Формируем итоговый результат
        result = {
            "query": query,
            "industry": industry,
            "traditional": {
                "current_traffic": business_data.get("traditional_traffic", 0),
                "traffic_increase": traditional_traffic_increase,
                "new_total_traffic": business_data.get("traditional_traffic", 0) + traditional_traffic_increase,
                "conversion_rate": business_data.get("traditional_conversion_rate", 0.02),
                "conversion_value": business_data.get("traditional_conversion_value", 15000),
                "additional_revenue": traditional_additional_revenue
            },
            "llm": {
                "current_traffic": business_data.get("llm_traffic", 0),
                "traffic_increase": llm_traffic_increase,
                "new_total_traffic": business_data.get("llm_traffic", 0) + llm_traffic_increase,
                "conversion_rate": business_data.get("llm_conversion_rate", 0.024),
                "conversion_value": business_data.get("llm_conversion_value", 16500),
                "additional_revenue": llm_additional_revenue
            },
            "total": {
                "current_traffic": business_data.get("traditional_traffic", 0) + business_data.get("llm_traffic", 0),
                "traffic_increase": traditional_traffic_increase + llm_traffic_increase,
                "new_total_traffic": (business_data.get("traditional_traffic", 0) + business_data.get("llm_traffic", 0) +
                                    traditional_traffic_increase + llm_traffic_increase),
                "additional_revenue": total_additional_revenue
            },
            "implementation_costs": implementation_costs,
            "roi": roi,
            "roi_percent": roi_percent,
            "payback_period_months": payback_period,
            "timeframe_months": timeframe_months,
            "recommendation_details": recommendation_details,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
