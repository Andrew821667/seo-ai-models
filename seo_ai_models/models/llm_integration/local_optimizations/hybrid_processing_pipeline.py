"""
Гибридный пайплайн обработки LLM-запросов.

Модуль предоставляет функционал для комбинирования облачных и локальных
LLM-моделей в рамках одного пайплайна обработки, оптимизируя
затраты и производительность.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from ..service.cost_estimator import CostEstimator
from .local_llm_manager import LocalLLMManager

class HybridProcessingPipeline:
    """
    Гибридный пайплайн обработки LLM-запросов.
    """
    
    def __init__(self, llm_service: LLMService, 
               local_llm_manager: LocalLLMManager,
               cost_estimator: Optional[CostEstimator] = None):
        """
        Инициализирует гибридный пайплайн обработки.
        
        Args:
            llm_service: Экземпляр LLMService для взаимодействия с облачными LLM
            local_llm_manager: Экземпляр LocalLLMManager для работы с локальными LLM
            cost_estimator: Экземпляр CostEstimator для оценки затрат (опционально)
        """
        self.llm_service = llm_service
        self.local_llm_manager = local_llm_manager
        self.cost_estimator = cost_estimator or CostEstimator()
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
        
        # Конфигурация стратегии распределения запросов
        self.fallback_mode = True  # Использовать облако при ошибках с локальными моделями
        self.quality_threshold = 0.7  # Порог качества для выбора облачной модели
        self.cost_optimization_level = "medium"  # Уровень оптимизации затрат (low, medium, high)
        
        # Стратегии обработки запросов
        self.processing_strategies = {
            "auto": self._strategy_auto,
            "cloud_first": self._strategy_cloud_first,
            "local_first": self._strategy_local_first,
            "cost_optimized": self._strategy_cost_optimized,
            "quality_optimized": self._strategy_quality_optimized,
            "balanced": self._strategy_balanced
        }
    
    def process(self, prompt: str, strategy: str = "auto",
              max_tokens: int = 500, temperature: float = 0.7,
              budget: Optional[float] = None, required_quality: str = "medium",
              cloud_model: Optional[str] = None, local_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Обрабатывает запрос, используя выбранную стратегию распределения.
        
        Args:
            prompt: Промпт для обработки
            strategy: Стратегия распределения запросов
                      ('auto', 'cloud_first', 'local_first', 'cost_optimized', 
                       'quality_optimized', 'balanced')
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации (от 0 до 1)
            budget: Максимальный бюджет в рублях (опционально)
            required_quality: Требуемое качество ответа ('low', 'medium', 'high')
            cloud_model: Облачная модель для использования (опционально)
            local_model: Локальная модель для использования (опционально)
            
        Returns:
            Dict[str, Any]: Результат обработки запроса
        """
        # Проверяем, что стратегия поддерживается
        if strategy not in self.processing_strategies:
            self.logger.warning(f"Стратегия {strategy} не поддерживается, используем 'auto'")
            strategy = "auto"
        
        # Применяем выбранную стратегию
        strategy_func = self.processing_strategies[strategy]
        
        # Засекаем время обработки
        start_time = time.time()
        
        # Обрабатываем запрос с помощью выбранной стратегии
        result = strategy_func(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            budget=budget,
            required_quality=required_quality,
            cloud_model=cloud_model,
            local_model=local_model
        )
        
        # Добавляем информацию о времени обработки
        result["processing_time"] = time.time() - start_time
        result["processing_strategy"] = strategy
        
        return result
    
    def _strategy_auto(self, prompt: str, max_tokens: int, temperature: float,
                     budget: Optional[float], required_quality: str,
                     cloud_model: Optional[str], local_model: Optional[str]) -> Dict[str, Any]:
        """
        Автоматическая стратегия выбора между облачной и локальной обработкой.
        
        Выбирает наиболее подходящую стратегию в зависимости от сложности запроса,
        доступного бюджета и требуемого качества.
        """
        # Оцениваем сложность запроса
        prompt_complexity = self._estimate_prompt_complexity(prompt)
        
        # Оцениваем затраты на облачную обработку
        cloud_cost = self._estimate_cloud_cost(prompt, max_tokens, cloud_model)
        
        # Выбираем стратегию в зависимости от параметров
        if required_quality == "high" or prompt_complexity == "high":
            # Для высокого качества или сложных запросов предпочитаем облако
            return self._strategy_cloud_first(
                prompt, max_tokens, temperature, budget, required_quality, cloud_model, local_model
            )
        elif budget is not None and budget < cloud_cost:
            # Если бюджет ограничен, выбираем стратегию оптимизации затрат
            return self._strategy_cost_optimized(
                prompt, max_tokens, temperature, budget, required_quality, cloud_model, local_model
            )
        else:
            # В остальных случаях используем сбалансированную стратегию
            return self._strategy_balanced(
                prompt, max_tokens, temperature, budget, required_quality, cloud_model, local_model
            )
    
    def _strategy_cloud_first(self, prompt: str, max_tokens: int, temperature: float,
                           budget: Optional[float], required_quality: str,
                           cloud_model: Optional[str], local_model: Optional[str]) -> Dict[str, Any]:
        """
        Стратегия с приоритетом облачной обработки.
        
        Сначала пытается использовать облачную модель, при ошибке или превышении
        бюджета использует локальную.
        """
        # Оцениваем затраты на облачную обработку
        cloud_cost = self._estimate_cloud_cost(prompt, max_tokens, cloud_model)
        
        # Проверяем, не превышает ли стоимость бюджет
        if budget is not None and cloud_cost > budget:
            self.logger.warning(f"Стоимость облачной обработки ({cloud_cost} руб.) превышает бюджет ({budget} руб.), используем локальную модель")
            return self._use_local_model(
                prompt, max_tokens, temperature, budget, required_quality, local_model
            )
        
        # Пытаемся использовать облачную модель
        try:
            # Запрос к облачной модели
            result = self.llm_service.query(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                model=cloud_model,
                budget=budget
            )
            
            # Добавляем информацию о стратегии
            result["processing_info"] = {
                "method": "cloud",
                "model": result.get("model", cloud_model),
                "fallback_used": False
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка при облачной обработке: {str(e)}")
            
            # Если включен режим fallback, используем локальную модель
            if self.fallback_mode:
                self.logger.info("Используем локальную модель в качестве fallback")
                result = self._use_local_model(
                    prompt, max_tokens, temperature, budget, required_quality, local_model
                )
                result["processing_info"]["fallback_used"] = True
                return result
            else:
                # Если fallback отключен, возвращаем ошибку
                return {
                    "error": f"Ошибка при облачной обработке: {str(e)}",
                    "text": "",
                    "provider": "error",
                    "model": cloud_model,
                    "tokens": {"prompt": 0, "completion": 0, "total": 0},
                    "cost": 0,
                    "processing_info": {
                        "method": "cloud",
                        "model": cloud_model,
                        "fallback_used": False,
                        "error": str(e)
                    }
                }
    
    def _strategy_local_first(self, prompt: str, max_tokens: int, temperature: float,
                           budget: Optional[float], required_quality: str,
                           cloud_model: Optional[str], local_model: Optional[str]) -> Dict[str, Any]:
        """
        Стратегия с приоритетом локальной обработки.
        
        Сначала пытается использовать локальную модель, при ошибке или недостаточном
        качестве использует облачную.
        """
        # Пытаемся использовать локальную модель
        try:
            # Выбираем локальную модель, если не указана
            selected_local_model = local_model
            if selected_local_model is None:
                selected_local_model = self.local_llm_manager.select_optimal_model(
                    prompt=prompt,
                    required_quality=required_quality,
                    max_cost=budget
                )
            
            # Запрос к локальной модели
            result = self.local_llm_manager.query_model(
                prompt=prompt,
                model_name=selected_local_model,
                max_tokens=max_tokens,
                temperature=temperature,
                use_cache=True
            )
            
            # Проверяем наличие ошибки
            if "error" in result:
                raise Exception(result["error"])
            
            # Оцениваем качество ответа
            quality_score = self._estimate_response_quality(prompt, result["text"])
            
            # Если качество ниже порога и включен fallback, используем облачную модель
            if quality_score < self.quality_threshold and self.fallback_mode:
                self.logger.info(f"Качество локального ответа ({quality_score:.2f}) ниже порога ({self.quality_threshold}), используем облачную модель")
                
                # Проверяем, не превышает ли стоимость облачной обработки бюджет
                cloud_cost = self._estimate_cloud_cost(prompt, max_tokens, cloud_model)
                if budget is not None and cloud_cost > budget:
                    self.logger.warning(f"Стоимость облачной обработки ({cloud_cost} руб.) превышает бюджет ({budget} руб.), используем локальный результат")
                    
                    # Добавляем информацию о стратегии
                    result["processing_info"] = {
                        "method": "local",
                        "model": result.get("model", selected_local_model),
                        "fallback_used": False,
                        "quality_score": quality_score,
                        "fallback_skipped_reason": "budget_exceeded"
                    }
                    
                    return result
                
                # Запрос к облачной модели
                cloud_result = self.llm_service.query(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model=cloud_model,
                    budget=budget
                )
                
                # Добавляем информацию о стратегии
                cloud_result["processing_info"] = {
                    "method": "cloud",
                    "model": cloud_result.get("model", cloud_model),
                    "fallback_used": True,
                    "local_quality_score": quality_score,
                    "local_model": selected_local_model
                }
                
                return cloud_result
                
            else:
                # Добавляем информацию о стратегии
                result["processing_info"] = {
                    "method": "local",
                    "model": result.get("model", selected_local_model),
                    "fallback_used": False,
                    "quality_score": quality_score
                }
                
                return result
                
        except Exception as e:
            self.logger.error(f"Ошибка при локальной обработке: {str(e)}")
            
            # Если включен режим fallback, используем облачную модель
            if self.fallback_mode:
                self.logger.info("Используем облачную модель в качестве fallback")
                
                # Проверяем, не превышает ли стоимость облачной обработки бюджет
                cloud_cost = self._estimate_cloud_cost(prompt, max_tokens, cloud_model)
                if budget is not None and cloud_cost > budget:
                    self.logger.warning(f"Стоимость облачной обработки ({cloud_cost} руб.) превышает бюджет ({budget} руб.), возвращаем ошибку")
                    
                    return {
                        "error": f"Ошибка при локальной обработке: {str(e)}. Облачная обработка превышает бюджет.",
                        "text": "",
                        "provider": "error",
                        "model": local_model,
                        "tokens": {"prompt": 0, "completion": 0, "total": 0},
                        "cost": 0,
                        "processing_info": {
                            "method": "local",
                            "model": local_model,
                            "fallback_used": False,
                            "error": str(e),
                            "fallback_skipped_reason": "budget_exceeded"
                        }
                    }
                
                # Запрос к облачной модели
                result = self.llm_service.query(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model=cloud_model,
                    budget=budget
                )
                
                # Добавляем информацию о стратегии
                result["processing_info"] = {
                    "method": "cloud",
                    "model": result.get("model", cloud_model),
                    "fallback_used": True,
                    "local_error": str(e)
                }
                
                return result
                
            else:
                # Если fallback отключен, возвращаем ошибку
                return {
                    "error": f"Ошибка при локальной обработке: {str(e)}",
                    "text": "",
                    "provider": "error",
                    "model": local_model,
                    "tokens": {"prompt": 0, "completion": 0, "total": 0},
                    "cost": 0,
                    "processing_info": {
                        "method": "local",
                        "model": local_model,
                        "fallback_used": False,
                        "error": str(e)
                    }
                }
    
    def _strategy_cost_optimized(self, prompt: str, max_tokens: int, temperature: float,
                               budget: Optional[float], required_quality: str,
                               cloud_model: Optional[str], local_model: Optional[str]) -> Dict[str, Any]:
        """
        Стратегия с оптимизацией затрат.
        
        Выбирает наиболее дешевый вариант обработки, удовлетворяющий минимальным
        требованиям к качеству.
        """
        # Оцениваем затраты на облачную обработку
        cloud_cost = self._estimate_cloud_cost(prompt, max_tokens, cloud_model)
        
        # Оцениваем затраты на локальную обработку
        # Выбираем локальную модель, если не указана
        selected_local_model = local_model
        if selected_local_model is None:
            selected_local_model = self.local_llm_manager.select_optimal_model(
                prompt=prompt,
                required_quality=required_quality,
                max_cost=budget
            )
        
        # Получаем стоимость локальной обработки
        local_cost = self.local_llm_manager.local_cost_per_1k_tokens.get(selected_local_model, 0.0005) * (len(prompt.split()) + max_tokens) / 1000
        
        # Выбираем более дешевый вариант, учитывая бюджет
        if budget is not None and cloud_cost > budget:
            # Если облачная обработка превышает бюджет, используем локальную
            self.logger.info(f"Облачная обработка ({cloud_cost} руб.) превышает бюджет ({budget} руб.), используем локальную модель")
            
            result = self._use_local_model(
                prompt, max_tokens, temperature, budget, required_quality, local_model
            )
        elif local_cost <= cloud_cost * 0.8:  # Локальная обработка значительно дешевле
            self.logger.info(f"Локальная обработка ({local_cost:.6f} руб.) значительно дешевле облачной ({cloud_cost} руб.), используем локальную модель")
            
            result = self._use_local_model(
                prompt, max_tokens, temperature, budget, required_quality, local_model
            )
        else:
            # Если разница в стоимости небольшая, выбираем вариант в зависимости от требуемого качества
            if required_quality == "high":
                self.logger.info(f"Для высокого качества используем облачную модель (стоимость: {cloud_cost} руб.)")
                
                result = self._use_cloud_model(
                    prompt, max_tokens, temperature, budget, cloud_model
                )
            else:
                self.logger.info(f"Используем локальную модель для оптимизации затрат (стоимость: {local_cost:.6f} руб.)")
                
                result = self._use_local_model(
                    prompt, max_tokens, temperature, budget, required_quality, local_model
                )
        
        # Добавляем информацию о затратах
        result["cost_comparison"] = {
            "cloud_estimated_cost": cloud_cost,
            "local_estimated_cost": local_cost,
            "savings": cloud_cost - local_cost if result["processing_info"]["method"] == "local" else 0
        }
        
        return result
    
    def _strategy_quality_optimized(self, prompt: str, max_tokens: int, temperature: float,
                                  budget: Optional[float], required_quality: str,
                                  cloud_model: Optional[str], local_model: Optional[str]) -> Dict[str, Any]:
        """
        Стратегия с оптимизацией качества.
        
        Выбирает вариант обработки, обеспечивающий наилучшее качество в рамках
        доступного бюджета.
        """
        # Оцениваем затраты на облачную обработку
        cloud_cost = self._estimate_cloud_cost(prompt, max_tokens, cloud_model)
        
        # Проверяем, не превышает ли стоимость облачной обработки бюджет
        if budget is not None and cloud_cost > budget:
            self.logger.warning(f"Стоимость облачной обработки ({cloud_cost} руб.) превышает бюджет ({budget} руб.), используем локальную модель")
            
            result = self._use_local_model(
                prompt, max_tokens, temperature, budget, required_quality, local_model
            )
        else:
            # Если бюджет позволяет, используем облачную модель для наилучшего качества
            self.logger.info(f"Для оптимизации качества используем облачную модель (стоимость: {cloud_cost} руб.)")
            
            result = self._use_cloud_model(
                prompt, max_tokens, temperature, budget, cloud_model
            )
        
        return result
    
    def _strategy_balanced(self, prompt: str, max_tokens: int, temperature: float,
                         budget: Optional[float], required_quality: str,
                         cloud_model: Optional[str], local_model: Optional[str]) -> Dict[str, Any]:
        """
        Сбалансированная стратегия.
        
        Пытается найти баланс между затратами и качеством, учитывая сложность запроса
        и требуемое качество.
        """
        # Оцениваем сложность запроса
        prompt_complexity = self._estimate_prompt_complexity(prompt)
        
        # Оцениваем затраты на облачную обработку
        cloud_cost = self._estimate_cloud_cost(prompt, max_tokens, cloud_model)
        
        # Проверяем, не превышает ли стоимость облачной обработки бюджет
        if budget is not None and cloud_cost > budget:
            self.logger.warning(f"Стоимость облачной обработки ({cloud_cost} руб.) превышает бюджет ({budget} руб.), используем локальную модель")
            
            result = self._use_local_model(
                prompt, max_tokens, temperature, budget, required_quality, local_model
            )
        elif prompt_complexity == "high" or required_quality == "high":
            # Для сложных запросов или высокого качества используем облако
            self.logger.info(f"Для сложного запроса или высокого качества используем облачную модель (стоимость: {cloud_cost} руб.)")
            
            result = self._use_cloud_model(
                prompt, max_tokens, temperature, budget, cloud_model
            )
        elif prompt_complexity == "low" and required_quality in ["low", "medium"]:
            # Для простых запросов и невысокого качества используем локальную модель
            self.logger.info("Для простого запроса и невысокого качества используем локальную модель")
            
            result = self._use_local_model(
                prompt, max_tokens, temperature, budget, required_quality, local_model
            )
        else:
            # Для среднего уровня сложности и качества выбираем по затратам
            # Если уровень оптимизации затрат высокий, предпочитаем локальную модель
            if self.cost_optimization_level == "high":
                self.logger.info("Согласно высокому уровню оптимизации затрат используем локальную модель")
                
                result = self._use_local_model(
                    prompt, max_tokens, temperature, budget, required_quality, local_model
                )
            # Если уровень оптимизации затрат низкий, предпочитаем облачную модель
            elif self.cost_optimization_level == "low":
                self.logger.info("Согласно низкому уровню оптимизации затрат используем облачную модель")
                
                result = self._use_cloud_model(
                    prompt, max_tokens, temperature, budget, cloud_model
                )
            # Для среднего уровня оптимизации выбираем по соотношению стоимость/качество
            else:
                # Если облачная обработка недорогая, используем ее
                if cloud_cost < 5.0:  # Пример порога стоимости
                    self.logger.info(f"Облачная обработка недорогая ({cloud_cost} руб.), используем облачную модель")
                    
                    result = self._use_cloud_model(
                        prompt, max_tokens, temperature, budget, cloud_model
                    )
                else:
                    self.logger.info("Для сбалансированного подхода используем локальную модель")
                    
                    result = self._use_local_model(
                        prompt, max_tokens, temperature, budget, required_quality, local_model
                    )
        
        return result
    
    def _use_cloud_model(self, prompt: str, max_tokens: int, temperature: float,
                       budget: Optional[float], cloud_model: Optional[str]) -> Dict[str, Any]:
        """
        Использует облачную модель для обработки запроса.
        
        Args:
            prompt: Промпт для обработки
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации (от 0 до 1)
            budget: Максимальный бюджет в рублях (опционально)
            cloud_model: Облачная модель для использования (опционально)
            
        Returns:
            Dict[str, Any]: Результат обработки запроса
        """
        try:
            # Запрос к облачной модели
            result = self.llm_service.query(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                model=cloud_model,
                budget=budget
            )
            
            # Добавляем информацию о стратегии
            result["processing_info"] = {
                "method": "cloud",
                "model": result.get("model", cloud_model),
                "fallback_used": False
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка при облачной обработке: {str(e)}")
            
            # Если включен режим fallback, используем локальную модель
            if self.fallback_mode:
                self.logger.info("Используем локальную модель в качестве fallback")
                
                # Выбираем локальную модель
                selected_local_model = self.local_llm_manager.select_optimal_model(
                    prompt=prompt,
                    required_quality="medium",  # Средний уровень качества для fallback
                    max_cost=budget
                )
                
                # Запрос к локальной модели
                result = self.local_llm_manager.query_model(
                    prompt=prompt,
                    model_name=selected_local_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    use_cache=True
                )
                
                # Добавляем информацию о стратегии
                result["processing_info"] = {
                    "method": "local",
                    "model": result.get("model", selected_local_model),
                    "fallback_used": True,
                    "cloud_error": str(e)
                }
                
                return result
                
            else:
                # Если fallback отключен, возвращаем ошибку
                return {
                    "error": f"Ошибка при облачной обработке: {str(e)}",
                    "text": "",
                    "provider": "error",
                    "model": cloud_model,
                    "tokens": {"prompt": 0, "completion": 0, "total": 0},
                    "cost": 0,
                    "processing_info": {
                        "method": "cloud",
                        "model": cloud_model,
                        "fallback_used": False,
                        "error": str(e)
                    }
                }
    
    def _use_local_model(self, prompt: str, max_tokens: int, temperature: float,
                       budget: Optional[float], required_quality: str,
                       local_model: Optional[str]) -> Dict[str, Any]:
        """
        Использует локальную модель для обработки запроса.
        
        Args:
            prompt: Промпт для обработки
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации (от 0 до 1)
            budget: Максимальный бюджет в рублях (опционально)
            required_quality: Требуемое качество ответа ('low', 'medium', 'high')
            local_model: Локальная модель для использования (опционально)
            
        Returns:
            Dict[str, Any]: Результат обработки запроса
        """
        try:
            # Выбираем локальную модель, если не указана
            selected_local_model = local_model
            if selected_local_model is None:
                selected_local_model = self.local_llm_manager.select_optimal_model(
                    prompt=prompt,
                    required_quality=required_quality,
                    max_cost=budget
                )
            
            # Запрос к локальной модели
            result = self.local_llm_manager.query_model(
                prompt=prompt,
                model_name=selected_local_model,
                max_tokens=max_tokens,
                temperature=temperature,
                use_cache=True
            )
            
            # Проверяем наличие ошибки
            if "error" in result:
                raise Exception(result["error"])
            
            # Оцениваем качество ответа
            quality_score = self._estimate_response_quality(prompt, result["text"])
            
            # Если качество ниже порога и включен fallback, используем облачную модель
            if quality_score < self.quality_threshold and self.fallback_mode:
                self.logger.info(f"Качество локального ответа ({quality_score:.2f}) ниже порога ({self.quality_threshold}), используем облачную модель")
                
                # Проверяем, не превышает ли стоимость облачной обработки бюджет
                cloud_cost = self._estimate_cloud_cost(prompt, max_tokens)
                if budget is not None and cloud_cost > budget:
                    self.logger.warning(f"Стоимость облачной обработки ({cloud_cost} руб.) превышает бюджет ({budget} руб.), используем локальный результат")
                    
                    # Добавляем информацию о стратегии
                    result["processing_info"] = {
                        "method": "local",
                        "model": result.get("model", selected_local_model),
                        "fallback_used": False,
                        "quality_score": quality_score,
                        "fallback_skipped_reason": "budget_exceeded"
                    }
                    
                    return result
                
                # Запрос к облачной модели
                cloud_result = self.llm_service.query(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    budget=budget
                )
                
                # Добавляем информацию о стратегии
                cloud_result["processing_info"] = {
                    "method": "cloud",
                    "model": cloud_result.get("model"),
                    "fallback_used": True,
                    "local_quality_score": quality_score,
                    "local_model": selected_local_model
                }
                
                return cloud_result
                
            else:
                # Добавляем информацию о стратегии
                result["processing_info"] = {
                    "method": "local",
                    "model": result.get("model", selected_local_model),
                    "fallback_used": False,
                    "quality_score": quality_score
                }
                
                return result
                
        except Exception as e:
            self.logger.error(f"Ошибка при локальной обработке: {str(e)}")
            
            # Если включен режим fallback, используем облачную модель
            if self.fallback_mode:
                self.logger.info("Используем облачную модель в качестве fallback")
                
                # Проверяем, не превышает ли стоимость облачной обработки бюджет
                cloud_cost = self._estimate_cloud_cost(prompt, max_tokens)
                if budget is not None and cloud_cost > budget:
                    self.logger.warning(f"Стоимость облачной обработки ({cloud_cost} руб.) превышает бюджет ({budget} руб.), возвращаем ошибку")
                    
                    return {
                        "error": f"Ошибка при локальной обработке: {str(e)}. Облачная обработка превышает бюджет.",
                        "text": "",
                        "provider": "error",
                        "model": local_model,
                        "tokens": {"prompt": 0, "completion": 0, "total": 0},
                        "cost": 0,
                        "processing_info": {
                            "method": "local",
                            "model": local_model,
                            "fallback_used": False,
                            "error": str(e),
                            "fallback_skipped_reason": "budget_exceeded"
                        }
                    }
                
                # Запрос к облачной модели
                result = self.llm_service.query(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    budget=budget
                )
                
                # Добавляем информацию о стратегии
                result["processing_info"] = {
                    "method": "cloud",
                    "model": result.get("model"),
                    "fallback_used": True,
                    "local_error": str(e)
                }
                
                return result
                
            else:
                # Если fallback отключен, возвращаем ошибку
                return {
                    "error": f"Ошибка при локальной обработке: {str(e)}",
                    "text": "",
                    "provider": "error",
                    "model": local_model,
                    "tokens": {"prompt": 0, "completion": 0, "total": 0},
                    "cost": 0,
                    "processing_info": {
                        "method": "local",
                        "model": local_model,
                        "fallback_used": False,
                        "error": str(e)
                    }
                }
    
    def _estimate_cloud_cost(self, prompt: str, max_tokens: int, 
                           model: Optional[str] = None) -> float:
        """
        Оценивает стоимость облачной обработки запроса.
        
        Args:
            prompt: Промпт для обработки
            max_tokens: Максимальное количество токенов в ответе
            model: Облачная модель для использования (опционально)
            
        Returns:
            float: Оценка стоимости в рублях
        """
        # Оцениваем количество токенов в промпте
        prompt_tokens = len(prompt.split())
        
        # Получаем стоимость выбранной модели
        if model:
            cost_per_1k = self.cost_estimator.get_cost_per_1k_tokens(model)
        else:
            # Используем стоимость модели по умолчанию
            cost_per_1k = self.cost_estimator.get_cost_per_1k_tokens("gpt-3.5-turbo")
        
        # Рассчитываем стоимость
        cost = (prompt_tokens + max_tokens) * cost_per_1k / 1000
        
        return cost
    
    def _estimate_prompt_complexity(self, prompt: str) -> str:
        """
        Оценивает сложность промпта.
        
        Args:
            prompt: Промпт для оценки
            
        Returns:
            str: Сложность промпта ('low', 'medium', 'high')
        """
        # Простая эвристическая оценка сложности промпта
        
        # Длина промпта
        prompt_length = len(prompt)
        
        # Количество вопросительных и восклицательных знаков
        question_marks = prompt.count("?")
        exclamation_marks = prompt.count("!")
        
        # Ключевые слова, указывающие на сложность
        complex_keywords = [
            "анализ", "оценка", "сравнение", "объясни", "разработай",
            "создай", "предложи", "спрогнозируй", "стратегия",
            "analyze", "evaluate", "compare", "explain", "develop", 
            "create", "suggest", "predict", "strategy"
        ]
        
        # Подсчитываем количество сложных ключевых слов
        complex_count = sum(1 for keyword in complex_keywords if keyword.lower() in prompt.lower())
        
        # Оцениваем сложность на основе эвристик
        if prompt_length > 1000 or complex_count >= 3 or question_marks >= 3:
            return "high"
        elif prompt_length > 500 or complex_count >= 1 or question_marks >= 1:
            return "medium"
        else:
            return "low"
    
    def _estimate_response_quality(self, prompt: str, response: str) -> float:
        """
        Оценивает качество ответа.
        
        Args:
            prompt: Исходный промпт
            response: Ответ модели
            
        Returns:
            float: Оценка качества от 0 до 1
        """
        # В реальном проекте здесь был бы более сложный алгоритм оценки качества
        # Для примера используем простую эвристику
        
        # Длина ответа
        response_length = len(response)
        
        # Соотношение длины ответа к длине промпта
        length_ratio = min(1.0, response_length / max(1, len(prompt)) * 2)
        
        # Количество абзацев в ответе
        paragraphs = response.count("\n\n") + 1
        
        # Наличие структуры (заголовки, списки)
        has_structure = "\n#" in response or "\n-" in response or "\n*" in response or "\n1." in response
        
        # Разнообразие словаря
        unique_words = len(set(response.lower().split()))
        vocabulary_diversity = min(1.0, unique_words / max(1, len(response.split())) * 2)
        
        # Финальная оценка (пример)
        quality_score = (
            0.3 * length_ratio +
            0.2 * min(1.0, paragraphs / 5) +
            0.2 * (1.0 if has_structure else 0.5) +
            0.3 * vocabulary_diversity
        )
        
        return quality_score
    
    def configure(self, fallback_mode: Optional[bool] = None,
                quality_threshold: Optional[float] = None,
                cost_optimization_level: Optional[str] = None) -> None:
        """
        Настраивает параметры гибридного пайплайна.
        
        Args:
            fallback_mode: Использовать ли резервный метод при ошибках
            quality_threshold: Порог качества для переключения на облачную модель
            cost_optimization_level: Уровень оптимизации затрат ('low', 'medium', 'high')
        """
        if fallback_mode is not None:
            self.fallback_mode = fallback_mode
        
        if quality_threshold is not None:
            self.quality_threshold = max(0.0, min(1.0, quality_threshold))
        
        if cost_optimization_level is not None:
            if cost_optimization_level not in ["low", "medium", "high"]:
                self.logger.warning(f"Уровень оптимизации затрат {cost_optimization_level} не поддерживается, используем 'medium'")
                cost_optimization_level = "medium"
            
            self.cost_optimization_level = cost_optimization_level
