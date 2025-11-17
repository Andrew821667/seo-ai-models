"""
Анализатор затрат на API-вызовы и их оптимизация.

Модуль предоставляет функционал для оценки затрат на использование LLM API
и их оптимизации в рамках заданного бюджета.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Импортируем компоненты из common
from ..common.constants import TOKEN_COSTS, LLM_MODELS
from ..common.exceptions import BudgetExceededError
from ..common.utils import (
    estimate_tokens,
    estimate_cost,
    chunk_text
)


class CostEstimator:
    """
    Анализатор затрат на API-вызовы и их оптимизация.
    """
    
    def __init__(self):
        """
        Инициализирует анализатор затрат.
        """
        # Статистика использования по провайдерам и моделям
        self.usage_stats = {
            "total_cost": 0.0,
            "total_tokens": 0,
            "providers": {}
        }
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
    
    def estimate_request_cost(self, provider: str, model: str, 
                             prompt_length: int, 
                             expected_response_length: Optional[int] = None) -> Dict[str, float]:
        """
        Оценивает стоимость запроса к API.
        
        Args:
            provider: Название провайдера
            model: Название модели
            prompt_length: Длина промпта в символах
            expected_response_length: Ожидаемая длина ответа в символах (опционально)
            
        Returns:
            Dict[str, float]: Оценка стоимости
                - tokens: Общее количество токенов
                - cost: Стоимость в рублях
        """
        # Оцениваем количество токенов в промпте
        input_tokens = estimate_tokens(prompt_length * " ", model)
        
        # Если ожидаемая длина ответа не указана, предполагаем
        # 25% от длины промпта, но не более 2048 токенов
        if expected_response_length is None:
            output_tokens = min(2048, input_tokens // 4)
        else:
            output_tokens = estimate_tokens(expected_response_length * " ", model)
        
        # Оцениваем стоимость
        cost = estimate_cost(input_tokens, output_tokens, provider, model)
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost
        }
    
    def optimize_content_for_budget(self, content: str, provider: str, model: str, 
                                  budget: float) -> Tuple[List[str], Dict[str, Any]]:
        """
        Оптимизирует контент для соответствия бюджету.
        
        Args:
            content: Исходный контент
            provider: Название провайдера
            model: Название модели
            budget: Максимальный бюджет в рублях
            
        Returns:
            Tuple[List[str], Dict[str, Any]]: Кортеж (список чанков контента, метаданные оптимизации)
        """
        # Оцениваем стоимость обработки всего контента
        content_length = len(content)
        estimate = self.estimate_request_cost(provider, model, content_length)
        
        # Если стоимость меньше бюджета, возвращаем весь контент как один чанк
        if estimate["cost"] <= budget:
            return [content], {
                "original_content_length": content_length,
                "chunks": 1,
                "estimated_cost": estimate["cost"],
                "budget": budget,
                "optimization_needed": False
            }
        
        # Если стоимость превышает бюджет, разбиваем контент на чанки
        # Определяем максимальную длину чанка, исходя из бюджета
        
        # Примерная стоимость на символ
        cost_per_symbol = estimate["cost"] / content_length
        
        # Максимальная длина чанка в символах
        max_chunk_length = int(budget / cost_per_symbol)
        
        # Разбиваем контент на чанки
        chunks = chunk_text(content, max_chunk_length)
        
        # Оцениваем стоимость обработки каждого чанка
        total_estimated_cost = 0.0
        chunk_costs = []
        
        for chunk in chunks:
            chunk_estimate = self.estimate_request_cost(provider, model, len(chunk))
            total_estimated_cost += chunk_estimate["cost"]
            chunk_costs.append(chunk_estimate["cost"])
        
        # Если даже после разбиения стоимость превышает бюджет,
        # убираем наименее важные чанки (пока что просто последние)
        while total_estimated_cost > budget and len(chunks) > 1:
            removed_chunk_cost = chunk_costs.pop()
            total_estimated_cost -= removed_chunk_cost
            chunks.pop()
        
        return chunks, {
            "original_content_length": content_length,
            "chunks": len(chunks),
            "estimated_cost": total_estimated_cost,
            "budget": budget,
            "optimization_needed": True,
            "chunk_costs": chunk_costs,
            "content_coverage": sum(len(chunk) for chunk in chunks) / content_length
        }
    
    def record_usage(self, provider: str, model: str, 
                    input_tokens: int, output_tokens: int) -> None:
        """
        Записывает статистику использования.
        
        Args:
            provider: Название провайдера
            model: Название модели
            input_tokens: Количество токенов ввода
            output_tokens: Количество токенов вывода
        """
        # Получаем текущую дату для группировки статистики
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Рассчитываем стоимость
        cost = estimate_cost(input_tokens, output_tokens, provider, model)
        
        # Обновляем общую статистику
        self.usage_stats["total_cost"] += cost
        self.usage_stats["total_tokens"] += input_tokens + output_tokens
        
        # Инициализируем структуру для провайдера, если ее еще нет
        if provider not in self.usage_stats["providers"]:
            self.usage_stats["providers"][provider] = {
                "total_cost": 0.0,
                "total_tokens": 0,
                "models": {}
            }
        
        # Обновляем статистику провайдера
        provider_stats = self.usage_stats["providers"][provider]
        provider_stats["total_cost"] += cost
        provider_stats["total_tokens"] += input_tokens + output_tokens
        
        # Инициализируем структуру для модели, если ее еще нет
        if model not in provider_stats["models"]:
            provider_stats["models"][model] = {
                "total_cost": 0.0,
                "total_tokens": 0,
                "daily_stats": {}
            }
        
        # Обновляем статистику модели
        model_stats = provider_stats["models"][model]
        model_stats["total_cost"] += cost
        model_stats["total_tokens"] += input_tokens + output_tokens
        
        # Инициализируем структуру для текущего дня, если ее еще нет
        if today not in model_stats["daily_stats"]:
            model_stats["daily_stats"][today] = {
                "cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "requests": 0
            }
        
        # Обновляем статистику текущего дня
        daily_stats = model_stats["daily_stats"][today]
        daily_stats["cost"] += cost
        daily_stats["input_tokens"] += input_tokens
        daily_stats["output_tokens"] += output_tokens
        daily_stats["total_tokens"] += input_tokens + output_tokens
        daily_stats["requests"] += 1
    
    def get_usage_stats(self, provider: Optional[str] = None, 
                      model: Optional[str] = None,
                      days: Optional[int] = None) -> Dict[str, Any]:
        """
        Возвращает статистику использования.
        
        Args:
            provider: Название провайдера (опционально)
            model: Название модели (опционально)
            days: Количество последних дней для статистики (опционально)
            
        Returns:
            Dict[str, Any]: Статистика использования
        """
        # Если указан провайдер
        if provider:
            # Если провайдер не найден, возвращаем пустую статистику
            if provider not in self.usage_stats["providers"]:
                return {
                    "provider": provider,
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "models": {}
                }
            
            provider_stats = self.usage_stats["providers"][provider]
            
            # Если указана модель
            if model:
                # Если модель не найдена, возвращаем пустую статистику
                if model not in provider_stats["models"]:
                    return {
                        "provider": provider,
                        "model": model,
                        "total_cost": 0.0,
                        "total_tokens": 0,
                        "daily_stats": {}
                    }
                
                model_stats = provider_stats["models"][model]
                
                # Если указано количество дней, фильтруем статистику
                if days:
                    # Получаем последние days дней
                    cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                    filtered_daily_stats = {
                        date: stats for date, stats in model_stats["daily_stats"].items()
                        if date >= cutoff_date
                    }
                    
                    # Рассчитываем общую статистику за выбранный период
                    period_cost = sum(stats["cost"] for stats in filtered_daily_stats.values())
                    period_input_tokens = sum(stats["input_tokens"] for stats in filtered_daily_stats.values())
                    period_output_tokens = sum(stats["output_tokens"] for stats in filtered_daily_stats.values())
                    period_total_tokens = sum(stats["total_tokens"] for stats in filtered_daily_stats.values())
                    period_requests = sum(stats["requests"] for stats in filtered_daily_stats.values())
                    
                    return {
                        "provider": provider,
                        "model": model,
                        "period_days": days,
                        "period_cost": period_cost,
                        "period_input_tokens": period_input_tokens,
                        "period_output_tokens": period_output_tokens,
                        "period_total_tokens": period_total_tokens,
                        "period_requests": period_requests,
                        "daily_stats": filtered_daily_stats
                    }
                
                # Если количество дней не указано, возвращаем всю статистику
                return {
                    "provider": provider,
                    "model": model,
                    "total_cost": model_stats["total_cost"],
                    "total_tokens": model_stats["total_tokens"],
                    "daily_stats": model_stats["daily_stats"]
                }
            
            # Если модель не указана, возвращаем статистику провайдера
            return {
                "provider": provider,
                "total_cost": provider_stats["total_cost"],
                "total_tokens": provider_stats["total_tokens"],
                "models": provider_stats["models"]
            }
        
        # Если не указан ни провайдер, ни модель, возвращаем общую статистику
        return self.usage_stats
    
    def save_usage_stats(self, filename: str) -> None:
        """
        Сохраняет статистику использования в файл.
        
        Args:
            filename: Имя файла
        """
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.usage_stats, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Статистика использования сохранена в файл {filename}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении статистики использования: {e}")
    
    def load_usage_stats(self, filename: str) -> None:
        """
        Загружает статистику использования из файла.
        
        Args:
            filename: Имя файла
        """
        try:
            with open(filename, "r", encoding="utf-8") as f:
                self.usage_stats = json.load(f)
            self.logger.info(f"Статистика использования загружена из файла {filename}")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке статистики использования: {e}")
    
    def forecast_monthly_cost(self, daily_budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Прогнозирует месячные затраты на основе статистики.
        
        Args:
            daily_budget: Дневной бюджет в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Прогноз затрат
        """
        # Получаем статистику за последние 7 дней
        seven_day_stats = self.get_usage_stats(days=7)
        
        # Если нет статистики, возвращаем нулевой прогноз
        if not seven_day_stats["providers"]:
            return {
                "monthly_forecast": 0.0,
                "daily_average": 0.0,
                "monthly_by_provider": {}
            }
        
        # Собираем статистику затрат по дням
        daily_costs = {}
        
        for provider_name, provider_stats in seven_day_stats["providers"].items():
            for model_name, model_stats in provider_stats["models"].items():
                for date, stats in model_stats["daily_stats"].items():
                    if date not in daily_costs:
                        daily_costs[date] = 0.0
                    daily_costs[date] += stats["cost"]
        
        # Рассчитываем среднедневные затраты
        total_days = len(daily_costs)
        if total_days == 0:
            daily_average = 0.0
        else:
            daily_average = sum(daily_costs.values()) / total_days
        
        # Прогнозируем месячные затраты (умножаем на 30)
        monthly_forecast = daily_average * 30
        
        # Рассчитываем прогноз по провайдерам
        monthly_by_provider = {}
        
        for provider_name, provider_stats in seven_day_stats["providers"].items():
            provider_daily_avg = provider_stats["total_cost"] / max(total_days, 1)
            monthly_by_provider[provider_name] = provider_daily_avg * 30
        
        # Добавляем информацию о соответствии бюджету
        budget_info = {}
        if daily_budget is not None:
            monthly_budget = daily_budget * 30
            budget_info = {
                "daily_budget": daily_budget,
                "monthly_budget": monthly_budget,
                "within_budget": daily_average <= daily_budget,
                "budget_utilization": daily_average / daily_budget if daily_budget > 0 else 0,
                "days_until_budget_exceeded": (monthly_budget / monthly_forecast) * 30 if monthly_forecast > 0 else float('inf')
            }
        
        return {
            "monthly_forecast": monthly_forecast,
            "daily_average": daily_average,
            "monthly_by_provider": monthly_by_provider,
            "total_days_in_stats": total_days,
            "budget_info": budget_info
        }
    
    def optimize_usage(self, target_cost_reduction: float) -> Dict[str, Any]:
        """
        Предлагает оптимизации для снижения затрат.
        
        Args:
            target_cost_reduction: Целевое снижение затрат в рублях
            
        Returns:
            Dict[str, Any]: Рекомендации по оптимизации
        """
        # Получаем текущую статистику
        stats = self.get_usage_stats(days=30)
        
        # Если нет данных, возвращаем пустые рекомендации
        if not stats["providers"]:
            return {
                "message": "Недостаточно данных для оптимизации",
                "recommendations": []
            }
        
        # Анализируем использование по провайдерам и моделям
        model_usage = []
        
        for provider_name, provider_stats in stats["providers"].items():
            for model_name, model_stats in provider_stats["models"].items():
                model_usage.append({
                    "provider": provider_name,
                    "model": model_name,
                    "cost": model_stats["total_cost"],
                    "tokens": model_stats["total_tokens"],
                    "cost_per_token": model_stats["total_cost"] / max(model_stats["total_tokens"], 1)
                })
        
        # Сортируем по стоимости (от большей к меньшей)
        model_usage.sort(key=lambda x: x["cost"], reverse=True)
        
        # Генерируем рекомендации
        recommendations = []
        estimated_savings = 0.0
        
        # Для самых дорогих моделей предлагаем альтернативы
        for i, usage in enumerate(model_usage):
            if i < 3:  # Анализируем только top-3 самые дорогие модели
                provider = usage["provider"]
                model = usage["model"]
                
                # Ищем более дешевые альтернативы в том же провайдере
                cheaper_alternatives = []
                
                for alt_model in LLM_MODELS.get(provider, []):
                    if alt_model != model:
                        # Оцениваем стоимость на 1000 токенов (500 вход + 500 выход)
                        alt_cost = estimate_cost(500, 500, provider, alt_model)
                        current_cost = estimate_cost(500, 500, provider, model)
                        
                        if alt_cost < current_cost:
                            savings_percent = (current_cost - alt_cost) / current_cost * 100
                            cheaper_alternatives.append({
                                "model": alt_model,
                                "savings_percent": savings_percent,
                                "estimated_monthly_savings": usage["cost"] * (savings_percent / 100)
                            })
                
                # Сортируем альтернативы по экономии
                cheaper_alternatives.sort(key=lambda x: x["savings_percent"], reverse=True)
                
                if cheaper_alternatives:
                    best_alternative = cheaper_alternatives[0]
                    recommendations.append({
                        "type": "model_change",
                        "provider": provider,
                        "current_model": model,
                        "recommended_model": best_alternative["model"],
                        "savings_percent": best_alternative["savings_percent"],
                        "estimated_monthly_savings": best_alternative["estimated_monthly_savings"]
                    })
                    estimated_savings += best_alternative["estimated_monthly_savings"]
        
        # Предлагаем оптимизации контента
        recommendations.append({
            "type": "chunking",
            "description": "Разбиение контента на меньшие чанки может снизить затраты",
            "estimated_savings_percent": 15,
            "estimated_monthly_savings": stats["total_cost"] * 0.15
        })
        estimated_savings += stats["total_cost"] * 0.15
        
        # Предлагаем кэширование
        recommendations.append({
            "type": "caching",
            "description": "Кэширование частых запросов может снизить затраты",
            "estimated_savings_percent": 20,
            "estimated_monthly_savings": stats["total_cost"] * 0.2
        })
        estimated_savings += stats["total_cost"] * 0.2
        
        return {
            "current_monthly_cost": stats["total_cost"],
            "target_cost_reduction": target_cost_reduction,
            "estimated_savings": estimated_savings,
            "target_achieved": estimated_savings >= target_cost_reduction,
            "recommendations": recommendations
        }
