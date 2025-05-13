"""
Система умного выбора и комбинирования результатов разных LLM.

Модуль предоставляет функционал для выбора оптимальной модели LLM
для конкретной задачи и комбинирования результатов нескольких моделей.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Импортируем компоненты из common
from ..common.constants import (
    LLM_PROVIDERS, 
    LLM_MODELS, 
    DEFAULT_MODELS,
    TOKEN_COSTS
)
from ..common.exceptions import (
    ProviderNotSupportedError,
    ModelNotSupportedError,
    BudgetExceededError
)
from ..common.utils import (
    estimate_cost,
    extract_scores_from_text,
    parse_json_response
)

# Импортируем остальные компоненты
from .llm_service import LLMService
from .prompt_generator import PromptGenerator


class MultiModelAgent:
    """
    Агент для выбора и комбинирования результатов разных LLM.
    """
    
    def __init__(self, llm_service: LLMService, prompt_generator: PromptGenerator):
        """
        Инициализирует агента.
        
        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM
            prompt_generator: Экземпляр PromptGenerator для генерации промптов
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
    
    def select_optimal_model(self, task_type: str, content_length: int, 
                           budget: Optional[float] = None) -> Tuple[str, str]:
        """
        Выбирает оптимальную модель для конкретной задачи.
        
        Args:
            task_type: Тип задачи (analysis, generation, classification, etc.)
            content_length: Длина контента в символах
            budget: Максимальный бюджет в рублях (опционально)
            
        Returns:
            Tuple[str, str]: Кортеж (провайдер, модель)
            
        Raises:
            BudgetExceededError: Если все доступные модели превышают бюджет
        """
        # Получаем список доступных провайдеров
        available_providers = self.llm_service.get_available_providers()
        
        # Если нет доступных провайдеров, возвращаем модель по умолчанию
        if not available_providers:
            return DEFAULT_MODELS["standard"]["provider"], DEFAULT_MODELS["standard"]["model"]
        
        # Выбираем оптимальную модель в зависимости от типа задачи и длины контента
        if task_type == "analysis":
            # Для анализа требуется более мощная модель
            if content_length > 50000:  # Очень длинный контент
                tier = "micro"  # Используем легкую модель для большого контента
            elif content_length > 20000:  # Длинный контент
                tier = "basic"
            elif content_length > 5000:  # Средний контент
                tier = "standard"
            else:  # Короткий контент
                tier = "premium"
        else:
            # Для других задач можно использовать более легкие модели
            if content_length > 30000:
                tier = "micro"
            elif content_length > 10000:
                tier = "basic"
            else:
                tier = "standard"
        
        # Получаем модель по умолчанию для выбранного уровня
        provider = DEFAULT_MODELS[tier]["provider"]
        model = DEFAULT_MODELS[tier]["model"]
        
        # Если провайдер не доступен, выбираем первый доступный
        if provider not in available_providers:
            provider = available_providers[0]
            # Получаем модель для выбранного провайдера
            model = self.llm_service.get_provider_model(provider)
        
        # Если указан бюджет, проверяем, что выбранная модель вписывается в него
        if budget is not None:
            # Предполагаемое количество токенов для запроса
            input_tokens = content_length // 4  # ~4 символа на токен
            output_tokens = min(2048, input_tokens // 2)  # Предполагаем длину ответа
            
            # Оцениваем стоимость
            cost = estimate_cost(input_tokens, output_tokens, provider, model)
            
            # Если стоимость превышает бюджет, пробуем выбрать более дешевую модель
            if cost > budget:
                # Перебираем провайдеров и модели в порядке возрастания стоимости
                available_options = []
                
                for available_provider in available_providers:
                    provider_model = self.llm_service.get_provider_model(available_provider)
                    provider_cost = estimate_cost(input_tokens, output_tokens, available_provider, provider_model)
                    available_options.append((available_provider, provider_model, provider_cost))
                
                # Сортируем по стоимости
                available_options.sort(key=lambda x: x[2])
                
                # Выбираем первую модель, вписывающуюся в бюджет
                for option in available_options:
                    if option[2] <= budget:
                        return option[0], option[1]
                
                # Если нет моделей, вписывающихся в бюджет, выбрасываем исключение
                raise BudgetExceededError(budget, min([option[2] for option in available_options]))
        
        return provider, model
    
    def generate_parallel(self, prompt: str, providers: List[Tuple[str, str]], 
                         timeout: int = 60) -> List[Dict[str, Any]]:
        """
        Параллельно генерирует ответы от нескольких провайдеров.
        
        Args:
            prompt: Текст промпта
            providers: Список кортежей (провайдер, модель)
            timeout: Таймаут выполнения в секундах
            
        Returns:
            List[Dict[str, Any]]: Список ответов от провайдеров
        """
        results = []
        
        # Функция для выполнения запроса к провайдеру
        def execute_request(provider_tuple):
            provider, model = provider_tuple
            # Оптимизируем промпт для данного провайдера
            optimized_prompt = self.prompt_generator.optimize_for_provider(
                prompt, provider, model
            )
            # Генерируем ответ
            try:
                return self.llm_service.generate(optimized_prompt, provider)
            except Exception as e:
                self.logger.error(f"Ошибка при генерации ответа от {provider}: {e}")
                return {
                    "provider": provider,
                    "model": model,
                    "error": str(e),
                    "text": None
                }
        
        # Параллельно выполняем запросы
        with ThreadPoolExecutor() as executor:
            # Создаем задачи для выполнения
            future_to_provider = {
                executor.submit(execute_request, provider_tuple): provider_tuple
                for provider_tuple in providers
            }
            
            # Получаем результаты по мере выполнения
            for future in as_completed(future_to_provider, timeout=timeout):
                provider_tuple = future_to_provider[future]
                try:
                    result = future.result()
                    result["provider"] = provider_tuple[0]
                    result["model"] = provider_tuple[1]
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Ошибка при выполнении запроса к {provider_tuple}: {e}")
                    results.append({
                        "provider": provider_tuple[0],
                        "model": provider_tuple[1],
                        "error": str(e),
                        "text": None
                    })
        
        return results
    
    def combine_results(self, results: List[Dict[str, Any]], 
                       combination_method: str = "weighted") -> Dict[str, Any]:
        """
        Комбинирует результаты от нескольких провайдеров.
        
        Args:
            results: Список ответов от провайдеров
            combination_method: Метод комбинирования (weighted, voting, best)
            
        Returns:
            Dict[str, Any]: Комбинированный результат
        """
        # Отфильтровываем результаты с ошибками
        valid_results = [r for r in results if r.get("text") is not None]
        
        # Если нет валидных результатов, возвращаем пустой результат
        if not valid_results:
            return {
                "text": None,
                "error": "Все провайдеры вернули ошибки",
                "providers": [r.get("provider") for r in results]
            }
        
        # Если только один валидный результат, возвращаем его
        if len(valid_results) == 1:
            return valid_results[0]
        
        # Комбинируем результаты в зависимости от метода
        if combination_method == "best":
            # Возвращаем результат от самой мощной модели
            # (предполагаем, что более мощные модели дороже)
            valid_results.sort(
                key=lambda r: estimate_cost(1000, 1000, r.get("provider"), r.get("model")),
                reverse=True
            )
            return valid_results[0]
        
        elif combination_method == "voting":
            # Для простых классификаций можно использовать голосование
            # Собираем все ответы и выбираем наиболее часто встречающийся
            # (работает только для задач с простыми ответами)
            answers = [r.get("text").strip().lower() for r in valid_results]
            answer_counts = {}
            
            for answer in answers:
                answer_counts[answer] = answer_counts.get(answer, 0) + 1
            
            most_common_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
            
            return {
                "text": most_common_answer,
                "providers": [r.get("provider") for r in valid_results],
                "models": [r.get("model") for r in valid_results],
                "votes": answer_counts
            }
        
        else:  # weighted (по умолчанию)
            # Взвешенное комбинирование результатов
            # Более мощные модели получают больший вес
            # Извлекаем оценки из текстов ответов
            scores_by_model = {}
            
            for result in valid_results:
                provider = result.get("provider")
                model = result.get("model")
                text = result.get("text")
                
                # Извлекаем оценки из текста
                scores = extract_scores_from_text(text)
                
                # Веса моделей (пропорциональны стоимости)
                model_weight = estimate_cost(1000, 1000, provider, model)
                scores_by_model[f"{provider}_{model}"] = {
                    "scores": scores,
                    "weight": model_weight,
                    "text": text
                }
            
            # Нормализуем веса
            total_weight = sum(info["weight"] for info in scores_by_model.values())
            
            for model, info in scores_by_model.items():
                info["normalized_weight"] = info["weight"] / total_weight if total_weight > 0 else 1 / len(scores_by_model)
            
            # Комбинируем оценки с учетом весов
            combined_scores = {}
            seen_categories = set()
            
            for model, info in scores_by_model.items():
                for category, score in info["scores"].items():
                    seen_categories.add(category)
                    
                    if category not in combined_scores:
                        combined_scores[category] = 0
                    
                    combined_scores[category] += score * info["normalized_weight"]
            
            # Для текстовых результатов выбираем ответ от модели с наибольшим весом
            best_model = max(scores_by_model.items(), key=lambda x: x[1]["weight"])[0]
            combined_text = scores_by_model[best_model]["text"]
            
            return {
                "text": combined_text,
                "scores": combined_scores,
                "providers": [r.get("provider") for r in valid_results],
                "models": [r.get("model") for r in valid_results],
                "weights": {model: info["normalized_weight"] for model, info in scores_by_model.items()}
            }
    
    # Продолжение файла multi_model_agent.py
    def analyze_content(self, content: str, analysis_type: str, 
                      budget: Optional[float] = None, 
                      use_multiple_models: bool = False) -> Dict[str, Any]:
        """
        Анализирует контент с помощью оптимальной модели или нескольких моделей.
        
        Args:
            content: Текст для анализа
            analysis_type: Тип анализа (compatibility, citability, structure, eeat, semantic)
            budget: Максимальный бюджет в рублях (опционально)
            use_multiple_models: Использовать несколько моделей для анализа
            
        Returns:
            Dict[str, Any]: Результат анализа
        """
        # Выбираем оптимальную модель для анализа
        provider, model = self.select_optimal_model("analysis", len(content), budget)
        
        # Генерируем промпт для анализа
        prompt = self.prompt_generator.generate_analysis_prompt(
            analysis_type, content, provider, model
        )
        
        # Если нужно использовать несколько моделей
        if use_multiple_models:
            # Получаем список доступных провайдеров
            available_providers = self.llm_service.get_available_providers()
            
            # Если бюджет ограничен, выбираем только модели, вписывающиеся в бюджет
            providers_to_use = []
            
            # Предполагаемое количество токенов для запроса
            input_tokens = len(content) // 4  # ~4 символа на токен
            output_tokens = min(2048, input_tokens // 2)  # Предполагаем длину ответа
            
            for provider_name in available_providers:
                model_name = self.llm_service.get_provider_model(provider_name)
                
                # Если бюджет не указан или модель вписывается в бюджет, добавляем ее
                if budget is None:
                    providers_to_use.append((provider_name, model_name))
                else:
                    cost = estimate_cost(input_tokens, output_tokens, provider_name, model_name)
                    if cost <= budget / len(available_providers):  # Делим бюджет на всех
                        providers_to_use.append((provider_name, model_name))
            
            # Если нет подходящих моделей, используем только выбранную оптимальную
            if not providers_to_use:
                providers_to_use = [(provider, model)]
            
            # Генерируем ответы параллельно
            results = self.generate_parallel(prompt, providers_to_use)
            
            # Комбинируем результаты
            return self.combine_results(results, "weighted")
        
        else:
            # Используем только выбранную оптимальную модель
            result = self.llm_service.generate(prompt, provider)
            result["provider"] = provider
            result["model"] = model
            
            return result
    
    def get_multiple_opinions(self, content: str, question: str, 
                            providers: Optional[List[Tuple[str, str]]] = None,
                            timeout: int = 60) -> Dict[str, Any]:
        """
        Получает мнения от нескольких моделей по заданному вопросу.
        
        Args:
            content: Текст для анализа
            question: Вопрос или задание для моделей
            providers: Список кортежей (провайдер, модель) (опционально)
            timeout: Таймаут выполнения в секундах
            
        Returns:
            Dict[str, Any]: Комбинированный результат мнений
        """
        # Если список провайдеров не указан, используем все доступные
        if providers is None:
            available_providers = self.llm_service.get_available_providers()
            providers = [
                (provider, self.llm_service.get_provider_model(provider))
                for provider in available_providers
            ]
        
        # Формируем промпт с вопросом и контентом
        prompt = f"""
        {question}
        
        Текст для анализа:
        {content}
        """
        
        # Генерируем ответы параллельно
        results = self.generate_parallel(prompt, providers, timeout)
        
        # Комбинируем результаты
        return self.combine_results(results, "best")
    
    def extract_insights(self, content: str, 
                        specific_aspects: Optional[List[str]] = None,
                        budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Извлекает инсайты из контента с использованием LLM.
        
        Args:
            content: Текст для анализа
            specific_aspects: Список конкретных аспектов для анализа (опционально)
            budget: Максимальный бюджет в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Извлеченные инсайты
        """
        # Выбираем оптимальную модель для анализа
        provider, model = self.select_optimal_model("analysis", len(content), budget)
        
        # Формируем промпт для извлечения инсайтов
        aspects_prompt = ""
        if specific_aspects:
            aspects_prompt = "\n\nОсобое внимание удели следующим аспектам:\n"
            for aspect in specific_aspects:
                aspects_prompt += f"- {aspect}\n"
        
        prompt = f"""
        Проанализируй следующий текст и выдели ключевые инсайты.
        Для каждого инсайта дай оценку его важности от 1 до 10.
        Результат представь в формате JSON с полями:
        1. insights - список объектов с полями "insight" (текст инсайта) и "importance" (оценка важности)
        2. summary - краткое резюме всех инсайтов
        3. key_topics - список ключевых тем, затронутых в тексте
        {aspects_prompt}
        
        Текст для анализа:
        {content}
        """
        
        # Генерируем ответ
        result = self.llm_service.generate(prompt, provider)
        
        # Пытаемся извлечь JSON из ответа
        try:
            insights_data = parse_json_response(result["text"])
            
            # Если JSON не извлечен, структурируем ответ сами
            if not insights_data or "insights" not in insights_data:
                insights_data = {
                    "insights": [],
                    "summary": "",
                    "key_topics": []
                }
                
                # Пытаемся извлечь инсайты из текстового ответа
                for line in result["text"].split("\n"):
                    if ":" in line and any(c.isdigit() for c in line):
                        parts = line.split(":")
                        insight_text = parts[0].strip()
                        
                        # Извлекаем оценку важности
                        importance = 5  # По умолчанию
                        for part in parts[1:]:
                            for char in part:
                                if char.isdigit():
                                    importance = int(char)
                                    break
                            if importance != 5:
                                break
                        
                        insights_data["insights"].append({
                            "insight": insight_text,
                            "importance": importance
                        })
                
                # Добавляем исходный текст ответа
                insights_data["raw_text"] = result["text"]
            
            return {
                "insights_data": insights_data,
                "provider": provider,
                "model": model,
                "tokens": result.get("tokens", {}),
                "cost": result.get("cost", 0)
            }
            
        except Exception as e:
            # В случае ошибки при обработке JSON, возвращаем ответ как есть
            self.logger.error(f"Ошибка при обработке JSON: {e}")
            return {
                "text": result["text"],
                "provider": provider,
                "model": model,
                "tokens": result.get("tokens", {}),
                "cost": result.get("cost", 0),
                "error": str(e)
            }
