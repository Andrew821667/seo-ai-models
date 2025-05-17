"""
Менеджер локальных LLM-моделей.

Модуль предоставляет функционал для работы с локальными LLM-моделями,
что позволяет снизить затраты и обеспечить работу без постоянного
подключения к облачным API.
"""

import os
import json
import logging
import subprocess
import tempfile
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.cost_estimator import CostEstimator
from ..common.utils import parse_json_response

class LocalLLMManager:
    """
    Менеджер для работы с локальными LLM-моделями.
    """
    
    def __init__(self, models_dir: Optional[str] = None, 
              device: str = "cpu", 
              max_memory: Optional[int] = None):
        """
        Инициализирует менеджер локальных LLM-моделей.
        
        Args:
            models_dir: Директория с локальными моделями (опционально)
            device: Устройство для запуска моделей ('cpu', 'cuda', 'mps')
            max_memory: Максимальный объем используемой памяти в МБ (опционально)
        """
        # Настройка директории моделей
        self.models_dir = models_dir or os.path.expanduser("~/.cache/seo_ai_models/llm_models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Настройка устройства
        self.device = device
        self.max_memory = max_memory
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
        
        # Список доступных локальных моделей
        self.available_models = []
        
        # Загруженная модель
        self.loaded_model = None
        self.loaded_model_name = None
        
        # Список поддерживаемых моделей
        self.supported_models = [
            # Маленькие модели для слабого железа
            "TinyLlama-1.1B-Chat",
            "phi-2",
            "gemma-2b",
            "qwen1.5-0.5b",
            "Mistral-7B-Instruct-v0.2",
            
            # Средние модели (7-14B)
            "Mistral-7B-Instruct-v0.2",
            "Llama-2-7b-chat",
            "Llama-3-8b-instruct",
            "mpt-7b-instruct",
            "gemma-7b",
            "qwen1.5-7b",
            
            # Большие модели (>14B)
            "Llama-2-13b-chat",
            "Llama-3-70b-instruct",
            "mixtral-8x7b",
            "qwen1.5-14b",
            "qwen1.5-32b",
            "qwen1.5-72b",
            "gemma-27b"
        ]
        
        # Стоимость локальных моделей (ориентировочно в рублях за 1000 токенов)
        self.local_cost_per_1k_tokens = {
            # Маленькие модели
            "TinyLlama-1.1B-Chat": 0.0001,
            "phi-2": 0.0002,
            "gemma-2b": 0.0002,
            "qwen1.5-0.5b": 0.0001,
            
            # Средние модели
            "Mistral-7B-Instruct-v0.2": 0.0005,
            "Llama-2-7b-chat": 0.0005,
            "Llama-3-8b-instruct": 0.0006,
            "mpt-7b-instruct": 0.0005,
            "gemma-7b": 0.0005,
            "qwen1.5-7b": 0.0005,
            
            # Большие модели
            "Llama-2-13b-chat": 0.001,
            "Llama-3-70b-instruct": 0.005,
            "mixtral-8x7b": 0.002,
            "qwen1.5-14b": 0.001,
            "qwen1.5-32b": 0.003,
            "qwen1.5-72b": 0.006,
            "gemma-27b": 0.003
        }
        
        # Сканируем доступные модели
        self._scan_available_models()
    
    def _scan_available_models(self) -> None:
        """
        Сканирует доступные локальные модели.
        """
        self.available_models = []
        
        # Проверяем, что директория существует
        if not os.path.exists(self.models_dir):
            self.logger.warning(f"Директория моделей {self.models_dir} не существует")
            return
        
        # Сканируем директорию моделей
        for model_dir in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, model_dir)
            
            # Проверяем, что это директория
            if not os.path.isdir(model_path):
                continue
            
            # Проверяем наличие файлов модели
            model_files = os.listdir(model_path)
            if "config.json" in model_files or "model.bin" in model_files or "pytorch_model.bin" in model_files:
                self.available_models.append(model_dir)
        
        # Логируем результаты
        if self.available_models:
            self.logger.info(f"Найдены локальные модели: {', '.join(self.available_models)}")
        else:
            self.logger.warning("Локальные модели не найдены")
    
    def download_model(self, model_name: str) -> bool:
        """
        Загружает модель из Hugging Face Hub.
        
        Args:
            model_name: Название модели для загрузки
            
        Returns:
            bool: True если загрузка успешна, иначе False
        """
        # Проверяем, что модель поддерживается
        if model_name not in self.supported_models:
            self.logger.error(f"Модель {model_name} не поддерживается")
            return False
        
        # Проверяем, что модель еще не загружена
        if model_name in self.available_models:
            self.logger.info(f"Модель {model_name} уже загружена")
            return True
        
        # Путь для сохранения модели
        model_path = os.path.join(self.models_dir, model_name)
        
        # Загружаем модель с помощью Hugging Face CLI
        try:
            self.logger.info(f"Загрузка модели {model_name}...")
            
            # Используем subprocess для запуска команды загрузки
            # В реальном проекте можно использовать прямые вызовы API Hugging Face
            result = subprocess.run(
                ["huggingface-cli", "download", model_name, "--local-dir", model_path],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Проверяем результат загрузки
            if os.path.exists(model_path):
                self.logger.info(f"Модель {model_name} успешно загружена")
                self.available_models.append(model_name)
                return True
            else:
                self.logger.error(f"Не удалось загрузить модель {model_name}")
                return False
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Ошибка при загрузке модели {model_name}: {e.stderr}")
            return False
        
        except Exception as e:
            self.logger.error(f"Непредвиденная ошибка при загрузке модели {model_name}: {str(e)}")
            return False
    
    def load_model(self, model_name: str) -> bool:
        """
        Загружает модель в память.
        
        Args:
            model_name: Название модели для загрузки
            
        Returns:
            bool: True если загрузка успешна, иначе False
        """
        # Проверяем, что модель доступна
        if model_name not in self.available_models and model_name not in self.supported_models:
            self.logger.error(f"Модель {model_name} не доступна")
            return False
        
        # Если модель не загружена локально, пытаемся загрузить
        if model_name not in self.available_models:
            if not self.download_model(model_name):
                return False
        
        # Если уже загружена эта модель, ничего не делаем
        if self.loaded_model_name == model_name and self.loaded_model is not None:
            self.logger.info(f"Модель {model_name} уже загружена в память")
            return True
        
        # Выгружаем предыдущую модель, если есть
        if self.loaded_model is not None:
            self.unload_model()
        
        # Путь к модели
        model_path = os.path.join(self.models_dir, model_name)
        
        # Загружаем модель
        try:
            self.logger.info(f"Загрузка модели {model_name} в память...")
            
            # В реальном проекте здесь был бы код загрузки модели с использованием
            # соответствующей библиотеки (transformers, llama.cpp и т.д.)
            # Для примера используем заглушку
            
            # Имитация загрузки модели
            self.loaded_model = {"name": model_name, "path": model_path}
            self.loaded_model_name = model_name
            
            self.logger.info(f"Модель {model_name} успешно загружена в память")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели {model_name} в память: {str(e)}")
            self.loaded_model = None
            self.loaded_model_name = None
            return False
    
    def unload_model(self) -> None:
        """
        Выгружает модель из памяти.
        """
        if self.loaded_model is None:
            return
        
        try:
            self.logger.info(f"Выгрузка модели {self.loaded_model_name} из памяти...")
            
            # В реальном проекте здесь был бы код выгрузки модели
            # Для примера просто очищаем переменные
            
            self.loaded_model = None
            self.loaded_model_name = None
            
            # Принудительно вызываем сборщик мусора для освобождения памяти
            import gc
            gc.collect()
            
            self.logger.info("Модель успешно выгружена из памяти")
            
        except Exception as e:
            self.logger.error(f"Ошибка при выгрузке модели из памяти: {str(e)}")
    
    def query_model(self, prompt: str, model_name: Optional[str] = None,
                  max_tokens: int = 500, temperature: float = 0.7,
                  use_cache: bool = True) -> Dict[str, Any]:
        """
        Выполняет запрос к локальной модели.
        
        Args:
            prompt: Промпт для модели
            model_name: Название модели (опционально, если не указано, используется загруженная модель)
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации (от 0 до 1)
            use_cache: Использовать ли кэширование запросов
            
        Returns:
            Dict[str, Any]: Результат запроса
        """
        # Определяем модель для запроса
        target_model = model_name or self.loaded_model_name
        
        # Проверяем, что модель указана
        if target_model is None:
            self.logger.error("Модель не указана и не загружена")
            return {
                "error": "Модель не указана и не загружена",
                "text": "",
                "provider": "local",
                "model": None,
                "tokens": {"prompt": 0, "completion": 0, "total": 0},
                "cost": 0
            }
        
        # Если модель не загружена, загружаем ее
        if self.loaded_model_name != target_model or self.loaded_model is None:
            if not self.load_model(target_model):
                return {
                    "error": f"Не удалось загрузить модель {target_model}",
                    "text": "",
                    "provider": "local",
                    "model": target_model,
                    "tokens": {"prompt": 0, "completion": 0, "total": 0},
                    "cost": 0
                }
        
        # Проверяем кэш, если включено кэширование
        cache_key = None
        if use_cache:
            # Создаем ключ кэша из параметров запроса
            cache_params = {
                "prompt": prompt,
                "model": target_model,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            cache_key = hash(json.dumps(cache_params, sort_keys=True))
            
            # Проверяем кэш
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.logger.info(f"Результат найден в кэше для модели {target_model}")
                return cached_result
        
        # Выполняем запрос к модели
        try:
            self.logger.info(f"Выполнение запроса к локальной модели {target_model}...")
            
            # Засекаем время выполнения
            start_time = time.time()
            
            # В реальном проекте здесь был бы код выполнения запроса к модели
            # Для примера используем заглушку
            
            # Имитация токенизации промпта
            prompt_tokens = len(prompt.split())
            
            # Имитация генерации ответа
            completion = f"Это ответ локальной модели {target_model} на запрос."
            completion_tokens = len(completion.split())
            
            # Оцениваем стоимость запроса
            cost_per_1k = self.local_cost_per_1k_tokens.get(target_model, 0.0005)
            cost = (prompt_tokens + completion_tokens) * cost_per_1k / 1000
            
            # Формируем результат
            result = {
                "text": completion,
                "provider": "local",
                "model": target_model,
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": prompt_tokens + completion_tokens
                },
                "cost": cost,
                "time_taken": time.time() - start_time
            }
            
            # Сохраняем в кэш, если включено кэширование
            if use_cache and cache_key:
                self._save_to_cache(cache_key, result)
            
            self.logger.info(f"Запрос к локальной модели {target_model} выполнен успешно")
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении запроса к модели {target_model}: {str(e)}")
            return {
                "error": f"Ошибка при выполнении запроса: {str(e)}",
                "text": "",
                "provider": "local",
                "model": target_model,
                "tokens": {"prompt": 0, "completion": 0, "total": 0},
                "cost": 0
            }
    
    def _get_from_cache(self, cache_key: int) -> Optional[Dict[str, Any]]:
        """
        Получает результат из кэша.
        
        Args:
            cache_key: Ключ кэша
            
        Returns:
            Optional[Dict[str, Any]]: Результат из кэша или None, если не найден
        """
        # Путь к файлу кэша
        cache_dir = os.path.join(self.models_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        
        # Проверяем наличие файла кэша
        if not os.path.exists(cache_file):
            return None
        
        # Загружаем данные из кэша
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            # Добавляем информацию о том, что результат из кэша
            cache_data["from_cache"] = True
            
            return cache_data
            
        except Exception as e:
            self.logger.error(f"Ошибка при чтении кэша: {str(e)}")
            return None
    
    def _save_to_cache(self, cache_key: int, result: Dict[str, Any]) -> None:
        """
        Сохраняет результат в кэш.
        
        Args:
            cache_key: Ключ кэша
            result: Результат для сохранения
        """
        # Путь к файлу кэша
        cache_dir = os.path.join(self.models_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        
        # Сохраняем данные в кэш
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении в кэш: {str(e)}")
    
    def select_optimal_model(self, prompt: str, required_quality: str = "medium",
                           max_cost: Optional[float] = None) -> str:
        """
        Выбирает оптимальную модель для запроса.
        
        Args:
            prompt: Промпт для модели
            required_quality: Требуемое качество ответа ('low', 'medium', 'high')
            max_cost: Максимальная стоимость запроса в рублях (опционально)
            
        Returns:
            str: Название оптимальной модели
        """
        # Оцениваем сложность запроса
        prompt_complexity = self._estimate_prompt_complexity(prompt)
        
        # Оцениваем примерное количество токенов в промпте
        prompt_tokens = len(prompt.split())
        
        # Определяем оптимальную модель в зависимости от требуемого качества
        if required_quality == "low":
            # Для низкого качества выбираем самые легкие модели
            candidates = [
                "TinyLlama-1.1B-Chat",
                "phi-2",
                "gemma-2b",
                "qwen1.5-0.5b"
            ]
        elif required_quality == "medium":
            # Для среднего качества выбираем средние модели
            candidates = [
                "Mistral-7B-Instruct-v0.2",
                "Llama-2-7b-chat",
                "Llama-3-8b-instruct",
                "gemma-7b",
                "qwen1.5-7b"
            ]
        else:  # high
            # Для высокого качества выбираем мощные модели
            candidates = [
                "Llama-2-13b-chat",
                "mixtral-8x7b",
                "qwen1.5-14b",
                "qwen1.5-32b"
            ]
        
        # Фильтруем по доступным моделям
        available_candidates = [model for model in candidates if model in self.available_models]
        
        # Если нет доступных моделей из кандидатов, выбираем любую доступную
        if not available_candidates:
            if self.available_models:
                self.logger.warning(f"Нет доступных моделей для качества {required_quality}, выбираем из доступных")
                available_candidates = self.available_models
            else:
                self.logger.error(f"Нет доступных локальных моделей")
                # Возвращаем модель по умолчанию, которую будем пытаться загрузить
                return self.supported_models[0]
        
        # Если указано ограничение по стоимости, фильтруем модели
        if max_cost is not None:
            affordable_models = []
            
            for model in available_candidates:
                # Оцениваем стоимость запроса
                cost_per_1k = self.local_cost_per_1k_tokens.get(model, 0.0005)
                estimated_cost = (prompt_tokens + 500) * cost_per_1k / 1000  # +500 токенов на ответ
                
                if estimated_cost <= max_cost:
                    affordable_models.append(model)
            
            if affordable_models:
                available_candidates = affordable_models
            else:
                self.logger.warning(f"Нет моделей, удовлетворяющих ограничению по стоимости {max_cost} руб.")
        
        # Выбираем наиболее подходящую модель из доступных
        if prompt_complexity == "high":
            # Для сложного запроса выбираем самую мощную модель из доступных
            return available_candidates[-1]
        elif prompt_complexity == "medium":
            # Для запроса средней сложности выбираем модель из середины списка
            return available_candidates[len(available_candidates) // 2]
        else:  # low
            # Для простого запроса выбираем самую легкую модель
            return available_candidates[0]
    
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
    
    def get_available_models(self) -> Dict[str, Any]:
        """
        Возвращает информацию о доступных моделях.
        
        Returns:
            Dict[str, Any]: Информация о доступных моделях
        """
        # Обновляем список доступных моделей
        self._scan_available_models()
        
        # Группируем модели по размеру
        small_models = []
        medium_models = []
        large_models = []
        
        for model in self.available_models:
            if model in ["TinyLlama-1.1B-Chat", "phi-2", "gemma-2b", "qwen1.5-0.5b"]:
                small_models.append(model)
            elif model in ["Mistral-7B-Instruct-v0.2", "Llama-2-7b-chat", "Llama-3-8b-instruct", "gemma-7b", "qwen1.5-7b"]:
                medium_models.append(model)
            else:
                large_models.append(model)
        
        # Формируем результат
        return {
            "available_models": self.available_models,
            "loaded_model": self.loaded_model_name,
            "models_by_size": {
                "small": small_models,
                "medium": medium_models,
                "large": large_models
            },
            "device": self.device,
            "models_dir": self.models_dir
        }
