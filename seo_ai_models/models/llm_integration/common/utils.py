"""
Утилиты для работы с LLM-интеграцией.

Модуль содержит вспомогательные функции и классы,
используемые в процессе интеграции с LLM.
"""

import re
import json
import tiktoken
from typing import Dict, List, Any, Optional, Union, Tuple

# Импортируем константы из того же модуля
from .constants import (
    LLM_PROVIDERS, 
    LLM_MODELS, 
    TOKEN_COSTS,
    DEFAULT_REQUEST_PARAMS
)
from .exceptions import (
    ProviderNotSupportedError,
    ModelNotSupportedError
)


def validate_provider_and_model(provider: str, model: str) -> Tuple[str, str]:
    """
    Проверяет, поддерживаются ли указанные провайдер и модель.
    
    Args:
        provider: Название провайдера LLM
        model: Название модели LLM
        
    Returns:
        Tuple[str, str]: Валидные провайдер и модель
        
    Raises:
        ProviderNotSupportedError: Если провайдер не поддерживается
        ModelNotSupportedError: Если модель не поддерживается для данного провайдера
    """
    # Проверяем провайдера
    if provider not in LLM_PROVIDERS:
        raise ProviderNotSupportedError(provider, list(LLM_PROVIDERS.keys()))
    
    # Проверяем модель
    if model not in LLM_MODELS.get(provider, []):
        raise ModelNotSupportedError(model, provider, LLM_MODELS.get(provider, []))
    
    return provider, model


def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Оценивает количество токенов в тексте для указанной модели.
    
    Args:
        text: Текст для оценки
        model: Модель для которой нужно оценить токены (по умолчанию gpt-3.5-turbo)
        
    Returns:
        int: Примерное количество токенов
    """
    # Для OpenAI моделей используем tiktoken
    if model.startswith(("gpt-", "text-")):
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            # Если не удалось получить кодировку для модели, используем cl100k_base
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
    
    # Для других моделей используем приблизительную оценку
    # (4 символа ~ 1 токен, это грубая оценка)
    return len(text) // 4


def estimate_cost(input_tokens: int, output_tokens: int, 
                 provider: str, model: str) -> float:
    """
    Оценивает стоимость запроса на основе количества токенов.
    
    Args:
        input_tokens: Количество токенов ввода
        output_tokens: Количество токенов вывода
        provider: Название провайдера LLM
        model: Название модели LLM
        
    Returns:
        float: Оценочная стоимость в рублях
    """
    try:
        # Получаем стоимость токенов для указанной модели
        provider_costs = TOKEN_COSTS.get(provider, {})
        model_costs = provider_costs.get(model, {"input": 0, "output": 0})
        
        # Рассчитываем стоимость (переводим в рубли)
        input_cost = (input_tokens / 1000) * model_costs.get("input", 0)
        output_cost = (output_tokens / 1000) * model_costs.get("output", 0)
        
        return input_cost + output_cost
    except Exception:
        # В случае ошибки возвращаем приблизительную оценку
        return (input_tokens + output_tokens) * 0.01 / 1000


def format_prompt(template: str, **kwargs) -> str:
    """
    Форматирует шаблон промпта с переданными параметрами.
    
    Args:
        template: Шаблон промпта
        **kwargs: Параметры для форматирования
        
    Returns:
        str: Отформатированный промпт
    """
    return template.format(**kwargs)


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Пытается извлечь JSON из ответа LLM.
    
    Args:
        response: Текстовый ответ от LLM
        
    Returns:
        Dict[str, Any]: Извлеченный JSON или пустой словарь в случае ошибки
    """
    # Пытаемся найти JSON в ответе с помощью регулярного выражения
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
    
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Если не удалось найти JSON в markdown блоке, 
    # пробуем разобрать весь ответ как JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Если не удалось разобрать JSON, возвращаем пустой словарь
        return {}


def get_default_params(provider: str) -> Dict[str, Any]:
    """
    Возвращает параметры запросов по умолчанию для указанного провайдера.
    
    Args:
        provider: Название провайдера LLM
        
    Returns:
        Dict[str, Any]: Параметры запросов по умолчанию
    """
    return DEFAULT_REQUEST_PARAMS.get(provider, {}).copy()


def extract_scores_from_text(text: str) -> Dict[str, float]:
    """
    Извлекает оценки из текста ответа.
    
    Поиск числовых оценок вида "Название: 8/10" или "Название - 8" и т.п.
    
    Args:
        text: Текст ответа от LLM
        
    Returns:
        Dict[str, float]: Словарь с извлеченными оценками
    """
    # Паттерн для поиска оценок вида "Название: 8/10" или "Название - 8"
    score_pattern = r'([A-Za-zА-Яа-я\s]+)[:\-]\s*(\d+(?:\.\d+)?)(?:/\d+)?'
    
    scores = {}
    for match in re.finditer(score_pattern, text):
        category = match.group(1).strip()
        score = float(match.group(2))
        scores[category] = score
    
    return scores


def chunk_text(text: str, max_chunk_size: int = 4000, 
              overlap: int = 200) -> List[str]:
    """
    Разбивает текст на перекрывающиеся чанки заданного размера.
    
    Args:
        text: Исходный текст
        max_chunk_size: Максимальный размер чанка в символах
        overlap: Размер перекрытия между соседними чанками
        
    Returns:
        List[str]: Список чанков текста
    """
    chunks = []
    text_length = len(text)
    
    # Если текст короче max_chunk_size, возвращаем его как один чанк
    if text_length <= max_chunk_size:
        return [text]
    
    # Иначе разбиваем на чанки с перекрытием
    start = 0
    while start < text_length:
        end = min(start + max_chunk_size, text_length)
        
        # Если это не последний чанк, ищем ближайший конец предложения
        if end < text_length:
            # Ищем ближайшую точку, вопросительный или восклицательный знак с пробелом после
            sentence_end = max(
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end)
            )
            
            # Если нашли конец предложения, используем его как границу чанка
            if sentence_end != -1:
                end = sentence_end + 2  # +2 чтобы включить точку и пробел
        
        # Добавляем чанк
        chunks.append(text[start:end])
        
        # Обновляем начальную позицию с учетом перекрытия
        start = end - overlap
        
        # Если после смещения мы оказались в середине слова, 
        # сдвигаемся до начала следующего слова
        if start > 0 and start < text_length and not text[start].isspace():
            # Ищем следующий пробел
            next_space = text.find(" ", start)
            if next_space != -1:
                start = next_space + 1
    
    return chunks
