"""
Константы для модуля LLM-интеграции.

Содержит определения моделей LLM, конфигурации запросов
и другие константы, используемые в процессе интеграции с LLM.
"""

# Поддерживаемые LLM провайдеры
LLM_PROVIDERS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gigachat": "GigaChat",  # Российский аналог
    "yandexgpt": "YandexGPT",  # Российский аналог
    "local": "Local LLM",  # Локально размещенные модели
}

# Поддерживаемые модели по провайдерам
LLM_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
    "gigachat": ["gigachat-pro", "gigachat-plus"],
    "yandexgpt": ["yandexgpt", "yandexgpt-mini"],
    "local": ["llama-3-70b", "llama-3-8b", "deepseek-coder", "mistral-7b"]
}

# Дефолтные модели для разных уровней использования
DEFAULT_MODELS = {
    "premium": {
        "provider": "openai",
        "model": "gpt-4o"
    },
    "standard": {
        "provider": "openai",
        "model": "gpt-4o-mini"
    },
    "basic": {
        "provider": "openai",
        "model": "gpt-3.5-turbo"
    },
    "micro": {
        "provider": "local",
        "model": "llama-3-8b"
    }
}

# Параметры запросов по умолчанию
DEFAULT_REQUEST_PARAMS = {
    "openai": {
        "temperature": 0.2,
        "max_tokens": 2048,
        "top_p": 0.95,
    },
    "anthropic": {
        "temperature": 0.2,
        "max_tokens": 2048,
        "top_p": 0.95,
    },
    "gigachat": {
        "temperature": 0.2,
        "max_tokens": 2048,
    },
    "yandexgpt": {
        "temperature": 0.2,
        "max_tokens": 1500,
    },
    "local": {
        "temperature": 0.1,
        "max_tokens": 1024,
        "top_p": 0.9,
    }
}

# Типы анализа контента для LLM
ANALYSIS_TYPES = {
    "compatibility": "Анализ совместимости контента с LLM",
    "citability": "Оценка вероятности цитирования",
    "structure": "Анализ структуры и предложения по улучшению",
    "eeat": "Анализ E-E-A-T для LLM-оптимизации",
    "semantic": "Семантический анализ для LLM"
}

# Ценность токенов (в рублях) для разных провайдеров и моделей
TOKEN_COSTS = {
    "openai": {
        "gpt-4o": {
            "input": 0.10,  # стоимость за 1K токенов ввода
            "output": 0.30  # стоимость за 1K токенов вывода
        },
        "gpt-4o-mini": {
            "input": 0.05,
            "output": 0.15
        },
        "gpt-4-turbo": {
            "input": 0.07,
            "output": 0.21
        },
        "gpt-3.5-turbo": {
            "input": 0.015,
            "output": 0.03
        }
    },
    "anthropic": {
        "claude-3-opus": {
            "input": 0.15,
            "output": 0.45
        },
        "claude-3-sonnet": {
            "input": 0.08,
            "output": 0.24
        },
        "claude-3-haiku": {
            "input": 0.02,
            "output": 0.06
        }
    },
    "gigachat": {
        "gigachat-pro": {
            "input": 0.05,
            "output": 0.12
        },
        "gigachat-plus": {
            "input": 0.03,
            "output": 0.07
        }
    },
    "yandexgpt": {
        "yandexgpt": {
            "input": 0.04,
            "output": 0.10
        },
        "yandexgpt-mini": {
            "input": 0.02,
            "output": 0.05
        }
    },
    "local": {
        "llama-3-70b": {
            "input": 0.01,
            "output": 0.01  # Локальные модели имеют только стоимость вычислений
        },
        "llama-3-8b": {
            "input": 0.005,
            "output": 0.005
        },
        "deepseek-coder": {
            "input": 0.005,
            "output": 0.005
        },
        "mistral-7b": {
            "input": 0.005,
            "output": 0.005
        }
    }
}

# Дефолтные промпты для разных типов анализа
DEFAULT_PROMPTS = {
    "compatibility": """
    Ты эксперт по SEO оптимизации для поисковых систем с интеграцией LLM (как Perplexity, Claude, GPT). 
    Проанализируй представленный контент с точки зрения его совместимости с LLM.
    
    Оцени следующие аспекты по шкале от 1 до 10:
    1. Ясность и структурированность информации
    2. Фактическая точность и актуальность
    3. Полнота раскрытия темы
    4. Наличие ключевой информации в начале материала
    5. Логичность и последовательность изложения
    
    Для каждого аспекта дай рекомендации по улучшению.
    
    Текст для анализа:
    {content}
    """,
    
    "citability": """
    Ты эксперт по оптимизации контента для цитирования в ответах LLM-моделей.
    
    Проанализируй следующий текст и оцени вероятность его цитирования LLM-моделями в ответах на пользовательские запросы по шкале от 1 до 10.
    
    Особое внимание обрати на:
    1. Информативность и полезность контента
    2. Уникальность представленной информации
    3. Авторитетность источника
    4. Наличие цитируемых данных и статистики
    5. Четкость и конкретность формулировок
    
    Дай развернутый анализ с рекомендациями по улучшению цитируемости.
    
    Текст для анализа:
    {content}
    """,
    
    "structure": """
    Ты эксперт по структурированию контента для максимальной эффективности в эру LLM-поисковиков.
    
    Проанализируй структуру представленного текста и предложи конкретные улучшения, которые сделают его более цитируемым и полезным для LLM-моделей.
    
    Обрати внимание на:
    1. Заголовки и подзаголовки
    2. Абзацы и их длину
    3. Маркированные и нумерованные списки
    4. Таблицы и структурированные данные
    5. Ключевые выводы и их размещение
    
    Предложи конкретную, улучшенную структуру контента с примерами.
    
    Текст для анализа:
    {content}
    """,
    
    "eeat": """
    Ты эксперт по E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness) в контексте LLM-оптимизации.
    
    Проанализируй следующий текст на соответствие принципам E-E-A-T, но с учетом специфики LLM-поисковиков.
    
    Для каждого компонента E-E-A-T дай оценку от 1 до 10 и конкретные рекомендации по улучшению:
    1. Experience (Опыт): наличие признаков реального опыта в данной теме
    2. Expertise (Экспертиза): демонстрация экспертных знаний в области
    3. Authoritativeness (Авторитетность): насколько текст воспринимается как авторитетный источник
    4. Trustworthiness (Надежность): надежность, точность и достоверность информации
    
    Для YMYL-тем особенно важно дать рекомендации по усилению надежности и экспертизы.
    
    Текст для анализа:
    {content}
    """,
    
    "semantic": """
    Ты эксперт по семантическому анализу в контексте поисковой оптимизации для LLM.
    
    Проведи глубокий семантический анализ представленного текста:
    1. Выдели основные тематические кластеры
    2. Определи ключевые семантические сущности и их взаимосвязи
    3. Оцени семантическую плотность и релевантность основной теме
    4. Выяви потенциальные семантические пробелы
    5. Предложи дополнительные семантические кластеры для расширения
    
    Дай конкретные рекомендации по улучшению семантического ядра с учетом особенностей LLM-поиска.
    
    Текст для анализа:
    {content}
    """
}
