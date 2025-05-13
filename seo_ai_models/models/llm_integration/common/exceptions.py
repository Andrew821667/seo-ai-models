"""
Исключения, используемые в LLM-интеграции.

Модуль содержит определения специальных исключений,
которые могут возникнуть в процессе работы с LLM.
"""


class LLMIntegrationError(Exception):
    """Базовый класс для всех исключений в модуле LLM-интеграции."""
    pass


class ProviderNotSupportedError(LLMIntegrationError):
    """Ошибка, вызываемая при использовании неподдерживаемого LLM-провайдера."""
    
    def __init__(self, provider, supported_providers=None):
        self.provider = provider
        self.supported_providers = supported_providers or []
        message = (f"Провайдер LLM '{provider}' не поддерживается. "
                   f"Поддерживаемые провайдеры: {', '.join(self.supported_providers)}")
        super().__init__(message)


class ModelNotSupportedError(LLMIntegrationError):
    """Ошибка, вызываемая при использовании неподдерживаемой модели LLM."""
    
    def __init__(self, model, provider=None, supported_models=None):
        self.model = model
        self.provider = provider
        self.supported_models = supported_models or []
        
        if provider:
            message = (f"Модель '{model}' не поддерживается провайдером '{provider}'. "
                      f"Поддерживаемые модели: {', '.join(self.supported_models)}")
        else:
            message = f"Модель '{model}' не поддерживается. "
        
        super().__init__(message)


class APIConnectionError(LLMIntegrationError):
    """Ошибка соединения с API LLM-провайдера."""
    
    def __init__(self, provider, details=None):
        self.provider = provider
        self.details = details
        
        message = f"Ошибка соединения с API провайдера '{provider}'."
        if details:
            message += f" Детали: {details}"
        
        super().__init__(message)


class APIResponseError(LLMIntegrationError):
    """Ошибка в ответе API LLM-провайдера."""
    
    def __init__(self, provider, status_code=None, response=None):
        self.provider = provider
        self.status_code = status_code
        self.response = response
        
        message = f"Ошибка в ответе API провайдера '{provider}'."
        if status_code:
            message += f" Код статуса: {status_code}."
        if response:
            message += f" Ответ: {response}"
        
        super().__init__(message)


class TokenLimitExceededError(LLMIntegrationError):
    """Ошибка, возникающая при превышении лимита токенов."""
    
    def __init__(self, max_tokens, actual_tokens):
        self.max_tokens = max_tokens
        self.actual_tokens = actual_tokens
        
        message = (f"Превышен лимит токенов. Максимум: {max_tokens}, "
                  f"фактически: {actual_tokens}.")
        
        super().__init__(message)


class BudgetExceededError(LLMIntegrationError):
    """Ошибка, возникающая при превышении бюджета на использование LLM."""
    
    def __init__(self, budget, estimated_cost):
        self.budget = budget
        self.estimated_cost = estimated_cost
        
        message = (f"Операция превышает установленный бюджет. "
                  f"Бюджет: {budget} ₽, оценочная стоимость: {estimated_cost} ₽.")
        
        super().__init__(message)


class PromptGenerationError(LLMIntegrationError):
    """Ошибка при генерации промпта."""
    
    def __init__(self, template_name=None, details=None):
        self.template_name = template_name
        self.details = details
        
        message = "Ошибка при генерации промпта."
        if template_name:
            message += f" Шаблон: '{template_name}'."
        if details:
            message += f" Детали: {details}"
        
        super().__init__(message)


class ResponseParsingError(LLMIntegrationError):
    """Ошибка при обработке ответа от LLM."""
    
    def __init__(self, details=None, response=None):
        self.details = details
        self.response = response
        
        message = "Ошибка при обработке ответа от LLM."
        if details:
            message += f" Детали: {details}"
        
        super().__init__(message)


class LocalModelError(LLMIntegrationError):
    """Ошибка при работе с локальной моделью LLM."""
    
    def __init__(self, model=None, details=None):
        self.model = model
        self.details = details
        
        message = "Ошибка при работе с локальной моделью LLM."
        if model:
            message += f" Модель: '{model}'."
        if details:
            message += f" Детали: {details}"
        
        super().__init__(message)
