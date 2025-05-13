"""
Упрощенный тестовый скрипт для проверки базовой интеграции.
"""

import os
import sys
import logging
from typing import Dict, Any

# Настраиваем логгирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Добавляем родительскую директорию в Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_llm_service_integration():
    """
    Тестирует базовую интеграцию с LLM-сервисом.
    """
    try:
        from seo_ai_models.models.llm_integration.service.llm_service import LLMService
        from seo_ai_models.models.llm_integration.service.prompt_generator import PromptGenerator
        
        logger.info("Импорт сервисов выполнен успешно.")
        
        # Получаем API ключ из переменной окружения
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            logger.error("API ключ OpenAI не найден. Укажите его через переменную окружения OPENAI_API_KEY")
            return "Тест не пройден: отсутствует API ключ"
        
        # Создаем базовые сервисы
        llm_service = LLMService()
        llm_service.add_provider("openai", api_key=api_key, model="gpt-4o-mini")
        prompt_generator = PromptGenerator()
        
        logger.info("Базовые сервисы созданы успешно.")
        
        # Проверяем работу сервисов
        prompt = "Опиши преимущества искусственного интеллекта в пяти предложениях."
        result = llm_service.generate(prompt, "openai")
        
        logger.info("Запрос к LLM-сервису выполнен успешно.")
        logger.info(f"Использовано токенов: {result.get('tokens', {}).get('total', 0)}")
        logger.info(f"Стоимость запроса: {result.get('cost', 0)} руб.")
        
        # Выводим часть ответа
        text = result.get("text", "")
        logger.info(f"Начало ответа: {text[:100]}...")
        
        return "Тест пройден успешно."
    
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Тест не пройден: {e}"

def test_llm_compatible_imports():
    """
    Тестирует импорт классов интеграции.
    """
    try:
        # Пробуем импортировать классы интеграции
        from seo_ai_models.models.seo_advisor.llm_integration_adapter import LLMAdvisorFactory, LLMEnhancedSEOAdvisor
        
        logger.info("Классы интеграции импортированы успешно.")
        
        # Создаем фабрику без API ключа
        factory = LLMAdvisorFactory()
        
        logger.info("Фабрика создана успешно.")
        
        return "Тест импорта пройден успешно."
    
    except Exception as e:
        logger.error(f"Ошибка при импорте: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Тест импорта не пройден: {e}"


if __name__ == "__main__":
    logger.info("=== Тест интеграции с LLM-сервисом ===")
    result1 = test_llm_service_integration()
    logger.info(f"Результат: {result1}")
    
    logger.info("\n=== Тест импорта классов интеграции ===")
    result2 = test_llm_compatible_imports()
    logger.info(f"Результат: {result2}")
