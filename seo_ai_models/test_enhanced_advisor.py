"""
Тестовый скрипт для EnhancedSEOAdvisor.

Скрипт демонстрирует использование EnhancedSEOAdvisor с LLM-компонентами.
"""

import os
import sys
import logging
from typing import Dict, Any

# Настраиваем логгирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# Добавляем родительскую директорию в Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Импортируем EnhancedSEOAdvisor
from seo_ai_models.models.seo_advisor.enhanced_advisor import EnhancedSEOAdvisor


def test_enhanced_advisor():
    """
    Тестирует EnhancedSEOAdvisor с LLM-компонентами.
    """
    # Получаем API ключ из переменной окружения
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        logger.error(
            "API ключ OpenAI не найден. Укажите его через переменную окружения OPENAI_API_KEY"
        )
        return

    # Создаем EnhancedSEOAdvisor
    advisor = EnhancedSEOAdvisor(llm_api_key=api_key)

    logger.info("EnhancedSEOAdvisor создан успешно")
    logger.info(f"LLM-функции {'доступны' if advisor.llm_enabled else 'недоступны'}")

    # Загружаем тестовый контент
    try:
        with open("test_content.txt", "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Ошибка при чтении файла: {e}")
        content = """
        # Преимущества использования искусственного интеллекта в бизнесе

        Искусственный интеллект (ИИ) становится все более важным инструментом для бизнеса. 
        Компании, которые эффективно внедряют ИИ, получают значительные конкурентные преимущества.

        ## Автоматизация рутинных процессов

        ИИ позволяет автоматизировать многие рутинные задачи, освобождая время сотрудников 
        для более творческой работы. Например, чат-боты могут отвечать на стандартные вопросы клиентов, 
        а системы машинного обучения могут сортировать и классифицировать документы.

        ## Аналитика данных и прогнозирование

        Алгоритмы машинного обучения способны анализировать огромные объемы данных 
        и находить скрытые закономерности, которые человек мог бы пропустить. 
        Это позволяет делать более точные прогнозы о поведении рынка, 
        клиентов и оптимизировать бизнес-процессы.

        ИИ помогает создавать персонализированный опыт для каждого клиента 
        на основе его предпочтений и истории взаимодействия с компанией. 
        Это повышает удовлетворенность клиентов и способствует их удержанию.
        """

    logger.info(f"Тестовый контент загружен, длина: {len(content)} символов")

    # Анализируем контент с использованием LLM-компонентов
    logger.info("Выполняем анализ с использованием LLM-компонентов...")
    result = advisor.analyze_content(content, use_llm=True)

    # Выводим основные результаты анализа
    logger.info("\n=== Результаты анализа ===")
    logger.info(f"Общая оценка: {result.get('overall_score', 0):.2f}/10")

    # Выводим базовые метрики
    logger.info("\n=== Базовые метрики ===")
    for metric, value in result.get("metrics", {}).items():
        logger.info(f"{metric}: {value}")

    # Выводим LLM-метрики
    if "llm_metrics" in result:
        logger.info("\n=== LLM-метрики ===")
        for metric, value in result.get("llm_metrics", {}).items():
            logger.info(f"{metric}: {value}")

    # Выводим топ-5 предложений по улучшению
    logger.info("\n=== Топ-5 предложений по улучшению ===")
    for i, suggestion in enumerate(result.get("suggestions", [])[:5]):
        logger.info(f"{i+1}. [{suggestion.get('type', '')}] {suggestion.get('description', '')}")

    # Улучшаем структуру контента
    logger.info("\n=== Улучшаем структуру контента ===")
    enhancement_result = advisor.enhance_content(content, enhancement_type="structure")

    if enhancement_result.get("success", False):
        logger.info("Контент успешно улучшен")

        # Выводим статистику изменений
        changes = enhancement_result.get("changes", {})
        logger.info("\n=== Статистика изменений ===")

        if "headings" in changes:
            headings = changes["headings"]
            logger.info(
                f"Заголовки: {headings.get('original_count', 0)} -> {headings.get('enhanced_count', 0)}"
            )

        if "paragraphs" in changes:
            paragraphs = changes["paragraphs"]
            logger.info(
                f"Абзацы: {paragraphs.get('original_count', 0)} -> {paragraphs.get('enhanced_count', 0)}"
            )

        if "lists" in changes:
            lists = changes["lists"]
            logger.info(
                f"Списки: {lists.get('original_count', 0)} -> {lists.get('enhanced_count', 0)}"
            )

        # Выводим улучшенный контент
        logger.info("\n=== Улучшенный контент ===")
        print(enhancement_result.get("enhanced_content", ""))
    else:
        logger.error(
            f"Ошибка при улучшении контента: {enhancement_result.get('error', 'Неизвестная ошибка')}"
        )

    # Выполняем специальный анализ для LLM
    logger.info("\n=== Выполняем специальный анализ для LLM ===")
    llm_result = advisor.analyze_content_for_llm(content, llm_type="search")

    # Выводим результаты специального анализа
    logger.info(f"Общая оценка для LLM: {llm_result.get('overall_score', 0):.2f}/10")
    logger.info(f"Стоимость анализа: {llm_result.get('total_cost', 0):.2f} руб.")

    # Выводим оценки компонентов
    if "compatibility" in llm_result:
        compat_score = llm_result["compatibility"].get("compatibility_scores", {}).get("overall", 0)
        logger.info(f"Оценка совместимости с LLM: {compat_score:.2f}/10")

    if "citability" in llm_result:
        cit_score = llm_result["citability"].get("citability_score", 0)
        logger.info(f"Оценка цитируемости: {cit_score:.2f}/10")

    if "eeat" in llm_result:
        eeat_score = llm_result["eeat"].get("eeat_scores", {}).get("overall", 0)
        logger.info(f"Оценка E-E-A-T для LLM: {eeat_score:.2f}/10")

    # Выводим топ-5 предложений по улучшению для LLM
    logger.info("\n=== Топ-5 предложений по улучшению для LLM ===")
    for i, suggestion in enumerate(llm_result.get("suggestions", [])[:5]):
        logger.info(f"{i+1}. [{suggestion.get('type', '')}] {suggestion.get('description', '')}")


if __name__ == "__main__":
    test_enhanced_advisor()
