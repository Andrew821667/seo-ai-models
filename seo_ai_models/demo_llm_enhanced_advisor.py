"""
Демонстрационный скрипт для LLM-Enhanced SEO Advisor.

Скрипт демонстрирует использование расширенной версии SEO Advisor
с интегрированными LLM-компонентами.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any

# Добавляем родительскую директорию в Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем расширенный SEO Advisor
from seo_ai_models.models.seo_advisor.llm_integration_adapter import LLMEnhancedSEOAdvisor


# Настраиваем логгирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def analyze_content(api_key: str, content: str, use_llm: bool = True, 
                  budget: float = None, output_file: str = None) -> None:
    """
    Анализирует контент с помощью LLM-Enhanced SEO Advisor.
    
    Args:
        api_key: API ключ для LLM-провайдера
        content: Текст для анализа
        use_llm: Использовать LLM-компоненты
        budget: Бюджет для LLM-анализа в рублях
        output_file: Путь к файлу для сохранения результатов (опционально)
    """
    # Создаем экземпляр LLM-Enhanced SEO Advisor
    advisor = LLMEnhancedSEOAdvisor(api_key=api_key)
    
    # Анализируем контент
    result = advisor.analyze_content(content, use_llm=use_llm, llm_budget=budget)
    
    # Выводим основные результаты
    logger.info("=== Результаты анализа ===")
    logger.info(f"Общая оценка: {result.get('overall_score', 0):.2f}/10")
    
    # Выводим базовые метрики
    logger.info("\n=== Базовые метрики ===")
    for metric, value in result.get('metrics', {}).items():
        logger.info(f"{metric}: {value}")
    
    # Если использовались LLM-компоненты, выводим их метрики
    if use_llm and 'llm_analysis' in result and 'llm_metrics' in result['llm_analysis']:
        logger.info("\n=== LLM-метрики ===")
        for metric, value in result['llm_analysis']['llm_metrics'].items():
            logger.info(f"{metric}: {value}")
    
    # Выводим топ-5 предложений по улучшению
    logger.info("\n=== Топ-5 предложений по улучшению ===")
    suggestions = sorted(
        result.get('suggestions', []),
        key=lambda x: x.get('importance', 0),
        reverse=True
    )[:5]
    
    for i, suggestion in enumerate(suggestions):
        logger.info(f"{i+1}. [{suggestion.get('type')}] {suggestion.get('description')}")
    
    # Если указан файл для сохранения, сохраняем в него результаты
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"\nРезультаты сохранены в файл: {output_file}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов: {e}")


def enhance_content(api_key: str, content: str, enhancement_type: str = "structure",
                  budget: float = None, output_file: str = None) -> None:
    """
    Улучшает контент с помощью LLM-Enhanced SEO Advisor.
    
    Args:
        api_key: API ключ для LLM-провайдера
        content: Текст для улучшения
        enhancement_type: Тип улучшения
        budget: Бюджет для LLM-улучшения в рублях
        output_file: Путь к файлу для сохранения результатов (опционально)
    """
    # Создаем экземпляр LLM-Enhanced SEO Advisor
    advisor = LLMEnhancedSEOAdvisor(api_key=api_key)
    
    # Улучшаем контент
    result = advisor.enhance_content(content, enhancement_type=enhancement_type, llm_budget=budget)
    
    # Проверяем успешность операции
    if result.get('success', False):
        logger.info("=== Результат улучшения контента ===")
        logger.info(f"Тип улучшения: {enhancement_type}")
        
        # Выводим краткую статистику изменений
        if 'changes' in result:
            changes = result['changes']
            logger.info("\n=== Статистика изменений ===")
            
            if 'headings' in changes:
                headings = changes['headings']
                logger.info(f"Заголовки: {headings.get('original_count', 0)} -> {headings.get('enhanced_count', 0)}")
            
            if 'paragraphs' in changes:
                paragraphs = changes['paragraphs']
                logger.info(f"Абзацы: {paragraphs.get('original_count', 0)} -> {paragraphs.get('enhanced_count', 0)}")
            
            if 'lists' in changes:
                lists = changes['lists']
                logger.info(f"Списки: {lists.get('original_count', 0)} -> {lists.get('enhanced_count', 0)}")
        
        # Выводим улучшенный контент
        logger.info("\n=== Улучшенный контент ===")
        print(result.get('enhanced_content', ''))
        
        # Если указан файл для сохранения, сохраняем в него результаты
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result.get('enhanced_content', ''))
                logger.info(f"\nУлучшенный контент сохранен в файл: {output_file}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении результатов: {e}")
    else:
        # Выводим информацию об ошибке
        logger.error(f"Ошибка при улучшении контента: {result.get('error', 'Неизвестная ошибка')}")


def main() -> None:
    """
    Основная функция приложения.
    """
    parser = argparse.ArgumentParser(description="Демонстрация LLM-Enhanced SEO Advisor")
    parser.add_argument("--api-key", help="API ключ OpenAI")
    parser.add_argument("--action", choices=["analyze", "enhance"], default="analyze",
                       help="Действие (analyze или enhance)")
    parser.add_argument("--type", default="structure",
                       help="Тип улучшения (для action=enhance)")
    parser.add_argument("--file", help="Путь к файлу с контентом")
    parser.add_argument("--output", help="Путь к файлу для сохранения результатов")
    parser.add_argument("--budget", type=float, help="Бюджет для LLM-операций в рублях")
    parser.add_argument("--no-llm", action="store_true", 
                       help="Отключить использование LLM-компонентов")
    
    args = parser.parse_args()
    
    # Получаем API ключ из аргументов или переменной окружения
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("API ключ OpenAI не указан. Укажите его через --api-key или переменную окружения OPENAI_API_KEY")
        return
    
    # Загружаем контент
    content = ""
    
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Ошибка при чтении файла: {e}")
            return
    else:
        print("Введите контент (завершите ввод комбинацией Ctrl+D в Unix или Ctrl+Z в Windows):")
        try:
            while True:
                line = input()
                content += line + "\n"
        except EOFError:
            pass
    
    # Выполняем действие
    if args.action == "analyze":
        analyze_content(
            api_key=api_key,
            content=content,
            use_llm=not args.no_llm,
            budget=args.budget,
            output_file=args.output
        )
    elif args.action == "enhance":
        enhance_content(
            api_key=api_key,
            content=content,
            enhancement_type=args.type,
            budget=args.budget,
            output_file=args.output
        )


if __name__ == "__main__":
    main()
