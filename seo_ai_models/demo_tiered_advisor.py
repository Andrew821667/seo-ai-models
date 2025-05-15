"""
Демонстрационный скрипт для многоуровневой системы TieredAdvisor.

Скрипт демонстрирует использование TieredAdvisor с различными планами
и показывает разницу в функциональности между ними.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any

# Добавляем родительскую директорию в Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем компоненты многоуровневой системы
from seo_ai_models.models.tiered_system.core.tiered_advisor import TieredAdvisor, TierPlan


# Настраиваем логгирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def parse_arguments():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description='Демонстрация TieredAdvisor')
    
    parser.add_argument(
        '--tier',
        type=str,
        choices=['micro', 'basic', 'professional', 'enterprise'],
        default='basic',
        help='План использования (micro, basic, professional, enterprise)'
    )
    
    parser.add_argument(
        '--content',
        type=str,
        default='test_content.txt',
        help='Путь к файлу с контентом для анализа'
    )
    
    parser.add_argument(
        '--keywords',
        type=str,
        help='Ключевые слова для анализа (через запятую)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='API ключ для LLM-сервиса'
    )
    
    parser.add_argument(
        '--beta',
        action='store_true',
        help='Активировать бета-функции'
    )
    
    return parser.parse_args()


def read_content(file_path):
    """Читает содержимое файла."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Ошибка при чтении файла: {e}")
        # Возвращаем тестовый контент в случае ошибки
        return """
        # Заголовок тестовой статьи
        
        Это тестовая статья для демонстрации работы TieredAdvisor.
        Статья содержит несколько абзацев текста с ключевыми словами.
        
        ## Подзаголовок первый
        
        В этом разделе мы рассмотрим основные аспекты SEO оптимизации.
        Важно учитывать все факторы ранжирования и особенности алгоритмов.
        
        ## Подзаголовок второй
        
        Здесь мы обсудим особенности оптимизации для разных типов сайтов.
        Электронная коммерция, блоги и корпоративные сайты требуют разных подходов.
        
        ### Заключение
        
        В заключение стоит отметить важность комплексного подхода к SEO.
        Только сочетание технической оптимизации, качественного контента и
        правильного построения ссылочной массы даст хороший результат.
        """


def main():
    """Основная функция демонстрации."""
    args = parse_arguments()
    
    # Подготавливаем контент и ключевые слова
    content = read_content(args.content)
    keywords = args.keywords.split(',') if args.keywords else ['seo', 'optimization']
    
    # Подготавливаем API ключи
    api_keys = {}
    if args.api_key:
        api_keys = {'openai': args.api_key}
    
    logger.info(f"Инициализация TieredAdvisor с планом: {args.tier}")
    
    # Создаем экземпляр TieredAdvisor
    advisor = TieredAdvisor(
        tier=args.tier,
        config={
            'max_parallel_processes': 4,
            'cache_ttl': 3600,
            'use_cache': True,
        },
        api_keys=api_keys,
        is_beta_tester=args.beta
    )
    
    # Получаем информацию о плане
    tier_info = advisor.get_tier_info()
    logger.info(f"Информация о плане: {json.dumps(tier_info, indent=2)}")
    
    # Анализируем контент
    logger.info("Анализ контента...")
    results = advisor.analyze_content(
        content=content,
        keywords=keywords,
        context={
            'url': 'https://example.com/test-article',
            'industry': 'technology',
        },
        analyze_llm_compatibility=True,  # Эта функция может быть недоступна для некоторых планов
        score_citability=True,          # Эта функция может быть недоступна для некоторых планов
    )
    
    # Выводим результаты
    logger.info("Результаты анализа:")
    
    # Ключи результатов, которые будут в выводе
    result_keys = [
        'basic_metrics',
        'readability',
        'keywords_basic',
        'structure_basic',
        'llm_compatibility',
        'citability_score',
    ]
    
    # Выводим только доступные результаты
    for key in result_keys:
        if key in results:
            value = results[key]
            if isinstance(value, dict):
                logger.info(f"{key}: {json.dumps(value, indent=2)}")
            else:
                logger.info(f"{key}: {value}")
    
    # Демонстрация обновления плана
    if args.tier != 'enterprise':
        new_tier = 'professional' if args.tier != 'professional' else 'enterprise'
        logger.info(f"Обновление плана до {new_tier}...")
        advisor.upgrade_tier(new_tier)
        
        # Получаем обновленную информацию о плане
        updated_tier_info = advisor.get_tier_info()
        logger.info(f"Обновленная информация о плане: {json.dumps(updated_tier_info, indent=2)}")


if __name__ == "__main__":
    main()
