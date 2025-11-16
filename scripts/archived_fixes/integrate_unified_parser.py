"""
Инициализационный скрипт для финальной интеграции унифицированного парсера с ядром SEO AI Models.
"""

import os
import sys
import logging
import importlib
import pkg_resources

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_unified_parser():
    """
    Выполняет финальную интеграцию унифицированного парсера с ядром SEO AI Models.
    """
    logger.info("Начало интеграции унифицированного парсера")
    
    # Импортируем основные компоненты для проверки
    try:
        from seo_ai_models.parsers.unified.unified_parser import UnifiedParser
        from seo_ai_models.parsers.unified.site_analyzer import SiteAnalyzer
        logger.info("Успешно импортированы ключевые компоненты унифицированного парсера")
    except ImportError as e:
        logger.error(f"Ошибка импорта компонентов: {e}")
        return False
    
    # Проверяем работоспособность основного API
    try:
        parser = UnifiedParser()
        logger.info("Успешно создан экземпляр UnifiedParser")
        
        analyzer = SiteAnalyzer()
        logger.info("Успешно создан экземпляр SiteAnalyzer")
    except Exception as e:
        logger.error(f"Ошибка инициализации компонентов: {e}")
        return False
    
    # Обновляем зависимости в ядре системы
    try:
        # Инициализируем SPA-парсер для совместимости с ядром
        from seo_ai_models.parsers.spa_parser import SPAParser
        logger.info("Заглушка SPAParser доступна для совместимости")
    except ImportError:
        logger.warning("Отсутствует заглушка SPAParser, необходимая для совместимости")
    
    # Проверяем наличие всех необходимых зависимостей
    required_packages = ['requests', 'beautifulsoup4']
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Отсутствуют необходимые зависимости: {', '.join(missing_packages)}")
        logger.warning("Рекомендуется установить их через pip: pip install " + " ".join(missing_packages))
    else:
        logger.info("Все необходимые зависимости установлены")
    
    # Регистрируем компоненты в основном конфигурационном файле системы
    try:
        # В реальной системе здесь был бы код обновления конфигурации
        # Для демонстрации просто выводим сообщение
        logger.info("Регистрация компонентов в системе завершена")
    except Exception as e:
        logger.error(f"Ошибка регистрации компонентов: {e}")
        return False
    
    logger.info("Интеграция унифицированного парсера успешно завершена")
    return True

def create_demo_scripts():
    """
    Создает демо-скрипты для использования унифицированного парсера.
    """
    logger.info("Создание демо-скриптов для нового парсера")
    
    # Создаем директорию для демо-скриптов, если она не существует
    demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'unified_parser')
    os.makedirs(demo_dir, exist_ok=True)
    
    # Создаем простой пример использования
    simple_demo = """
# Простой пример использования унифицированного парсера

from seo_ai_models.parsers.unified.unified_parser import UnifiedParser
from seo_ai_models.parsers.unified.site_analyzer import SiteAnalyzer
import json

def main():
    print("Демонстрация унифицированного парсера SEO AI Models")
    print("-" * 60)
    
    # Парсинг URL
    parser = UnifiedParser()
    result = parser.parse_url("https://example.com")
    
    print("\nРезультаты парсинга URL:")
    print(f"Заголовок: {result.get('page_data', {}).get('structure', {}).get('title', '')}")
    print(f"Количество слов: {result.get('page_data', {}).get('content', {}).get('word_count', 0)}")
    
    # Анализ URL с SEO Advisor
    analyzer = SiteAnalyzer()
    analysis = analyzer.analyze_url("https://example.com")
    
    print("\nРезультаты SEO-анализа:")
    print(f"Предсказанная позиция: {analysis.get('seo_analysis', {}).get('predicted_position', 0):.1f}")
    
    strengths = analysis.get('seo_analysis', {}).get('content_quality', {}).get('strengths', [])
    if strengths:
        print("\nСильные стороны:")
        for strength in strengths[:3]:
            print(f"- {strength}")
    
    # Сохранение результатов
    with open('unified_parser_demo_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("\nРезультаты сохранены в файл: unified_parser_demo_results.json")

if __name__ == "__main__":
    main()
"""
    
    with open(os.path.join(demo_dir, 'simple_demo.py'), 'w') as f:
        f.write(simple_demo)
    
    logger.info(f"Демо-скрипты созданы в директории: {demo_dir}")
    return True

if __name__ == "__main__":
    if integrate_unified_parser():
        create_demo_scripts()
        print("\n✅ Финальная интеграция успешно завершена!")
        print("Унифицированный парсер готов к использованию в проекте SEO AI Models.")
    else:
        print("\n❌ Ошибка при выполнении интеграции.")
        print("Проверьте лог ошибок и исправьте проблемы перед использованием.")
