"""
Интеграция всех улучшений унифицированного парсера в проект SEO AI Models.
"""

import os
import sys
import logging
import subprocess
import importlib
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """
    Проверяет наличие необходимых зависимостей.
    """
    required_packages = [
        "requests",
        "beautifulsoup4",
        "lxml",
        "concurrent-futures"
    ]
    
    optional_packages = [
        "playwright",
        "spacy",
        "nltk",
        "gensim"
    ]
    
    # Проверяем обязательные зависимости
    missing_required = []
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
        except ImportError:
            missing_required.append(package)
            
    # Проверяем опциональные зависимости
    missing_optional = []
    for package in optional_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_optional.append(package)
    
    return missing_required, missing_optional

def install_dependencies(packages):
    """
    Устанавливает отсутствующие пакеты.
    """
    if not packages:
        return True
        
    logger.info(f"Установка пакетов: {', '.join(packages)}")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка установки пакетов: {e}")
        return False

def update_init_files():
    """
    Обновляет файлы __init__.py для правильной инициализации модулей.
    """
    # Корневой модуль парсеров
    parsers_init = """
Модуль парсеров для проекта SEO AI Models.
Предоставляет компоненты для парсинга и анализа сайтов.

from seo_ai_models.parsers.unified.unified_parser import UnifiedParser
from seo_ai_models.parsers.unified.site_analyzer import SiteAnalyzer

# Экспортируем основные классы
__all__ = ['UnifiedParser', 'SiteAnalyzer']
"""
    
    parsers_init_path = Path("seo_ai_models/parsers/__init__.py")
    with open(parsers_init_path, 'w') as f:
        f.write(parsers_init)
        
    # Проверяем наличие всех необходимых __init__.py
    subdirs = [
        'seo_ai_models/parsers/unified/crawlers',
        'seo_ai_models/parsers/unified/extractors',
        'seo_ai_models/parsers/unified/analyzers',
        'seo_ai_models/parsers/unified/utils'
    ]
    
    for subdir in subdirs:
        init_path = Path(subdir) / "__init__.py"
        if not init_path.exists():
            with open(init_path, 'w') as f:
                f.write("# Инициализация модуля\n")
                
    logger.info("Файлы __init__.py обновлены")
    return True

def create_demo_script():
    """
    Создает демонстрационный скрипт для унифицированного парсера.
    """
    demo_dir = Path("examples/unified_parser")
    demo_dir.mkdir(parents=True, exist_ok=True)

    demo_script = """
Демонстрация возможностей унифицированного парсера SEO AI Models.

import argparse
import json
import time
from pprint import pprint

from seo_ai_models.parsers.unified.unified_parser import UnifiedParser
from seo_ai_models.parsers.unified.site_analyzer import SiteAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Демонстрация унифицированного парсера")
    parser.add_argument("url", help="URL для анализа")
    parser.add_argument("--keywords", help="Ключевые слова через запятую")
    parser.add_argument("--spa", action="store_true", help="Использовать режим SPA")
    parser.add_argument("--output", help="Файл для сохранения результатов")
    
    args = parser.parse_args()
    
    print(f"Демонстрация унифицированного парсера SEO AI Models")
    print("-" * 60)
    
    # Создаем экземпляр парсера
    unified_parser = UnifiedParser(
        force_spa_mode=args.spa,
        auto_detect_spa=not args.spa,
        extract_semantic=True
    )
    
    print(f"Анализ URL: {args.url}")
    start_time = time.time()
    
    # Парсим URL
    result = unified_parser.parse_url(args.url)
    
    if result.get("success", False):
        page_data = result.get("page_data", {})
        
        print(f"\nОсновная информация:")
        print(f"Заголовок: {page_data.get('structure', {}).get('title', '')}")
        print(f"Количество слов: {page_data.get('content', {}).get('word_count', 0)}")
        print(f"Мета-описание: {page_data.get('metadata', {}).get('description', '')}")
        
        # Получаем ключевые слова
        keywords = args.keywords.split(",") if args.keywords else list(
            page_data.get("content", {}).get("keywords", {}).keys()
        )[:5]
        
        print(f"\nИспользуемые ключевые слова: {', '.join(keywords)}")
        
        # Создаем анализатор сайтов
        site_analyzer = SiteAnalyzer()
        
        # Анализируем URL
        seo_analysis = site_analyzer.analyze_url(args.url, target_keywords=keywords)
        
        if seo_analysis.get("success", False):
            analysis = seo_analysis.get("seo_analysis", {})
            
            print(f"\nРезультаты SEO анализа:")
            print(f"Предсказанная позиция: {analysis.get('predicted_position', 0):.1f}")
            
            # Выводим сильные стороны
            strengths = analysis.get("content_quality", {}).get("strengths", [])
            if strengths:
                print("\nСильные стороны:")
                for strength in strengths[:3]:
                    print(f"- {strength}")
            
            # Выводим слабые стороны
            weaknesses = analysis.get("content_quality", {}).get("weaknesses", [])
            if weaknesses:
                print("\nСлабые стороны:")
                for weakness in weaknesses[:3]:
                    print(f"- {weakness}")
            
            # Выводим рекомендации
            recommendations = analysis.get("recommendations", {})
            if recommendations:
                print("\nОсновные рекомендации:")
                for category, items in recommendations.items():
                    print(f"\n{category.upper()}:")
                    for item in items[:2]:  # Выводим только первые 2 рекомендации в каждой категории
                        print(f"- {item}")
        else:
            print(f"\nОшибка при выполнении SEO анализа: {seo_analysis.get('error', 'Неизвестная ошибка')}")
        
        # Сохраняем результаты, если указан выходной файл
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nРезультаты сохранены в: {args.output}")
    else:
        print(f"Ошибка при парсинге URL: {result.get('error', 'Неизвестная ошибка')}")
    
    execution_time = time.time() - start_time
    print(f"\nВремя выполнения: {execution_time:.2f} секунд")

if __name__ == "__main__":
    main()
"""
    
    demo_path = demo_dir / "demo.py"
    with open(demo_path, 'w') as f:
        f.write(demo_script)
        
    logger.info(f"Демонстрационный скрипт создан: {demo_path}")
    return True

def main():
    logger.info("Интеграция улучшений унифицированного парсера")
    
    # Проверяем зависимости
    missing_required, missing_optional = check_dependencies()
    
    if missing_required:
        logger.warning(f"Обнаружены отсутствующие обязательные зависимости: {', '.join(missing_required)}")
        logger.info("Установка обязательных зависимостей...")
        if not install_dependencies(missing_required):
            logger.error("Не удалось установить обязательные зависимости")
            return False
    
    if missing_optional:
        logger.info(f"Обнаружены отсутствующие опциональные зависимости: {', '.join(missing_optional)}")
        install = input("Установить опциональные зависимости? (y/n): ").lower() == 'y'
        if install and not install_dependencies(missing_optional):
            logger.warning("Не удалось установить опциональные зависимости")
    
    # Обновляем файлы инициализации
    if not update_init_files():
        logger.error("Не удалось обновить файлы инициализации")
        return False
    
    # Создаем демонстрационный скрипт
    if not create_demo_script():
        logger.warning("Не удалось создать демонстрационный скрипт")
    
    # Выполняем базовую проверку модуля
    try:
        from seo_ai_models.parsers.unified.unified_parser import UnifiedParser
        parser = UnifiedParser()
        logger.info("Базовая проверка унифицированного парсера успешна")
    except Exception as e:
        logger.error(f"Ошибка при проверке унифицированного парсера: {e}")
        return False
    
    logger.info("Интеграция улучшений унифицированного парсера завершена успешно")
    print("\n" + "=" * 60)
    print("ИНТЕГРАЦИЯ УЛУЧШЕНИЙ ЗАВЕРШЕНА УСПЕШНО")
    print("=" * 60)
    print("\nТеперь вы можете использовать унифицированный парсер с расширенными возможностями.")
    print("Для демонстрации запустите скрипт: python examples/unified_parser/demo.py URL")
    print("Для тестирования расширенных возможностей: python test_enhanced_parser.py")
    
    return True

if __name__ == "__main__":
    main()
