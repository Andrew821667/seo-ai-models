
"""
Сравнительное тестирование улучшенных JavaScript-компонентов на разных сайтах.
"""

import sys
import os
import json
import logging
import time
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("comparative_test")

# Добавляем директорию проекта в путь импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем компоненты
from seo_ai_models.parsers.unified.js_integrator import JSIntegrator

# Создаем директорию для результатов
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results")
os.makedirs(results_dir, exist_ok=True)

# Список тестовых сайтов
TEST_SITES = [
    {
        "name": "React Router",
        "url": "https://reactrouter.com/en/main/start/tutorial",
        "expected": ["client_routing"]
    },
    {
        "name": "GitHub",
        "url": "https://github.com/explore",
        "expected": ["graphql"]
    },
    {
        "name": "Vue.js",
        "url": "https://vuejs.org/guide/introduction.html",
        "expected": ["client_routing"]
    },
    {
        "name": "Facebook",
        "url": "https://www.facebook.com/",
        "expected": ["websocket", "graphql"]
    },
    {
        "name": "Twitter/X",
        "url": "https://twitter.com/explore",
        "expected": ["websocket", "client_routing"]
    }
]

def test_site(site_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Тестирует сайт с помощью интегратора.
    
    Args:
        site_info: Информация о сайте
        
    Returns:
        Dict[str, Any]: Результаты тестирования
    """
    name = site_info["name"]
    url = site_info["url"]
    expected = site_info["expected"]
    
    logger.info(f"Тестирование {name} ({url})...")
    
    # Создание интегратора
    integrator = JSIntegrator(
        enable_websocket=True,
        enable_graphql=True,
        enable_client_routing=True,
        emulate_user_behavior=True,
        bypass_protection=True
    )
    
    # Дополнительные опции для краулера
    crawler_options = {
        "max_pages": 1,     # Для быстрого тестирования сканируем только главную страницу
        "max_depth": 0,     # Не сканируем внутренние страницы
        "delay": 0.5,       # Уменьшаем задержку для быстрого тестирования
        "headless": True,
        "wait_for_timeout": 10000  # 10 секунд на загрузку
    }
    
    try:
        # Парсинг сайта
        start_time = time.time()
        result = integrator.parse_site(url, **crawler_options)
        duration = time.time() - start_time
        
        # Получение объединенных результатов
        combined_results = integrator.get_combined_results(result)
        
        # Файл для сохранения результатов
        output_file = os.path.join(results_dir, f"{name.lower().replace(' ', '_')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Построение итогового результата
        test_result = {
            "name": name,
            "url": url,
            "expected": expected,
            "duration": duration,
            "detected": {
                "websocket": combined_results["summary"]["websocket_detected"],
                "graphql": combined_results["summary"]["graphql_detected"],
                "client_routing": combined_results["summary"]["client_routing_detected"]
            },
            "technologies": combined_results.get("technologies", []),
            "success": True,
            "output_file": output_file
        }
        
        # Проверка ожидаемых результатов
        matches = []
        for tech in expected:
            if test_result["detected"].get(tech, False):
                matches.append(tech)
        
        test_result["matches"] = matches
        test_result["match_ratio"] = len(matches) / len(expected) if expected else 1.0
        
        logger.info(f"Тестирование {name} завершено за {duration:.2f}с")
        return test_result
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании {name}: {str(e)}")
        return {
            "name": name,
            "url": url,
            "expected": expected,
            "success": False,
            "error": str(e)
        }

def main():
    """Основная функция сравнительного тестирования"""
    logger.info(f"Запуск сравнительного тестирования на {len(TEST_SITES)} сайтах")
    
    results = []
    
    # Последовательное тестирование сайтов
    for site in TEST_SITES:
        result = test_site(site)
        results.append(result)
    
    # Сводка результатов
    logger.info("\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
    
    success_count = sum(1 for r in results if r.get("success", False))
    logger.info(f"Успешно протестировано: {success_count}/{len(results)} сайтов")
    
    match_ratio_sum = sum(r.get("match_ratio", 0) for r in results if r.get("success", False))
    average_match_ratio = match_ratio_sum / success_count if success_count > 0 else 0
    logger.info(f"Средняя точность обнаружения: {average_match_ratio:.2%}")
    
    # Детальные результаты
    logger.info("\n=== ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ===")
    for result in results:
        name = result["name"]
        if result.get("success", False):
            matches = result.get("matches", [])
            expected = result.get("expected", [])
            detected = []
            
            for tech, is_detected in result.get("detected", {}).items():
                if is_detected:
                    detected.append(tech)
            
            logger.info(f"{name}:")
            logger.info(f"  Ожидалось: {', '.join(expected)}")
            logger.info(f"  Обнаружено: {', '.join(detected)}")
            logger.info(f"  Совпадения: {', '.join(matches)} ({result.get('match_ratio', 0):.2%})")
            
            # Вывод обнаруженных технологий
            techs = result.get("technologies", [])
            if techs:
                logger.info("  Технологии:")
                for tech in techs:
                    logger.info(f"    • {tech.get('name')} ({tech.get('type')})")
        else:
            logger.info(f"{name}: ОШИБКА - {result.get('error', 'Неизвестная ошибка')}")
    
    # Сохранение сводного отчета
    summary_file = os.path.join(results_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Сводный отчет сохранен в {summary_file}")
    logger.info("Сравнительное тестирование завершено")

if __name__ == "__main__":
    main()
