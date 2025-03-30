"""Запуск тестов проекта."""

import os
import unittest
import sys
import importlib.util

def load_module_from_path(path):
    """Загружает модуль из указанного пути."""
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def run_all_tests():
    """Запуск тестов из директории tests с фокусом на новой структуре."""
    # Определяем базовую директорию проекта
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Добавляем корневую директорию в путь для импортов
    sys.path.insert(0, base_dir)
    
    # Создаем тестовый набор
    test_suite = unittest.TestSuite()
    
    # Путь к тесту анализатора
    unit_eeat_test_path = os.path.join(base_dir, 'tests', 'unit', 'seo_advisor', 'analyzers', 'test_eeat_analyzer.py')
    
    # Путь к интеграционному тесту
    integration_test_path = os.path.join(base_dir, 'tests', 'integration', 'test_eeat_integration.py')
    
    # Собираем тесты, если файлы существуют
    if os.path.exists(unit_eeat_test_path):
        print(f"Найден файл: {unit_eeat_test_path}")
        # Загружаем модуль
        try:
            # Напрямую загружаем тесты
            loader = unittest.TestLoader()
            module = load_module_from_path(unit_eeat_test_path)
            if module:
                tests = loader.loadTestsFromModule(module)
                test_suite.addTests(tests)
        except Exception as e:
            print(f"Ошибка при загрузке модуля {unit_eeat_test_path}: {e}")
    else:
        print(f"Файл не найден: {unit_eeat_test_path}")

    # Аналогично для интеграционного теста
    if os.path.exists(integration_test_path):
        print(f"Найден файл: {integration_test_path}")
        try:
            # Напрямую загружаем тесты
            loader = unittest.TestLoader()
            module = load_module_from_path(integration_test_path)
            if module:
                tests = loader.loadTestsFromModule(module)
                test_suite.addTests(tests)
        except Exception as e:
            print(f"Ошибка при загрузке модуля {integration_test_path}: {e}")
    else:
        print(f"Файл не найден: {integration_test_path}")
    
    # Запускаем тесты
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Возвращаем статус выполнения
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
