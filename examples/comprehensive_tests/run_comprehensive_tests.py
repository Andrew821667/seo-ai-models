"""
Запуск всех комплексных тестов.
"""
import sys
import os
import argparse
from datetime import datetime
import importlib.util

# Добавляем путь к проекту
current_dir = os.path.dirname(os.path.abspath(__file__ if '__file__' in globals() else '.'))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

# Функция для импорта модуля из файла
def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Импортируем модули тестов
test_text_processor_path = os.path.join(current_dir, 'test_text_processor.py')
test_text_processor = import_module_from_file('test_text_processor', test_text_processor_path)
test_text_processor_and_consistency = test_text_processor.test_text_processor_and_consistency

# Другие тесты можно импортировать аналогично
# Например:
# test_analyzers_path = os.path.join(current_dir, 'test_analyzers.py')
# test_analyzers = import_module_from_file('test_analyzers', test_analyzers_path)
# test_content_analyzers = test_analyzers.test_content_analyzers
# test_eeat_analyzers = test_analyzers.test_eeat_analyzers

def read_text_from_file(file_path):
    """Чтение текста из файла."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def run_text_processor_test(test_text):
    """Запуск теста текстового процессора."""
    print("\n" + "*"*80)
    print("ТЕСТИРОВАНИЕ ТЕКСТОВОГО ПРОЦЕССОРА")
    print("*"*80)
    print(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Текст для анализа содержит {len(test_text.split())} слов")
    
    # Тестирование текстового процессора и проверки согласованности метрик
    text_processor_results = test_text_processor_and_consistency(test_text)
    
    print("\n" + "*"*80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("*"*80)
    print(f"Дата и время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return text_processor_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Запуск комплексных тестов ядра SEO AI Models')
    parser.add_argument('--file', type=str, help='Путь к файлу с текстом для анализа')
    parser.add_argument('--test', type=str, choices=['all', 'text_processor'], 
                        default='text_processor', help='Какой тест запустить')
    
    args = parser.parse_args()
    
    # Определяем текст для анализа
    if args.file and os.path.exists(args.file):
        test_text = read_text_from_file(args.file)
    else:
        # Тестовый текст по умолчанию
        test_text = """
        # Основы защиты данных в эпоху цифровой трансформации
        
        ## Введение
        
        В современном цифровом мире защита данных становится одним из важнейших приоритетов как для организаций, 
        так и для частных лиц. Каждый день генерируются петабайты информации, которая содержит всё: 
        от личных сообщений до корпоративных секретов.
        
        ## Основные угрозы
        
        Среди основных угроз можно выделить:
        - Фишинг
        - Вредоносное ПО
        - Утечка данных
        
        ## Меры защиты
        
        Для обеспечения безопасности данных рекомендуется:
        - Использовать надежные пароли
        - Применять двухфакторную аутентификацию
        - Регулярно обновлять ПО
        
        Автор: Иван Петров, эксперт по кибербезопасности
        Дата публикации: 15 апреля 2025 г.
        """
    
    # Запускаем выбранные тесты
    if args.test in ['all', 'text_processor']:
        run_text_processor_test(test_text)
