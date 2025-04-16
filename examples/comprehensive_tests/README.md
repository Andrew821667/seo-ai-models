# Комплексные тесты ядра SEO AI Models

Этот набор тестов предназначен для проверки функциональности основных компонентов ядра SEO AI Models.

## Доступные тесты

1. **test_text_processor.py** - Тестирование TextProcessor и MetricsConsistencyChecker
2. **run_comprehensive_tests.py** - Запуск всех тестов (в текущей версии поддерживается только тест текстового процессора)

## Использование

Чтобы запустить базовый тест текстового процессора:

    python run_comprehensive_tests.py

Чтобы использовать собственный текст из файла:

    python run_comprehensive_tests.py --file path/to/your/text.txt

## Дополнение тестов

Вы можете добавить тесты для других компонентов, следуя структуре файла `test_text_processor.py`. Для интеграции нового теста в `run_comprehensive_tests.py`, используйте функцию `import_module_from_file()`.
