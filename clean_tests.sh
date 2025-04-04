#!/bin/bash
# Скрипт для очистки устаревших тестов в проекте seo-ai-models

PROJECT_DIR="$(pwd)"
cd "${PROJECT_DIR}"

# Создание каталога для бэкапа
mkdir -p old_tests_backup

# 1. Перемещение тестов с неправильными импортами во временную директорию
echo "Перемещение тестов с неправильными импортами..."

# Находим тесты с импортами из "models." или "seo_advisor."
grep -rl "from models\." tests/ 2>/dev/null | xargs -I{} cp {} old_tests_backup/ 2>/dev/null
grep -rl "from seo_advisor\." tests/ 2>/dev/null | xargs -I{} cp {} old_tests_backup/ 2>/dev/null

# 2. Копируем исправленные тесты в правильную структуру
echo "Копирование исправленных тестов..."

# Создаем структуру директорий
mkdir -p tests/unit/fixed
mkdir -p tests/integration/fixed

# Копируем исправленные тесты
cp test_text_processor.py tests/unit/fixed/
cp test_content_analyzer.py tests/unit/seo_advisor/analyzers/
cp test_eeat_analyzer.py tests/unit/seo_advisor/analyzers/
cp test_calibrated_rank_predictor.py tests/unit/seo_advisor/predictors/
cp test_seo_advisor.py tests/unit/seo_advisor/

echo "Очистка тестов завершена."
echo "Резервные копии сохранены в директории: old_tests_backup"
echo "Исправленные тесты установлены в правильные директории."
