# -*- coding: utf-8 -*-
"""
Запуск тестов для компонентов Фазы 5.
"""

import unittest
import sys
import os

# Добавляем корневую директорию проекта в путь для импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем тесты
from seo_ai_models.tests.test_freemium_components import *
from seo_ai_models.tests.test_scaling_components import *

if __name__ == "__main__":
    # Запускаем все тесты
    unittest.main()
