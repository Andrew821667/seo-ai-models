#!/usr/bin/env python3
"""
Скрипт для настройки окружения и установки всех необходимых зависимостей.
"""

import subprocess
import sys
import os


def install_dependencies():
    """Установка необходимых зависимостей."""
    print("Установка необходимых зависимостей...")

    # Основные зависимости
    dependencies = ["playwright", "beautifulsoup4", "nltk", "spacy", "requests"]

    for dependency in dependencies:
        print(f"Установка {dependency}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dependency])
            print(f"{dependency} успешно установлен.")
        except subprocess.CalledProcessError:
            print(f"Ошибка при установке {dependency}.")
            return False

    # Установка Playwright браузеров
    try:
        print("Установка браузеров для Playwright...")
        subprocess.check_call([sys.executable, "-m", "playwright", "install"])
        print("Браузеры для Playwright успешно установлены.")
    except subprocess.CalledProcessError:
        print("Ошибка при установке браузеров для Playwright.")
        return False

    print("Все зависимости успешно установлены!")
    return True


def main():
    """Основная функция скрипта."""
    success = install_dependencies()
    if success:
        print("\nОкружение успешно настроено. Теперь вы можете запустить демо-скрипт.")
        print("Пример запуска демо-скрипта:")
        print("python demo_consistency_checker.py")
    else:
        print("\nПроизошла ошибка при настройке окружения.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
