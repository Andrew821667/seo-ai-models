#!/usr/bin/env python3
"""
Упрощённая генерация SEO-рекомендаций на основе плоской структуры данных.
Работает напрямую с выходом EnhancedContentAnalyzer.
"""
import argparse
import json
import sys
from pathlib import Path


def generate_recommendations_simple(input_file: str, output_file: str) -> bool:
    """
    Генерирует SEO-рекомендации из плоской структуры данных.

    Args:
        input_file: Путь к JSON файлу с результатами анализа
        output_file: Путь к выходному Markdown файлу

    Returns:
        bool: True если успешно, False в противном случае
    """
    try:
        # Читаем результаты анализа
        print(f"📂 Чтение результатов анализа из {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("▶ Генерация SEO-рекомендаций...")

        # Извлекаем метрики
        word_count = data.get('word_count', 0)
        sentence_count = data.get('sentence_count', 0)
        headers_count = data.get('headers_count', 0)
        paragraphs_count = data.get('paragraphs_count', 0)
        readability = data.get('readability', 0)
        has_intro = data.get('has_introduction', False)
        has_conclusion = data.get('has_conclusion', False)

        recommendations = []

        # === КРИТИЧЕСКИЕ ПРОБЛЕМЫ ===
        if word_count == 0 and sentence_count == 0:
            recommendations.append("## 🔴 КРИТИЧЕСКАЯ ОШИБКА")
            recommendations.append("")
            recommendations.append("Анализатор не смог получить контент со страницы!")
            recommendations.append("")
            recommendations.append("**Возможные причины:**")
            recommendations.append("1. Сайт использует JavaScript для рендеринга (SPA)")
            recommendations.append("2. Сайт блокирует ботов")
            recommendations.append("3. Проблемы с Playwright")
            recommendations.append("")
            recommendations.append("**Действия:**")
            recommendations.append("- Проверьте доступность сайта в браузере")
            recommendations.append("- Проверьте логи GitHub Actions")
            recommendations.append("- Увеличьте таймауты парсера")
            recommendations.append("")

        # === КОНТЕНТ ===
        if word_count > 0:
            recommendations.append("## 📝 Контент")
            recommendations.append("")

            if word_count < 300:
                recommendations.append(f"🔴 **Критически мало текста:** {word_count} слов")
                recommendations.append("   - Минимум для индексации: 500-1000 слов")
                recommendations.append("   - Добавьте детальное описание темы")
                recommendations.append("")
            elif word_count < 1000:
                recommendations.append(f"🟡 **Недостаточно текста:** {word_count} слов")
                recommendations.append("   - Рекомендуется: 1000-1500 слов")
                recommendations.append("   - Расширьте контент дополнительными разделами")
                recommendations.append("")
            else:
                recommendations.append(f"✅ **Объём контента хороший:** {word_count} слов")
                recommendations.append("")

        # === СТРУКТУРА ===
        if word_count > 0:
            recommendations.append("## 🏗️ Структура")
            recommendations.append("")

            if headers_count == 0:
                recommendations.append("🔴 **Отсутствуют заголовки (H1-H6)**")
                recommendations.append("   - Добавьте H1 с главным ключевым словом")
                recommendations.append("   - Используйте H2-H3 для структурирования")
                recommendations.append("")
            elif headers_count < 3:
                recommendations.append(f"🟡 **Мало заголовков:** {headers_count}")
                recommendations.append("   - Добавьте подзаголовки H2-H3")
                recommendations.append("   - Структурируйте контент по разделам")
                recommendations.append("")
            else:
                recommendations.append(f"✅ **Заголовков достаточно:** {headers_count}")
                recommendations.append("")

            if paragraphs_count <= 1:
                recommendations.append(f"🟠 **Мало параграфов:** {paragraphs_count}")
                recommendations.append("   - Разбейте текст на логические блоки")
                recommendations.append("   - Используйте абзацы по 2-4 предложения")
                recommendations.append("")

            if not has_intro:
                recommendations.append("🟡 **Нет введения**")
                recommendations.append("   - Добавьте вводный раздел с обзором темы")
                recommendations.append("")

            if not has_conclusion:
                recommendations.append("🟡 **Нет заключения**")
                recommendations.append("   - Добавьте выводы и краткое резюме")
                recommendations.append("")

        # === ЧИТАБЕЛЬНОСТЬ ===
        if word_count > 0:
            recommendations.append("## 📖 Читабельность")
            recommendations.append("")

            if readability < 0.3:
                recommendations.append(f"🔴 **Низкая читабельность:** {readability:.2f}")
                recommendations.append("   - Упростите предложения")
                recommendations.append("   - Используйте короткие фразы")
                recommendations.append("   - Добавьте списки и таблицы")
                recommendations.append("")
            elif readability < 0.6:
                recommendations.append(f"🟡 **Средняя читабельность:** {readability:.2f}")
                recommendations.append("   - Разнообразьте длину предложений")
                recommendations.append("   - Добавьте больше структурирования")
                recommendations.append("")
            else:
                recommendations.append(f"✅ **Читабельность хорошая:** {readability:.2f}")
                recommendations.append("")

        # === ИТОГИ ===
        if not recommendations:
            recommendations.append("## ✅ Результат")
            recommendations.append("")
            recommendations.append("Основные метрики в норме!")
            recommendations.append("")

        # Формируем Markdown
        print(f"💾 Сохранение в {output_file}...")

        markdown = []
        markdown.append("# 📊 Рекомендации по SEO-оптимизации\n\n")
        markdown.append(f"**URL:** {data.get('metadata', {}).get('url', 'Неизвестно')}  \n")
        markdown.append(f"**Дата:** {data.get('metadata', {}).get('analyzed_at', 'Неизвестно')}\n\n")
        markdown.append("---\n\n")
        markdown.extend([rec + "\n" for rec in recommendations])

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(''.join(markdown))

        print("✅ Рекомендации успешно сгенерированы!")
        return True

    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Главная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description='Генерация SEO-рекомендаций (упрощённая версия)'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Путь к JSON файлу с результатами анализа'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Путь к выходному Markdown файлу'
    )

    args = parser.parse_args()

    # Запускаем генерацию
    success = generate_recommendations_simple(args.input, args.output)

    # Возвращаем код выхода
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
