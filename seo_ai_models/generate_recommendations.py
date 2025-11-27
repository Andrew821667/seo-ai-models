#!/usr/bin/env python3
"""
Генерация SEO-рекомендаций на основе результатов анализа.
"""
import argparse
import json
import sys
from pathlib import Path
from seo_ai_models.models.seo_advisor.suggester.suggester import Suggester


def generate_recommendations(input_file: str, output_file: str) -> bool:
    """
    Генерирует SEO-рекомендации на основе результатов анализа.
    
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
            analysis_data = json.load(f)
        
        # Инициализируем генератор рекомендаций
        print("⚙️ Инициализация Suggester...")
        suggester = Suggester()
        
        # Генерируем рекомендации
        print("▶ Генерация SEO-рекомендаций...")
        
        # Извлекаем необходимые данные из analysis_data
        # Создаем basic_recommendations из результатов анализа
        basic_recommendations = {}
        
        # Создаем feature_scores на основе метрик анализа
        feature_scores = {}
        if 'content_analysis' in analysis_data:
            content = analysis_data['content_analysis']
            if 'readability' in content:
                feature_scores['readability'] = content['readability'].get('score', 0.5)
            if 'keyword_analysis' in content:
                feature_scores['keyword_density'] = content['keyword_analysis'].get('density', 0.5)
        
        if 'technical_seo' in analysis_data:
            tech = analysis_data['technical_seo']
            feature_scores['content_length'] = min(1.0, tech.get('content_length', 0) / 2000)
            
        if 'meta_tags' in analysis_data:
            meta = analysis_data['meta_tags']
            feature_scores['meta_tags'] = 1.0 if meta.get('title') and meta.get('description') else 0.5
            
        # Определяем industry (по умолчанию 'general')
        industry = analysis_data.get('industry', 'general')
        
        # Вызываем метод с правильными параметрами
        recommendations = suggester.generate_suggestions(
            basic_recommendations=basic_recommendations,
            feature_scores=feature_scores,
            industry=industry
        )        # Формируем Markdown отчет
        # Формируем Markdown отчет
        print(f"■ Создание Markdown отчета...")
        
        markdown_lines = []
        markdown_lines.append("# Рекомендации по SEO-оптимизации\n")
        markdown_lines.append(f"*Дата генерации:* {analysis_data.get('timestamp', 'Неизвестно')}\n")
        markdown_lines.append(f"*URL:* {analysis_data.get('url', 'Неизвестно')}\n\n")
        markdown_lines.append("---\n\n")
        
        # Добавляем рекомендации по категориям
        if recommendations:
            for category, suggestions_list in recommendations.items():
                # Заголовок категории
                category_name = category.replace('_', ' ').title()
                markdown_lines.append(f"## {category_name}\n\n")
                
                # Добавляем рекомендации
                if suggestions_list:
                    for suggestion in suggestions_list:
                        markdown_lines.append(f"- {suggestion}\n")
                    markdown_lines.append("\n")
                else:
                    markdown_lines.append("*Рекомендаций нет*\n\n")
        else:
            markdown_lines.append("Рекомендации не сгенерированы.\n")
        
        markdown_content = ''.join(markdown_lines)        
        # Сохраняем в файл
        print(f"💾 Сохранение рекомендаций в {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"\n✅ Рекомендации успешно сгенерированы!\n")
        return True
        
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {input_file} не найден")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка парсинга JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Главная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description='Генерация SEO-рекомендаций на основе результатов анализа'
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
    success = generate_recommendations(args.input, args.output)
    
    # Возвращаем код выхода
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
