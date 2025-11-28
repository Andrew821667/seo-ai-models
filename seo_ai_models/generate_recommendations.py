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
        # Создаем basic_recommendations на основе результатов анализа
        basic_recommendations = {}
        
        # Генерируем базовые рекомендации на основе метрик
        
        # Анализ content_length
        if 'technical_seo' in analysis_data:
            tech = analysis_data['technical_seo']
            content_len = tech.get('content_length', 0)
            
            if content_len < 300:
                basic_recommendations.setdefault('content_length', []).append(
                    "Критически малый объем контента ({} слов). Рекомендуется минимум 500-1000 слов.".format(content_len)
                )
            elif content_len < 1000:
                basic_recommendations.setdefault('content_length', []).append(
                    "Объем контента ({} слов) ниже рекомендуемого. Добавьте детальное описание темы.".format(content_len)
                )
                
        # Анализ meta_tags
        if 'meta_tags' in analysis_data:
            meta = analysis_data['meta_tags']
            
            if not meta.get('title'):
                basic_recommendations.setdefault('meta_tags', []).append(
                    "Отсутствует meta title. Добавьте уникальный заголовок с ключевыми словами."
                )
            elif len(meta.get('title', '')) < 30:
                basic_recommendations.setdefault('meta_tags', []).append(
                    "Meta title слишком короткий. Рекомендуемая длина: 50-60 символов."
                )
            elif len(meta.get('title', '')) > 70:
                basic_recommendations.setdefault('meta_tags', []).append(
                    "Meta title слишком длинный. Сократите до 60 символов."
                )
                
            if not meta.get('description'):
                basic_recommendations.setdefault('meta_tags', []).append(
                    "Отсутствует meta description. Добавьте описание до 160 символов."
                )
                
        # Анализ readability
        if 'content_analysis' in analysis_data:
            content = analysis_data['content_analysis']
            if 'readability' in content:
                read_score = content['readability'].get('score', 0)
                if read_score < 0.4:
                    basic_recommendations.setdefault('readability', []).append(
                        "Низкая читабельность текста. Упростите предложения и добавьте подзаголовки."
                    )
                    
        # Если нет ни одной рекомендации, добавим общую
        if not basic_recommendations:
            basic_recommendations['general'] = ['Основные SEO-метрики в норме. Продолжайте работу над контентом.']

        
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
        )
        
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
        
        # Сохраняем рекомендации в файл
        print(f"💾 Сохранение рекомендаций в {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print("✅ Рекомендации успешно сгенерированы!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при генерации рекомендаций: {str(e)}")
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
