#!/usr/bin/env python3
"""
Генерация SEO-рекомендаций на основе результатов анализа.
Использует все доступные метрики из EnhancedContentAnalyzer.
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
        
        # Создаем basic_recommendations на основе результатов анализа
        basic_recommendations = {}
        
        # === 1. АНАЛИЗ КОНТЕНТА ===
        if 'technical_seo' in analysis_data:
            tech = analysis_data['technical_seo']
            content_len = tech.get('content_length', 0)
            
            if content_len < 300:
                basic_recommendations.setdefault('content_length', []).append(
                    f"🔴 Критически малый объем контента ({content_len} слов). "
                    f"Рекомендуется минимум 500-1000 слов для хорошей индексации."
                )
            elif content_len < 1000:
                basic_recommendations.setdefault('content_length', []).append(
                    f"🟡 Объем контента ({content_len} слов) ниже рекомендуемого. "
                    f"Добавьте детальное описание темы до 1000-1500 слов."
                )
        
        # === 2. META-ТЕГИ ===
        if 'meta_tags' in analysis_data:
            meta = analysis_data['meta_tags']
            
            # Title
            if not meta.get('title'):
                basic_recommendations.setdefault('meta_tags', []).append(
                    "🔴 Отсутствует meta title. Добавьте уникальный заголовок с ключевыми словами (50-60 символов)."
                )
            else:
                title_len = len(meta.get('title', ''))
                if title_len < 30:
                    basic_recommendations.setdefault('meta_tags', []).append(
                        f"🟠 Meta title слишком короткий ({title_len} символов). Рекомендуемая длина: 50-60 символов."
                    )
                elif title_len > 70:
                    basic_recommendations.setdefault('meta_tags', []).append(
                        f"🟠 Meta title слишком длинный ({title_len} символов). Сократите до 60 символов."
                    )
            
            # Description
            if not meta.get('description'):
                basic_recommendations.setdefault('meta_tags', []).append(
                    "🔴 Отсутствует meta description. Добавьте привлекательное описание до 160 символов."
                )
            else:
                desc_len = len(meta.get('description', ''))
                if desc_len < 70:
                    basic_recommendations.setdefault('meta_tags', []).append(
                        f"🟡 Meta description короткий ({desc_len} символов). Оптимально: 120-160 символов."
                    )
                elif desc_len > 160:
                    basic_recommendations.setdefault('meta_tags', []).append(
                        f"🟡 Meta description длинный ({desc_len} символов). Сократите до 160 символов."
                    )
        
        # === 3. СТРУКТУРА ЗАГОЛОВКОВ ===
        if 'technical_seo' in analysis_data:
            tech = analysis_data['technical_seo']
            
            # H1
            h1_count = tech.get('h1_count', 0)
            if h1_count == 0:
                basic_recommendations.setdefault('headers', []).append(
                    "🔴 Отсутствует заголовок H1. Добавьте главный заголовок страницы."
                )
            elif h1_count > 1:
                basic_recommendations.setdefault('headers', []).append(
                    f"🟠 Найдено {h1_count} заголовков H1. Должен быть только один H1."
                )
            
            # Общее количество заголовков
            total_headers = sum(tech.get(f'h{i}_count', 0) for i in range(1, 7))
            if total_headers < 3:
                basic_recommendations.setdefault('headers', []).append(
                    "🟡 Мало подзаголовков. Добавьте H2-H3 для структурирования контента."
                )
        
        # === 4. ИЗОБРАЖЕНИЯ ===
        if 'technical_seo' in analysis_data:
            tech = analysis_data['technical_seo']
            images_count = tech.get('images_count', 0)
            images_without_alt = tech.get('images_without_alt', 0)
            
            if images_count > 0 and images_without_alt > 0:
                basic_recommendations.setdefault('images', []).append(
                    f"🟠 У {images_without_alt} из {images_count} изображений нет alt-текста. "
                    f"Добавьте описательные alt для доступности и SEO."
                )
            elif images_count == 0 and content_len > 500:
                basic_recommendations.setdefault('images', []).append(
                    "🟡 На странице нет изображений. Добавьте релевантные изображения."
                )
        
        # === 5. ВНУТРЕННИЕ ССЫЛКИ ===
        if 'technical_seo' in analysis_data:
            tech = analysis_data['technical_seo']
            internal_links = tech.get('internal_links_count', 0)
            
            if internal_links < 3:
                basic_recommendations.setdefault('linking', []).append(
                    f"🟠 Мало внутренних ссылок ({internal_links}). Добавьте минимум 3-5 ссылок."
                )
            elif internal_links > 100:
                basic_recommendations.setdefault('linking', []).append(
                    f"🟡 Очень много ссылок ({internal_links}). Оставьте только важные."
                )
        
        # === 6. ТЕХНИЧЕСКОЕ SEO ===
        if 'technical_seo' in analysis_data:
            tech = analysis_data['technical_seo']
            
            # Canonical URL
            if not tech.get('canonical_url'):
                basic_recommendations.setdefault('technical', []).append(
                    "🟡 Отсутствует canonical URL. Добавьте для предотвращения дублей."
                )
            
            # Schema.org
            if not tech.get('has_schema_markup', False):
                basic_recommendations.setdefault('technical', []).append(
                    "🟠 Отсутствует Schema.org разметка. Добавьте для расширенных сниппетов."
                )
            
            # Внешние ссылки
            external_links = tech.get('external_links_count', 0)
            if external_links == 0:
                basic_recommendations.setdefault('linking', []).append(
                    "🟡 Нет внешних ссылок. Добавьте ссылки на авторитетные источники."
                )
        
        # === 7. ЧИТАБЕЛЬНОСТЬ ===
        if 'content_analysis' in analysis_data:
            content = analysis_data['content_analysis']
            if 'readability' in content:
                read_score = content['readability'].get('score', 0)
                if read_score < 0.4:
                    basic_recommendations.setdefault('readability', []).append(
                        f"🟠 Низкая читабельность ({read_score:.2f}). Упростите предложения."
                    )
        
        # === 8. СТРУКТУРА КОНТЕНТА ===
        if 'content_analysis' in analysis_data:
            content = analysis_data['content_analysis']
            
            if not content.get('has_introduction', False):
                basic_recommendations.setdefault('structure', []).append(
                    "🟡 Добавьте введение с обзором темы."
                )
            if not content.get('has_conclusion', False):
                basic_recommendations.setdefault('structure', []).append(
                    "🟡 Добавьте заключение с выводами."
                )
        
        # === 9. KEYWORD АНАЛИЗ ===
        if 'content_analysis' in analysis_data:
            content = analysis_data['content_analysis']
            if 'keyword_analysis' in content:
                kw = content['keyword_analysis']
                density = kw.get('density', 0)
                
                if density < 0.01:
                    basic_recommendations.setdefault('keywords', []).append(
                        "🟠 Низкая плотность ключевых слов. Добавьте целевые keywords."
                    )
                elif density > 0.05:
                    basic_recommendations.setdefault('keywords', []).append(
                        f"🟠 Высокая плотность keywords ({density:.1%}). Оптимально: 1-3%."
                    )
        
        # === 10. МУЛЬТИМЕДИА ===
        if 'technical_seo' in analysis_data:
            tech = analysis_data['technical_seo']
            
            # Списки
            if tech.get('lists_count', 0) == 0:
                basic_recommendations.setdefault('structure', []).append(
                    "🟡 Используйте списки для структурирования информации."
                )
        
        # Если нет рекомендаций
        if not basic_recommendations:
            basic_recommendations['general'] = [
                '✅ Основные SEO-метрики в хорошем состоянии!'
            ]
        
        # Создаем feature_scores
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
            feature_scores['meta_score'] = tech.get('meta_score', 0.5)
            feature_scores['tech_seo_score'] = tech.get('tech_seo_score', 0.5)
        
        if 'meta_tags' in analysis_data:
            meta = analysis_data['meta_tags']
            feature_scores['meta_tags'] = 1.0 if meta.get('title') and meta.get('description') else 0.5
        
        # Industry
        industry = analysis_data.get('industry', 'general')
        
        # Вызываем suggester
        recommendations = suggester.generate_suggestions(
            basic_recommendations=basic_recommendations,
            feature_scores=feature_scores,
            industry=industry
        )
        
        # Формируем Markdown
        print(f"■ Создание Markdown отчета...")
        
        markdown_lines = []
        markdown_lines.append("# 📊 Рекомендации по SEO-оптимизации\n\n")
        markdown_lines.append(f"**Дата:** {analysis_data.get('timestamp', 'Неизвестно')}  \n")
        markdown_lines.append(f"**URL:** {analysis_data.get('url', 'Неизвестно')}\n\n")
        markdown_lines.append("---\n\n")
        
        # Добавляем рекомендации
        if recommendations:
            for category, suggestions_list in recommendations.items():
                category_name = category.replace('_', ' ').title()
                markdown_lines.append(f"## {category_name}\n\n")
                
                if suggestions_list:
                    for suggestion in suggestions_list:
                        markdown_lines.append(f"- {suggestion}\n")
                    markdown_lines.append("\n")
                else:
                    markdown_lines.append("*Рекомендаций нет*\n\n")
        else:
            markdown_lines.append("Рекомендации не сгенерированы.\n")
        
        markdown_content = ''.join(markdown_lines)
        
        # Сохраняем
        print(f"💾 Сохранение в {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
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
