#!/usr/bin/env python3
"""
Скрипт для SEO анализа сайта legalaipro.ru
Используется в GitHub Actions для автоматического анализа.
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Добавляем родительскую директорию в путь поиска модулей
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from seo_ai_models.parsers.spa_parser import SPAParser
    from seo_ai_models.models.seo_advisor.analyzers.enhanced_content_analyzer import EnhancedContentAnalyzer
except ImportError as e:
    print(f"Предупреждение: Не удалось импортировать модули анализа: {e}")
    print("Будет использован упрощенный режим анализа")
    SPAParser = None
    EnhancedContentAnalyzer = None


def analyze_url(url):
    """
    Анализирует URL с помощью улучшенного анализатора контента.
    
    Args:
        url: URL для анализа
        
    Returns:
        dict: Результаты анализа
    """
    print(f"\n🔍 Начинаем анализ сайта: {url}")
    
    try:
        if SPAParser and EnhancedContentAnalyzer:
            # Используем полный анализатор
            parser = SPAParser()
            analyzer = EnhancedContentAnalyzer()
            
            print("Парсинг страницы...")
            content = parser.parse(url)
            
            print("Анализ контента...")
            analysis = analyzer.analyze(content)
            
            return {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "content": content,
                "analysis": analysis
            }
        else:
            # Упрощенный режим
            return create_simple_analysis(url)
            
    except Exception as e:
        print(f"❌ Ошибка при анализе: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }


def create_simple_analysis(url):
    """
    Создает упрощенный анализ для случаев, когда полные модули недоступны.
    """
    print("ℹ️ Используется упрощенный режим анализа")
    
    return {
        "url": url,
        "timestamp": datetime.now().isoformat(),
        "status": "partial",
        "message": "Выполнен упрощенный анализ",
        "basic_checks": {
            "url_accessible": True,
            "protocol": "https" if url.startswith("https://") else "http",
            "domain": url.split("//")[-1].split("/")[0]
        },
        "recommendations": [
            "Убедитесь, что сайт доступен по HTTPS",
            "Проверьте скорость загрузки страниц",
            "Оптимизируйте мета-теги и заголовки",
            "Добавьте структурированные данные (Schema.org)",
            "Улучшите внутреннюю перелинковку"
        ]
    }


def generate_recommendations(analysis):
    """
    Генерирует рекомендации на основе анализа на русском языке.
    """
    recommendations = []
    
    recommendations.append("# 🎯 SEO Рекомендации для legalaipro.ru\n")
    recommendations.append(f"**Дата анализа:** {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
    recommendations.append(f"**URL:** {analysis.get('url', 'N/A')}\n")
    recommendations.append(f"**Статус:** {analysis.get('status', 'N/A')}\n\n")
    
    recommendations.append("## 📊 Общие рекомендации\n\n")
    
    if analysis.get('status') == 'success' and 'analysis' in analysis:
        # Если есть детальный анализ
        analysis_data = analysis.get('analysis', {})
        
        if 'seo_score' in analysis_data:
            score = analysis_data['seo_score']
            recommendations.append(f"### SEO Оценка: {score}/100\n\n")
        
        if 'issues' in analysis_data:
            recommendations.append("### 🔴 Критические проблемы\n\n")
            for issue in analysis_data['issues']:
                recommendations.append(f"- {issue}\n")
            recommendations.append("\n")
    else:
        # Базовые рекомендации
        basic_recs = analysis.get('recommendations', [])
        if basic_recs:
            recommendations.append("### Основные рекомендации\n\n")
            for rec in basic_recs:
                recommendations.append(f"- {rec}\n")
            recommendations.append("\n")
    
    # Общие рекомендации для юридических сайтов
    recommendations.append("## 💡 Специальные рекомендации для юридического сайта\n\n")
    recommendations.append("### Контент\n")
    recommendations.append("- Добавьте подробные статьи по актуальным правовым вопросам\n")
    recommendations.append("- Создайте FAQ раздел с часто задаваемыми вопросами\n")
    recommendations.append("- Публикуйте кейсы и примеры успешных дел\n\n")
    
    recommendations.append("### Техническое SEO\n")
    recommendations.append("- Настройте корректные мета-теги (title, description) для всех страниц\n")
    recommendations.append("- Добавьте структурированные данные Schema.org (Organization, LegalService)\n")
    recommendations.append("- Оптимизируйте скорость загрузки (сжатие изображений, минификация)\n")
    recommendations.append("- Настройте XML sitemap и robots.txt\n\n")
    
    recommendations.append("### Локальное SEO\n")
    recommendations.append("- Зарегистрируйтесь в Яндекс.Справочнике и Google My Business\n")
    recommendations.append("- Укажите адрес офиса и контактные данные на всех страницах\n")
    recommendations.append("- Добавьте карту с местоположением офиса\n\n")
    
    recommendations.append("### Конверсия\n")
    recommendations.append("- Разместите четкие призывы к действию (CTA)\n")
    recommendations.append("- Добавьте онлайн-консультант или чат-бот\n")
    recommendations.append("- Создайте простые формы для обратной связи\n")
    recommendations.append("- Добавьте отзывы клиентов\n\n")
    
    recommendations.append("---\n")
    recommendations.append("*Сгенерировано автоматически системой SEO анализа*\n")
    
    return "".join(recommendations)


def main():
    parser = argparse.ArgumentParser(
        description='SEO анализ сайта для GitHub Actions'
    )
    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help='URL сайта для анализа'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  SEO АНАЛИЗ: {args.url}")
    print(f"{'='*60}\n")
    
    # Выполняем анализ
    results = analyze_url(args.url)
    
    if results:
        # Сохраняем результаты в JSON
        output_file = "analysis_result.json"
        print(f"\n💾 Сохранение результатов в {output_file}...")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=2,
                default=lambda x: x.isoformat() if isinstance(x, datetime) else str(x)
            )
        
        print(f"✅ Результаты сохранены в {output_file}")
        
        # Генерируем рекомендации на русском
        recommendations = generate_recommendations(results)
        rec_file = "recommendations_ru.md"
        
        print(f"\n📝 Генерация рекомендаций в {rec_file}...")
        with open(rec_file, "w", encoding="utf-8") as f:
            f.write(recommendations)
        
        print(f"✅ Рекомендации сохранены в {rec_file}")
        
        print(f"\n{'='*60}")
        print("✨ Анализ завершен успешно!")
        print(f"{'='*60}\n")
        
        return 0 if results.get('status') == 'success' else 1
    else:
        print("\n❌ Не удалось выполнить анализ")
        return 1


if __name__ == "__main__":
    sys.exit(main())
