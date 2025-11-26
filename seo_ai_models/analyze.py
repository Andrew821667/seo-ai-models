#!/usr/bin/env python3
"""
Полноценный SEO анализ с использованием всех возможностей seo-ai-models.
Используется в GitHub Actions для автоматического анализа legalaipro.ru.
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
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать модули анализа: {e}")
    print("Установите все зависимости через: pip install -r requirements.txt")
    MODULES_AVAILABLE = False


def analyze_url_full(url):
    """
    Полноценный анализ URL с использованием SPAParser и EnhancedContentAnalyzer.
    
    Args:
        url: URL для анализа
        
    Returns:
        dict: Полные результаты анализа
    """
    print(f"\n{'='*60}")
    print(f"🔍 SEO АНАЛИЗ: {url}")
    print(f"{'='*60}\n")
    
    if not MODULES_AVAILABLE:
        print("❌ Модули недоступны. Невозможно выполнить анализ.")
        return None
    
    try:
        # Инициализируем парсер с правильными параметрами
        print("⚙️  Инициализация SPAParser...")
        parser = SPAParser(
            wait_for_load=7000,
            wait_for_timeout=45000,
            record_ajax=True
        )
        
        # Парсим URL
        print(f"📄 Парсинг страницы {url}...")
        parsed_data = parser.parse(url)
        
        if not parsed_data.get("success"):
            error_msg = parsed_data.get('error', 'Неизвестная ошибка')
            print(f"❌ Ошибка парсинга: {error_msg}")
            return {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": error_msg
            }
        
        # Извлекаем контент
        print("📝 Извлечение контента...")
        content = ""
        if "content" in parsed_data and "all_text" in parsed_data["content"].get("content", {}):
            content = parsed_data["content"]["content"]["all_text"]
        else:
            # Альтернативный способ извлечения контента
            for paragraph in parsed_data.get("content", {}).get("content", {}).get("paragraphs", []):
                content += paragraph + "\n\n"
        
        print(f"✅ Извлечено {len(content)} символов контента")
        
        # Получаем HTML контент
        html_content = parsed_data.get("html", "")
        
        # Инициализируем анализатор
        print("⚙️  Инициализация EnhancedContentAnalyzer...")
        analyzer = EnhancedContentAnalyzer()
        
        # Анализируем контент правильным методом
        print("🔬 Анализ контента...")
        metrics = analyzer.analyze_content(content, html_content)
        
        # Анализируем ключевые слова (если есть)
        keywords = [
            "юридические услуги",
            "юрист",
            "правовая помощь",
            "консультация",
            "договор"
        ]
        print("🔑 Анализ ключевых слов...")
        keyword_analysis = analyzer.extract_keywords(content, keywords)
        
        print("\n✨ Анализ завершён успешно!\n")
        
        return {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "parsed_data": parsed_data,
            "content_length": len(content),
            "metrics": metrics,
            "keyword_analysis": keyword_analysis,
            "html_length": len(html_content)
        }
        
    except Exception as e:
        print(f"\n❌ ОШИБКА при анализе: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def generate_recommendations(analysis):
    """
    Генерирует подробные рекомендации на русском языке.
    """
    recommendations = []
    
    recommendations.append("# 🎯 SEO Рекомендации для legalaipro.ru\n\n")
    recommendations.append(f"**Дата анализа:** {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
    recommendations.append(f"**URL:** {analysis.get('url', 'N/A')}\n")
    recommendations.append(f"**Статус:** {analysis.get('status', 'N/A')}\n\n")
    
    if analysis.get('status') == 'success':
        recommendations.append("## ✅ Результаты анализа\n\n")
        
        content_len = analysis.get('content_length', 0)
        recommendations.append(f"- **Объём контента:** {content_len} символов\n")
        
        if content_len < 1000:
            recommendations.append("  ⚠️ ВНИМАНИЕ: Контента недостаточно! Рекомендуемый минимум: 2000 символов\n")
        elif content_len < 2000:
            recommendations.append("  ⚠️ Контента маловато. Добавьте ещё хотя бы 1000 символов\n")
        else:
            recommendations.append("  ✅ Объём контента достаточный\n")
        
        recommendations.append("\n")
        
        # Метрики
        metrics = analysis.get('metrics', {})
        if metrics:
            recommendations.append("### 📊 Метрики качества\n\n")
            for key, value in metrics.items():
                recommendations.append(f"- **{key}**: {value}\n")
            recommendations.append("\n")
        
        # Ключевые слова
        kw_analysis = analysis.get('keyword_analysis', {})
        if kw_analysis:
            recommendations.append("### 🔑 Анализ ключевых слов\n\n")
            for kw, data in kw_analysis.items():
                recommendations.append(f"- **{kw}**: {data}\n")
            recommendations.append("\n")
    
    # Общие рекомендации для юридического сайта
    recommendations.append("## 💡 Общие рекомендации\n\n")
    
    recommendations.append("### 📝 Контент\n")
    recommendations.append("- Добавьте подробные статьи по актуальным правовым вопросам (минимум 2000 знаков)\n")
    recommendations.append("- Создайте FAQ раздел с ответами на частые вопросы клиентов\n")
    recommendations.append("- Публикуйте кейсы успешных дел с описанием решения\n")
    recommendations.append("- Регулярно обновляйте информацию о изменениях в законодательстве\n\n")
    
    recommendations.append("### 🔧 Техническое SEO\n")
    recommendations.append("- Настройте уникальные meta-теги (title, description) для каждой страницы\n")
    recommendations.append("- Добавьте структурированные данные Schema.org: Organization, LegalService, Attorney\n")
    recommendations.append("- Оптимизируйте скорость загрузки (цель: < 3 сек):\n")
    recommendations.append("  - Сжатие изображений (WebP формат)\n")
    recommendations.append("  - Минификация CSS/JS\n")
    recommendations.append("  - Включение gzip/brotli сжатия\n")
    recommendations.append("- Настройте корректный XML sitemap и robots.txt\n")
    recommendations.append("- Используйте rel='canonical' для предотвращения дублей\n\n")
    
    recommendations.append("### 📍 Локальное SEO\n")
    recommendations.append("- Зарегистрируйтесь в Яндекс.Справочнике и Google My Business\n")
    recommendations.append("- Добавьте адрес офиса и контакты на все страницы\n")
    recommendations.append("- Встройте Google Maps с местоположением офиса\n")
    recommendations.append("- Укажите часы работы и способы связи\n\n")
    
    recommendations.append("### 🎯 Конверсия\n")
    recommendations.append("- Разместите чёткие призывы к действию (CTA)\n")
    recommendations.append("- Добавьте онлайн-консультант или чат-бот для быстрой связи\n")
    recommendations.append("- Создайте простые формы обратной связи (макс. 3-5 полей)\n")
    recommendations.append("- Добавьте раздел с отзывами клиентов\n")
    recommendations.append("- Укажите конкретные результаты работы (цифры, кейсы)\n\n")
    
    recommendations.append("---\n")
    recommendations.append(f"*Сгенерировано системой SEO анализа seo-ai-models v1.0*\n")
    recommendations.append(f"*Время генерации: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}*\n")
    
    return "".join(recommendations)


def main():
    parser = argparse.ArgumentParser(
        description='Полноценный SEO анализ сайта для GitHub Actions'
    )
    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help='URL сайта для анализа'
    )
    
    args = parser.parse_args()
    
    # Выполняем полный анализ
    results = analyze_url_full(args.url)
    
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
        
        # Генерируем рекомендации
        recommendations = generate_recommendations(results)
        rec_file = "recommendations_ru.md"
        
        print(f"\n📝 Генерация рекомендаций в {rec_file}...")
        with open(rec_file, "w", encoding="utf-8") as f:
            f.write(recommendations)
        
        print(f"✅ Рекомендации сохранены в {rec_file}")
        
        print(f"\n{'='*60}")
        print("🎉 АНАЛИЗ ЗАВЕРШЁН УСПЕШНО!")
        print(f"{'='*60}\n")
        
        return 0 if results.get('status') == 'success' else 1
    else:
        print("\n❌ НЕ УДАЛОСЬ ВЫПОЛНИТЬ АНАЛИЗ")
        print("Проверьте наличие всех зависимостей\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
