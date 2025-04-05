# SEO AI Models

Библиотека для анализа и оптимизации контента с использованием искусственного интеллекта и машинного обучения.

## Компоненты проекта

### Основные компоненты

SEOAdvisor - Основной класс для анализа контента и выдачи рекомендаций
TextProcessor - Обработка текста и расчет метрик читабельности
ContentAnalyzer - Анализ содержимого, ключевых слов и структуры контента
EEATAnalyzer - Оценка E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)
SemanticAnalyzer - Семантический анализ контента
CalibratedRankPredictor - Предсказание позиций в поисковой выдаче
Suggester - Генерация рекомендаций по улучшению контента

### Компоненты парсинга и анализа

WebCrawler - Сканирование сайтов и сбор URL
ContentExtractor - Извлечение текста и структуры из HTML
MetaExtractor - Анализ мета-тегов, заголовков и ссылок
SERPAnalyzer - Анализ поисковой выдачи и конкурентов
ParsingPipeline - Интеграция парсинг-компонентов в единый конвейер

## Установка

pip install -r requirements.txt

## Примеры использования

### Анализ контента

from seo_ai_models.models.seo_advisor.advisor import SEOAdvisor

advisor = SEOAdvisor()
report = advisor.analyze_content("# Заголовок статьи\n\nЭто текст статьи...")
print(report)

### Парсинг и анализ сайта

from seo_ai_models.parsers.parsing_pipeline import ParsingPipeline

# Инициализация парсинг-конвейера
pipeline = ParsingPipeline()

# Анализ одной страницы
page_analysis = pipeline.analyze_url("https://example.com")

# Сканирование и анализ всего сайта
site_analysis = pipeline.crawl_and_analyze_site("https://example.com", max_pages=10)

# Анализ ключевого слова
keyword_analysis = pipeline.analyze_keyword("ключевое слово")

# Сохранение результатов в файл
pipeline.save_analysis_to_file(site_analysis, "site_analysis.json")

### Использование отдельных парсинг-компонентов

from seo_ai_models.parsers.crawlers.web_crawler import WebCrawler
from seo_ai_models.parsers.extractors.content_extractor import ContentExtractor
from seo_ai_models.parsers.extractors.meta_extractor import MetaExtractor
from seo_ai_models.parsers.analyzers.serp_analyzer import SERPAnalyzer

# Сканирование сайта
crawler = WebCrawler(base_url="https://example.com", max_pages=20)
crawl_results = crawler.crawl()

# Извлечение контента из HTML
content_extractor = ContentExtractor()
with open("page.html", "r", encoding="utf-8") as f:
    html_content = f.read()
content_data = content_extractor.extract_content(html_content, "https://example.com")

# Извлечение мета-информации
meta_extractor = MetaExtractor()
meta_data = meta_extractor.extract_meta_information(html_content, "https://example.com")

# Анализ поисковой выдачи
serp_analyzer = SERPAnalyzer()
serp_results = serp_analyzer.analyze_top_results("запрос")

### Демонстрационный скрипт

В проекте есть демонстрационный скрипт, показывающий использование компонентов парсинга:

python -m seo_ai_models.demo_parsing pipeline --mode site --url https://example.com --max-pages 5 --output analysis.json

## Разработка и тестирование

# Запуск всех тестов
PYTHONPATH=/path/to/seo-ai-models pytest tests/

# Запуск тестов для конкретного модуля
PYTHONPATH=/path/to/seo-ai-models pytest tests/parsers/
