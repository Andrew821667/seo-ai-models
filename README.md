# SEO AI Models

Библиотека для анализа и оптимизации контента с использованием искусственного интеллекта и машинного обучения.

## Компоненты проекта

### Основные компоненты

- **SEOAdvisor** - Основной класс для анализа контента и выдачи рекомендаций
- **TextProcessor** - Обработка текста и расчет метрик читабельности
- **ContentAnalyzer** - Анализ содержимого, ключевых слов и структуры контента
- **EEATAnalyzer** - Оценка E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)
- **SemanticAnalyzer** - Семантический анализ контента
- **CalibratedRankPredictor** - Предсказание позиций в поисковой выдаче
- **Suggester** - Генерация рекомендаций по улучшению контента

## Улучшенный унифицированный парсер SEO AI Models

Расширенная версия парсера для проекта SEO AI Models с поддержкой SPA-сайтов, JavaScript-фреймворков, многопоточной обработки и интеграцией с поисковыми API.

### Ключевые улучшения

#### 1. Поддержка SPA-сайтов

Улучшенный парсер теперь может эффективно работать с одностраничными приложениями (SPA) и сайтами, построенными на JavaScript-фреймворках:

- **Использование Playwright** для полноценного рендеринга JavaScript
- **Автоматическое определение SPA-сайтов** на основе маркеров и структуры HTML
- **Перехват AJAX-запросов** для анализа динамически загружаемого контента
- **Настраиваемое ожидание загрузки** для корректной обработки асинхронного контента

#### 2. Многопоточный параллельный парсинг

Повышение производительности и скорости анализа за счет многопоточной обработки:

- **Параллельная обработка URL** с настраиваемым количеством потоков
- **Контроль нагрузки** с ограничением скорости запросов (rate limiting)
- **Доменно-ориентированное ограничение** для соблюдения политики сайтов
- **Управление ресурсами** с автоматическим перезапуском при ошибках

#### 3. Интеграция с поисковыми API

Доступ к поисковой выдаче и анализ конкурентов из топа:

- **Интеграция с SerpAPI** для получения реальных данных поисковой выдачи
- **Режим имитации** для работы без API-ключей в тестовых целях
- **Извлечение связанных запросов** и "Люди также спрашивают" (PAA)
- **Анализ топ-результатов** для сравнения с конкурентами

#### 4. Расширенный семантический анализ

Улучшенный анализ контента с использованием NLP-подходов:

- **Расширенная токенизация** и лемматизация с использованием SpaCy/NLTK
- **Оценка семантической плотности** и когерентности контента
- **Определение языка** и анализ читабельности
- **Извлечение ключевых слов** и оценка тематической релевантности

## Использование

### Парсинг SPA-сайтов

```python
from seo_ai_models.parsers.unified.unified_parser import UnifiedParser

# Инициализация парсера с SPA-режимом
parser = UnifiedParser(
    force_spa_mode=True,  # Принудительно использовать SPA-режим
    spa_settings={
        "headless": True,  # Фоновый режим браузера
        "wait_for_idle": 3000,  # Ожидание после событий networkidle (мс)
        "wait_for_timeout": 10000,  # Максимальное время ожидания (мс)
        "browser_type": "chromium",  # Тип браузера (chromium, firefox, webkit)
        "intercept_ajax": True  # Перехватывать AJAX-запросы
    },
    extract_semantic=True  # Выполнять семантический анализ
)

# Парсинг URL
result = parser.parse_url("https://react-redux.realworld.io/")

# Обработка результатов
if result.get("success", False):
    page_data = result.get("page_data", {})
    print(f"Заголовок: {page_data.get('structure', {}).get('title', '')}")
    print(f"Слов в контенте: {page_data.get('content', {}).get('word_count', 0)}")
    
    # Доступ к семантическому анализу
    semantic = page_data.get("performance", {}).get("semantic_analysis", {})
    print(f"Семантическая плотность: {semantic.get('semantic_density', 0):.2f}")
```

### Использование поискового API

```python
from seo_ai_models.parsers.unified.unified_parser import UnifiedParser

# Инициализация парсера с API-ключом
parser = UnifiedParser(
    search_api_keys=["YOUR_SERPAPI_KEY"],  # Опционально: API-ключ для SerpAPI
    search_engine="google"  # или "custom" для режима имитации
)

# Выполнение поискового запроса с анализом топ-результатов
search_result = parser.parse_search_results(
    "content optimization techniques",  # Поисковый запрос
    results_count=5,  # Количество результатов
    analyze_content=True  # Анализировать контент топ-результатов
)

# Обработка результатов
if search_result.get("success", False):
    search_data = search_result.get("search_data", {})
    results = search_data.get("results", [])
    
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. {result.get('title', '')}")
        print(f"   URL: {result.get('url', '')}")
        
        # Доступ к анализу содержимого
        if "detailed_analysis" in result:
            analysis = result.get("detailed_analysis", {})
            word_count = analysis.get("content", {}).get("word_count", 0)
            print(f"   Слов: {word_count}")
```

### Параллельный парсинг множества URL

```python
from seo_ai_models.parsers.unified.unified_parser import UnifiedParser

# Инициализация парсера с параллельной обработкой
parser = UnifiedParser(
    parallel_parsing=True,
    max_workers=5,  # Количество потоков
    auto_detect_spa=True  # Автоматически определять SPA-сайты
)

# Парсинг сайта с обходом страниц
site_result = parser.crawl_site(
    "https://example.com",
    max_pages=20,  # Максимальное количество страниц
    delay=1.0  # Задержка между запросами в секундах
)

# Обработка результатов
if site_result.get("success", False):
    site_data = site_result.get("site_data", {})
    pages = site_data.get("pages", {})
    
    print(f"Просканировано страниц: {len(pages)}")
    
    # Доступ к статистике по сайту
    stats = site_data.get("statistics", {})
    avg_words = stats.get("content", {}).get("avg_word_count", 0)
    print(f"Среднее количество слов на странице: {avg_words:.1f}")
```

## Зависимости

Для работы улучшенного парсера требуются следующие дополнительные библиотеки:

### Playwright

```bash
pip install playwright
python -m playwright install
```

### SpaCy/NLTK

```bash
pip install spacy nltk
python -m nltk.downloader stopwords
```

### Requests/BeautifulSoup

```bash
pip install requests beautifulsoup4 lxml
```

## Планы развития

- Добавление поддержки WebSocket для анализа приложений на основе веб-сокетов
- Расширенная интеграция с API поисковых систем (Bing, Yandex)
- Обнаружение и обход защиты от автоматизированного доступа
- Параллельная обработка сайтов с использованием распределенных вычислений

## Установка

### Базовая установка (основной функционал)

Включает numpy, scikit-learn, tiktoken для работы SEOAdvisor:

```bash
pip install seo-ai-models
```

### Установка с дополнительными возможностями

Для расширенного NLP анализа (spacy, nltk, gensim):
```bash
pip install seo-ai-models[ml]
```

Для парсинга SPA-сайтов и работы с поисковыми API:
```bash
pip install seo-ai-models[parsing]
```

Для веб-интерфейса и API:
```bash
pip install seo-ai-models[web]
```

Для разработки (включает инструменты тестирования и линтинга):
```bash
pip install seo-ai-models[dev]
```

Полная установка со всеми зависимостями:
```bash
pip install seo-ai-models[all]
```

### Установка из исходников

```bash
git clone https://github.com/Andrew821667/seo-ai-models.git
cd seo-ai-models
pip install -e ".[all]"
```

## Примеры использования

### Анализ контента

```python
from seo_ai_models.models.seo_advisor.advisor import SEOAdvisor

advisor = SEOAdvisor()
report = advisor.analyze_content("# Заголовок статьи\n\nЭто текст статьи...")
print(report)
```

### Парсинг и анализ сайта

```python
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
```

### Использование отдельных парсинг-компонентов

```python
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
```

## Демонстрационный скрипт

В проекте есть демонстрационный скрипт, показывающий использование компонентов парсинга:

```bash
python -m seo_ai_models.demo_parsing pipeline --mode site --url https://example.com --max-pages 5 --output analysis.json
```

## Разработка и тестирование

### Запуск всех тестов

```bash
PYTHONPATH=/path/to/seo-ai-models pytest tests/
```

### Запуск тестов для конкретного модуля

```bash
PYTHONPATH=/path/to/seo-ai-models pytest tests/parsers/
```
