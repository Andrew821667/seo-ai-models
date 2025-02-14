
from models.seo_advisor.content_analyzer import ContentAnalyzer

# Создаем тестовый контент
test_content = """
<h1>Главный заголовок</h1>
<meta name="description" content="Описание страницы">
<title>Заголовок страницы</title>

<h2>Подзаголовок 1</h2>
Это тестовый контент для проверки работы анализатора. 
Ключевые слова: тест, анализ, контент.

<h2>Подзаголовок 2</h2>
Дополнительный текст для тестирования.
"""

# Создаем список ключевых слов
keywords = ['тест', 'анализ', 'контент']

# Инициализируем анализатор
analyzer = ContentAnalyzer()

# Проводим анализ
metrics = analyzer.analyze_text(test_content, keywords)

# Выводим результаты
print(f"Плотность ключевых слов: {metrics.keyword_density:.2%}")
print(f"Оценка читаемости: {metrics.readability_score:.2f}")
print(f"Структура заголовков: {metrics.header_structure}")
print(f"Оценка мета-тегов: {metrics.meta_tags_score}")
