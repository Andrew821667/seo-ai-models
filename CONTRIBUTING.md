# Руководство по внесению вклада в SEO AI Models

Спасибо за интерес к проекту SEO AI Models! Это руководство поможет вам внести свой вклад в развитие проекта.

## 📋 Содержание

- [Код поведения](#код-поведения)
- [Начало работы](#начало-работы)
- [Процесс разработки](#процесс-разработки)
- [Стандарты кода](#стандарты-кода)
- [Тестирование](#тестирование)
- [Коммиты и Pull Requests](#коммиты-и-pull-requests)

## 🤝 Код поведения

- Будьте уважительны ко всем участникам проекта
- Конструктивная критика приветствуется
- Фокусируйтесь на улучшении проекта

## 🚀 Начало работы

### Подготовка окружения

```bash
# Клонируйте репозиторий
git clone https://github.com/yourusername/seo-ai-models.git
cd seo-ai-models

# Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установите зависимости
pip install -r requirements.txt
pip install -r web_requirements.txt  # для web-компонентов

# Установите Playwright (для SPA-краулеров)
python -m playwright install
```

### Структура проекта

```
seo-ai-models/
├── seo_ai_models/          # Основной пакет
│   ├── common/             # Общие утилиты и конфиги
│   ├── data/               # Модели данных
│   ├── models/             # AI/ML модели и анализаторы
│   ├── parsers/            # Парсеры и краулеры
│   └── web/                # Web API и дашборд
├── tests/                  # Тесты
├── examples/               # Примеры использования
└── docs/                   # Документация
```

## 💻 Процесс разработки

### 1. Создание feature branch

```bash
git checkout -b feature/your-feature-name
# или
git checkout -b fix/bug-description
```

### 2. Внесение изменений

- Пишите чистый, читаемый код
- Следуйте стандартам кода (см. ниже)
- Добавляйте docstrings ко всем публичным методам
- Пишите тесты для новой функциональности

### 3. Тестирование

```bash
# Запустите все тесты
pytest tests/

# Запустите конкретный тест
pytest tests/unit/test_specific.py

# С покрытием
pytest --cov=seo_ai_models tests/
```

## 📝 Стандарты кода

### Python Code Style

- Следуйте **PEP 8**
- Используйте **type hints** где возможно
- Максимальная длина строки: **100 символов**
- Используйте **f-strings** для форматирования

### Naming Conventions

```python
# Классы: PascalCase
class ContentAnalyzer:
    pass

# Функции и методы: snake_case
def extract_content(html: str) -> Dict[str, Any]:
    pass

# Константы: UPPER_CASE
MAX_PAGES = 100

# Приватные методы: _leading_underscore
def _internal_method(self):
    pass
```

### Docstrings

Используйте Google Style docstrings:

```python
def parse_url(url: str, options: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Парсит URL и извлекает контент.

    Args:
        url: URL для парсинга
        options: Дополнительные опции парсинга

    Returns:
        Dict[str, Any]: Словарь с извлеченными данными

    Raises:
        ValueError: Если URL невалиден
        RequestException: Если запрос не удался

    Example:
        >>> result = parse_url("https://example.com")
        >>> print(result['title'])
        'Example Domain'
    """
    pass
```

### Error Handling

```python
# ✅ ХОРОШО: Конкретный exception с логированием
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {str(e)}")
    raise

# ❌ ПЛОХО: Пустой except
try:
    result = risky_operation()
except:
    pass
```

### Abstract Methods

```python
from abc import ABC, abstractmethod

class BaseParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> Dict[str, Any]:
        """
        Парсит контент.

        Raises:
            NotImplementedError: Должен быть реализован в подклассе
        """
        raise NotImplementedError("Subclasses must implement parse() method")
```

## 🧪 Тестирование

### Структура тестов

```python
import pytest
from seo_ai_models.parsers import ContentExtractor

class TestContentExtractor:
    """Тесты для ContentExtractor."""

    @pytest.fixture
    def extractor(self):
        """Фикстура extractor."""
        return ContentExtractor()

    def test_extract_title(self, extractor):
        """Тест извлечения заголовка."""
        html = "<html><title>Test</title></html>"
        result = extractor.extract_content(html)
        assert result['title'] == "Test"

    def test_empty_html(self, extractor):
        """Тест обработки пустого HTML."""
        result = extractor.extract_content("")
        assert 'error' in result
```

### Минимальное покрытие

- **Unit tests**: Покрытие > 60%
- **Integration tests**: Критические пути
- Все новые функции должны иметь тесты

## 📦 Коммиты и Pull Requests

### Формат коммитов

Используйте [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: Новая функциональность
- `fix`: Исправление бага
- `docs`: Документация
- `style`: Форматирование кода
- `refactor`: Рефакторинг
- `test`: Тесты
- `chore`: Обслуживание

**Примеры:**

```bash
feat(parsers): добавлена поддержка WebSocket анализа

Реализован новый модуль для анализа WebSocket соединений
в SPA-приложениях. Поддерживает перехват сообщений и
извлечение структурированных данных.

Closes #123
```

```bash
fix(api): исправлена ошибка 500 при DELETE /projects/{id}

DELETE endpoint возвращал None вместо HTTP response.
Теперь возвращает HTTP 501 Not Implemented с понятным сообщением.
```

### Pull Request Process

1. **Перед созданием PR:**
   - Убедитесь, что все тесты проходят
   - Обновите документацию если нужно
   - Проверьте code style

2. **Создание PR:**
   - Заполните шаблон PR (если есть)
   - Добавьте понятное описание изменений
   - Свяжите с соответствующим issue

3. **После создания PR:**
   - Отвечайте на комментарии ревьюеров
   - Вносите запрошенные изменения
   - Держите PR актуальным (rebase if needed)

### PR Title Format

```
<type>(<scope>): <short description>
```

Примеры:
- `feat(llm): add GPT-4 integration`
- `fix(crawlers): resolve timeout in SPA crawler`
- `docs(readme): update installation instructions`

## 🐛 Reporting Bugs

При создании issue о баге укажите:

- **Версия Python** и зависимостей
- **Шаги для воспроизведения**
- **Ожидаемое поведение**
- **Фактическое поведение**
- **Traceback** (если есть)

## 💡 Feature Requests

При предложении новой функциональности:

- Опишите use case
- Объясните, почему это важно
- Предложите возможную реализацию
- Обсудите альтернативы

## 📚 Дополнительные ресурсы

- [Python PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Pytest Documentation](https://docs.pytest.org/)

## 📞 Контакты

- **Issues**: [GitHub Issues](https://github.com/yourusername/seo-ai-models/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/seo-ai-models/discussions)

---

**Спасибо за ваш вклад в SEO AI Models! 🚀**
