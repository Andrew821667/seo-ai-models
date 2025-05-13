# LLM-интеграция для SEO AI Models

Модуль LLM-интеграции предоставляет функционал для анализа и оптимизации контента под LLM-поисковики и генеративные модели.

## Компоненты

### Service

- **LLMService**: Универсальный сервис для взаимодействия с различными LLM API
- **PromptGenerator**: Система создания эффективных промптов для разных LLM
- **MultiModelAgent**: Система умного выбора и комбинирования результатов разных LLM
- **CostEstimator**: Анализатор затрат на API-вызовы и их оптимизация

### Analyzers

- **LLMCompatibilityAnalyzer**: Анализ совместимости контента с требованиями LLM
- **CitabilityScorer**: Оценка вероятности цитирования контента в LLM-ответах
- **ContentStructureEnhancer**: Улучшение структуры для повышения цитируемости
- **LLMEEATAnalyzer**: Расширение E-E-A-T для LLM

### Dimension Map

- **LLMDimensionMap**: Модель для выявления факторов, важных для LLM
- **SemanticStructureExtractor**: Извлечение семантической структуры для анализа LLM
- **FeatureImportanceAnalyzer**: Анализ важности факторов для LLM-оптимизации

## Использование

### Анализ совместимости с LLM

```python
from seo_ai_models.models.llm_integration.service.llm_service import LLMService
from seo_ai_models.models.llm_integration.service.prompt_generator import PromptGenerator
from seo_ai_models.models.llm_integration.analyzers.llm_compatibility_analyzer import LLMCompatibilityAnalyzer

# Инициализация сервисов
llm_service = LLMService()
llm_service.add_provider("openai", api_key="your-api-key", model="gpt-4o-mini")
prompt_generator = PromptGenerator()

# Создание анализатора
analyzer = LLMCompatibilityAnalyzer(llm_service, prompt_generator)

# Анализ контента
result = analyzer.analyze_compatibility(content)

# Использование результатов
print(f"Общая оценка совместимости: {result['compatibility_scores']['overall']}")
for factor, score in result['compatibility_scores'].items():
    if factor != "overall":
        print(f"{factor}: {score}")
```

### Оценка цитируемости контента

```python
from seo_ai_models.models.llm_integration.analyzers.citability_scorer import CitabilityScorer

# Создание оценщика цитируемости
scorer = CitabilityScorer(llm_service, prompt_generator)

# Оценка цитируемости
result = scorer.score_citability(content, category="tech")

# Использование результатов
print(f"Общая оценка цитируемости: {result['citability_score']}")
for factor, score in result['factor_scores'].items():
    print(f"{factor}: {score}")
```

### Улучшение структуры контента

```python
from seo_ai_models.models.llm_integration.analyzers.content_structure_enhancer import ContentStructureEnhancer

# Создание улучшителя структуры
enhancer = ContentStructureEnhancer(llm_service, prompt_generator)

# Улучшение структуры
result = enhancer.enhance_structure(content, content_type="article")

# Использование результатов
print(result['enhanced_content'])
```
