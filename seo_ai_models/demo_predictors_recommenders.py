"""
Демонстрационный скрипт для улучшенных предикторов и рекомендаций.

Скрипт демонстрирует использование компонентов для:
1. Предсказания ранжирования в LLM-поисковиках
2. Генерации рекомендаций, оптимизированных для обоих типов поиска
3. Расчета ROI от внедрения рекомендаций
4. Создания плана действий с приоритизацией по эффективности
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional

from seo_ai_models.models.llm_integration.service.llm_service import LLMService
from seo_ai_models.models.llm_integration.service.prompt_generator import PromptGenerator
from seo_ai_models.models.llm_integration.service.cost_estimator import CostEstimator
from seo_ai_models.models.llm_integration.predictors_recommenders.llm_rank_predictor import LLMRankPredictor
from seo_ai_models.models.llm_integration.predictors_recommenders.hybrid_recommender import HybridRecommender
from seo_ai_models.models.llm_integration.predictors_recommenders.roi_calculator import ROICalculator
from seo_ai_models.models.llm_integration.predictors_recommenders.prioritized_action_plan import PrioritizedActionPlan

# Настраиваем логгирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_services(openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Настраивает сервисы для демонстрации.
    
    Args:
        openai_api_key: API ключ OpenAI (опционально)
        
    Returns:
        Dict[str, Any]: Словарь с сервисами
    """
    # Если API ключ не указан, пытаемся получить его из переменной окружения
    if not openai_api_key:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Создаем экземпляры базовых сервисов
    llm_service = LLMService(openai_api_key=openai_api_key)
    prompt_generator = PromptGenerator()
    cost_estimator = CostEstimator()
    
    # Создаем экземпляры компонентов
    rank_predictor = LLMRankPredictor(llm_service, prompt_generator)
    hybrid_recommender = HybridRecommender(llm_service, prompt_generator)
    roi_calculator = ROICalculator(llm_service, prompt_generator, rank_predictor)
    prioritized_action_plan = PrioritizedActionPlan(llm_service, prompt_generator, roi_calculator, hybrid_recommender)
    
    # Возвращаем словарь с сервисами
    return {
        "llm_service": llm_service,
        "prompt_generator": prompt_generator,
        "cost_estimator": cost_estimator,
        "rank_predictor": rank_predictor,
        "hybrid_recommender": hybrid_recommender,
        "roi_calculator": roi_calculator,
        "prioritized_action_plan": prioritized_action_plan
    }

def demonstrate_rank_predictor(services: Dict[str, Any]) -> None:
    """
    Демонстрирует работу LLMRankPredictor.
    
    Args:
        services: Словарь с сервисами
    """
    logger.info("=== Демонстрация LLMRankPredictor ===")
    
    rank_predictor = services["rank_predictor"]
    
    # Пример контента для анализа
    content = """
    # Руководство по SEO-оптимизации для LLM-поисковиков
    
    Поисковые системы, основанные на больших языковых моделях (LLM), становятся все более популярными. В отличие от традиционных поисковиков, LLM-поисковики не просто показывают список результатов, а генерируют ответы на основе найденной информации.
    
    ## Что такое LLM-поисковики
    
    LLM-поисковики, такие как Perplexity, Claude Search, You.com и другие, используют большие языковые модели для генерации ответов на запросы пользователей. Они анализируют контент из различных источников и формируют связный ответ.
    
    ## Особенности оптимизации для LLM-поисковиков
    
    ### 1. Структура и ясность контента
    
    Для LLM-поисковиков особенно важна четкая структура контента. Используйте:
    - Информативные заголовки H1, H2, H3
    - Короткие, ясные параграфы
    - Маркированные и нумерованные списки
    - Таблицы для сравнения данных
    
    ### 2. Фактическая точность и цитируемость
    
    LLM-модели стремятся предоставлять точную информацию. Повысьте вероятность цитирования вашего контента:
    - Включайте конкретные факты и статистику
    - Указывайте источники информации
    - Используйте актуальные данные
    - Приводите примеры из практики
    
    ### 3. Полнота и экспертность
    
    Создавайте контент, демонстрирующий глубокое понимание темы:
    - Рассматривайте вопрос с разных сторон
    - Отвечайте на сопутствующие вопросы
    - Приводите экспертные мнения
    - Объясняйте сложные концепции простым языком
    
    ## Практические рекомендации
    
    1. **Оптимизируйте для конкретных запросов** - определите, на какие вопросы должен отвечать ваш контент
    2. **Регулярно обновляйте информацию** - LLM ценят актуальность
    3. **Используйте таблицы и визуализации** - они повышают информативность
    4. **Проверяйте видимость в LLM-поисковиках** - анализируйте, как часто ваш контент цитируется
    
    Следуя этим рекомендациям, вы сможете повысить видимость и цитируемость вашего контента в новом поколении поисковых систем.
    """
    
    # Пример конкурентов
    competitors = [
        {
            "id": "competitor1",
            "name": "SEO Expert Blog",
            "content": "# Оптимизация для LLM-поисковиков\n\nПоисковые системы на базе LLM становятся все более популярными. Вот как оптимизировать контент для них:\n\n1. Используйте четкую структуру\n2. Добавляйте фактические данные\n3. Демонстрируйте экспертность"
        },
        {
            "id": "competitor2",
            "name": "Digital Marketing Insights",
            "content": "# LLM SEO: Новый фронтир\n\nСистемы поиска на базе LLM меняют правила игры в SEO. Ключевые стратегии:\n\n- Структурированный контент с заголовками\n- Фактическая информация с источниками\n- Ответы на конкретные вопросы"
        }
    ]
    
    # Демонстрационный режим - пропускаем реальные запросы
    logger.info("В демонстрационном режиме пропускаем реальные запросы LLMRankPredictor.")
    logger.info("LLMRankPredictor предсказывает ранжирование контента в LLM-поисковиках.")
    logger.info("Основные методы:")
    logger.info("- predict_ranking: Предсказывает ранжирование контента")
    logger.info("- predict_impact_of_changes: Предсказывает влияние изменений на ранжирование")
    
    # Модификация контента для демонстрации
    improved_content = content + "\n\n## Примеры успешной оптимизации\n\nРассмотрим несколько кейсов успешной оптимизации контента для LLM-поисковиков:\n\n1. **Кейс компании X** - увеличение видимости на 150% за счет внедрения структурированных данных\n2. **Кейс сайта Y** - рост цитируемости на 200% благодаря добавлению исследований и статистики"
    
    logger.info("\nПри прогнозировании влияния изменений контента, LLMRankPredictor учитывает следующие факторы:")
    logger.info("- Цитируемость (вероятность цитирования контента в ответах LLM)")
    logger.info("- E-E-A-T (опыт, экспертиза, авторитетность, надежность)")
    logger.info("- Структура контента (заголовки, списки, таблицы)")
    logger.info("- Информационная ценность и уникальность")

def demonstrate_hybrid_recommender(services: Dict[str, Any]) -> None:
    """
    Демонстрирует работу HybridRecommender.
    
    Args:
        services: Словарь с сервисами
    """
    logger.info("\n=== Демонстрация HybridRecommender ===")
    
    hybrid_recommender = services["hybrid_recommender"]
    
    # Пример контента для анализа
    content = """
    # Преимущества облачных вычислений для бизнеса
    
    Облачные вычисления становятся все более важными для современных предприятий. Они предлагают множество преимуществ, от снижения затрат до повышения гибкости.
    
    ## Экономическая эффективность
    
    Облачные решения позволяют значительно снизить капитальные затраты на ИТ-инфраструктуру. Вместо покупки серверов, компании могут арендовать вычислительные мощности по мере необходимости.
    
    ## Масштабируемость
    
    Облачные сервисы легко масштабируются в зависимости от потребностей бизнеса. Вы можете увеличивать или уменьшать ресурсы в режиме реального времени.
    
    ## Доступность
    
    Доступ к облачным сервисам возможен из любой точки мира при наличии интернет-соединения. Это особенно важно для удаленной работы и распределенных команд.
    
    ## Надежность и безопасность
    
    Ведущие облачные провайдеры обеспечивают высокий уровень надежности и безопасности данных. Они используют передовые технологии защиты и регулярное резервное копирование.
    """
    
    # Демонстрационный режим - пропускаем реальные запросы
    logger.info("В демонстрационном режиме пропускаем реальные запросы HybridRecommender.")
    logger.info("HybridRecommender генерирует рекомендации, оптимизированные как для традиционных, так и для LLM-поисковиков.")
    logger.info("Основные методы:")
    logger.info("- generate_recommendations: Генерирует рекомендации для обоих типов поиска")
    logger.info("- simulate_recommendation_impact: Симулирует влияние рекомендации на контент")
    
    logger.info("\nHybridRecommender поддерживает три режима баланса рекомендаций:")
    logger.info("- traditional: Приоритет рекомендациям для традиционных поисковиков (80/20)")
    logger.info("- llm: Приоритет рекомендациям для LLM-поисковиков (20/80)")
    logger.info("- balanced: Равный приоритет для обоих типов (50/50)")
    
    logger.info("\nПример рекомендаций, которые может генерировать HybridRecommender:")
    logger.info("1. [Traditional SEO] Добавьте ключевые слова 'облачные вычисления' в заголовок H1")
    logger.info("2. [Traditional SEO] Оптимизируйте мета-описание страницы")
    logger.info("3. [LLM Optimization] Добавьте конкретные примеры использования облачных вычислений")
    logger.info("4. [LLM Optimization] Включите статистику экономии затрат при переходе на облако")
    logger.info("5. [Hybrid] Создайте структурированные блоки 'вопрос-ответ' для повышения видимости в обоих типах поиска")

def demonstrate_roi_calculator(services: Dict[str, Any]) -> None:
    """
    Демонстрирует работу ROICalculator.
    
    Args:
        services: Словарь с сервисами
    """
    logger.info("\n=== Демонстрация ROICalculator ===")
    
    roi_calculator = services["roi_calculator"]
    
    # Пример рекомендаций
    recommendations = [
        {
            "type": "traditional_seo",
            "category": "on_page",
            "recommendation": "Добавьте ключевые слова в заголовок H1",
            "priority": 4,
            "impact": 3,
            "implementation_difficulty": 1
        },
        {
            "type": "llm_optimization",
            "category": "citability",
            "recommendation": "Добавьте статистику и ссылки на исследования",
            "priority": 5,
            "impact": 4,
            "implementation_difficulty": 2
        },
        {
            "type": "llm_optimization",
            "category": "eeat",
            "recommendation": "Добавьте информацию об авторе и его экспертизе",
            "priority": 4,
            "impact": 5,
            "implementation_difficulty": 1
        },
        {
            "type": "traditional_seo",
            "category": "content",
            "recommendation": "Расширьте контент до 1500+ слов",
            "priority": 3,
            "impact": 4,
            "implementation_difficulty": 3
        },
        {
            "type": "hybrid",
            "category": "content_structure",
            "recommendation": "Добавьте FAQ-секцию с вопросами и ответами",
            "priority": 4,
            "impact": 5,
            "implementation_difficulty": 2
        }
    ]
    
    # Пример бизнес-данных
    business_data = {
        "traditional_traffic": 500,
        "llm_traffic": 200,
        "traditional_conversion_rate": 0.02,
        "llm_conversion_rate": 0.025,
        "traditional_conversion_value": 15000,
        "llm_conversion_value": 18000
    }
    
    # Демонстрационный режим - пропускаем реальные запросы
    logger.info("В демонстрационном режиме пропускаем реальные запросы ROICalculator.")
    logger.info("ROICalculator рассчитывает потенциальный возврат инвестиций от внедрения рекомендаций.")
    logger.info("Основные методы:")
    logger.info("- calculate_roi: Рассчитывает ROI от внедрения рекомендаций")
    logger.info("- calculate_detailed_roi: Рассчитывает детальный ROI с разделением на традиционные и LLM-поисковики")
    logger.info("- estimate_baseline_traffic: Оценивает базовый трафик для запроса и позиции")
    
    logger.info("\nПример расчета ROI для рекомендаций:")
    logger.info("1. Оценка текущего трафика: 700 визитов (500 из традиционных поисковиков, 200 из LLM)")
    logger.info("2. Прогноз увеличения трафика: +350 визитов (+150 из традиционных, +200 из LLM)")
    logger.info("3. Конверсия: 2% для традиционных, 2.5% для LLM")
    logger.info("4. Средний чек: 15 000 руб. для традиционных, 18 000 руб. для LLM")
    logger.info("5. Стоимость внедрения рекомендаций: 120 000 руб.")
    logger.info("6. Дополнительная выручка: 1 035 000 руб. в год")
    logger.info("7. ROI: 762.5%")
    logger.info("8. Срок окупаемости: 1.4 месяца")

def demonstrate_prioritized_action_plan(services: Dict[str, Any]) -> None:
    """
    Демонстрирует работу PrioritizedActionPlan.
    
    Args:
        services: Словарь с сервисами
    """
    logger.info("\n=== Демонстрация PrioritizedActionPlan ===")
    
    prioritized_action_plan = services["prioritized_action_plan"]
    
    # Демонстрационный режим - пропускаем реальные запросы
    logger.info("В демонстрационном режиме пропускаем реальные запросы PrioritizedActionPlan.")
    logger.info("PrioritizedActionPlan создает план действий с приоритизацией по эффективности.")
    logger.info("Основные методы:")
    logger.info("- create_action_plan: Создает план действий на основе рекомендаций")
    logger.info("- generate_resource_allocation: Генерирует план распределения ресурсов")
    
    logger.info("\nПлан действий включает следующие фазы:")
    logger.info("1. Немедленные действия (1-2 недели) - быстрые задачи с высоким ROI")
    logger.info("2. Краткосрочные действия (1-2 месяца) - задачи средней сложности с хорошим ROI")
    logger.info("3. Среднесрочные действия (3-6 месяцев) - комплексные задачи с высоким ROI")
    logger.info("4. Долгосрочные действия (6-12 месяцев) - стратегические задачи с высоким долгосрочным ROI")
    
    logger.info("\nПример приоритизации задач:")
    logger.info("- Высокий приоритет: Добавление информации об авторе (+5 к E-E-A-T, сложность: 1, ROI: 1200%)")
    logger.info("- Средний приоритет: Добавление FAQ-секции (+4 к цитируемости, сложность: 2, ROI: 450%)")
    logger.info("- Низкий приоритет: Расширение контента до 1500+ слов (сложность: 3, ROI: 210%)")
    
    logger.info("\nПример распределения ресурсов:")
    logger.info("- SEO-специалист: 40 часов")
    logger.info("- LLM-специалист: 35 часов")
    logger.info("- Копирайтер: 25 часов")
    logger.info("- Разработчик: 10 часов")

def main():
    parser = argparse.ArgumentParser(description="Демонстрация улучшенных предикторов и рекомендаций")
    parser.add_argument("--api_key", help="API ключ OpenAI")
    parser.add_argument("--demo", choices=["all", "rank", "recommendations", "roi", "plan"],
                      default="all", help="Какую демонстрацию запустить")
    
    args = parser.parse_args()
    
    # Настраиваем сервисы
    services = setup_services(args.api_key)
    
    # Запускаем выбранную демонстрацию
    if args.demo in ["all", "rank"]:
        demonstrate_rank_predictor(services)
    
    if args.demo in ["all", "recommendations"]:
        demonstrate_hybrid_recommender(services)
    
    if args.demo in ["all", "roi"]:
        demonstrate_roi_calculator(services)
    
    if args.demo in ["all", "plan"]:
        demonstrate_prioritized_action_plan(services)
    
    logger.info("\n=== Демонстрация завершена ===")

if __name__ == "__main__":
    main()
