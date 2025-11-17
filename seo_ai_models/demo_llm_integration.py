"""
Демонстрационный скрипт для LLM-интеграции.

Скрипт демонстрирует использование компонентов LLM-интеграции
для анализа и улучшения контента для LLM-поисковиков.
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional

from seo_ai_models.models.llm_integration.service.llm_service import LLMService
from seo_ai_models.models.llm_integration.service.prompt_generator import PromptGenerator
from seo_ai_models.models.llm_integration.service.multi_model_agent import MultiModelAgent
from seo_ai_models.models.llm_integration.service.cost_estimator import CostEstimator

from seo_ai_models.models.llm_integration.analyzers.llm_compatibility_analyzer import (
    LLMCompatibilityAnalyzer,
)
from seo_ai_models.models.llm_integration.analyzers.citability_scorer import CitabilityScorer
from seo_ai_models.models.llm_integration.analyzers.content_structure_enhancer import (
    ContentStructureEnhancer,
)
from seo_ai_models.models.llm_integration.analyzers.llm_eeat_analyzer import LLMEEATAnalyzer

from seo_ai_models.models.llm_integration.dimension_map.feature_importance_analyzer import (
    FeatureImportanceAnalyzer,
)
from seo_ai_models.models.llm_integration.dimension_map.semantic_structure_extractor import (
    SemanticStructureExtractor,
)


# Настраиваем логгирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def setup_services(openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Настраивает сервисы для LLM-интеграции.

    Args:
        openai_api_key: API ключ OpenAI (опционально)

    Returns:
        Dict[str, Any]: Словарь с сервисами
    """
    # Если API ключ не указан, пытаемся получить его из переменной окружения
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        logger.warning(
            "API ключ OpenAI не найден. Укажите его явно или через переменную окружения OPENAI_API_KEY"
        )
        return {}

    # Создаем сервисы
    llm_service = LLMService()
    prompt_generator = PromptGenerator()

    # Добавляем провайдера OpenAI
    try:
        llm_service.add_provider("openai", api_key=api_key, model="gpt-4o-mini")

        # Создаем вспомогательные сервисы
        multi_model_agent = MultiModelAgent(llm_service, prompt_generator)
        cost_estimator = CostEstimator()

        # Создаем анализаторы
        compatibility_analyzer = LLMCompatibilityAnalyzer(llm_service, prompt_generator)
        citability_scorer = CitabilityScorer(llm_service, prompt_generator)
        structure_enhancer = ContentStructureEnhancer(llm_service, prompt_generator)
        eeat_analyzer = LLMEEATAnalyzer(llm_service, prompt_generator)

        # Создаем инструменты для dimension_map
        feature_analyzer = FeatureImportanceAnalyzer(llm_service, prompt_generator)
        semantic_extractor = SemanticStructureExtractor(llm_service, prompt_generator)

        return {
            "llm_service": llm_service,
            "prompt_generator": prompt_generator,
            "multi_model_agent": multi_model_agent,
            "cost_estimator": cost_estimator,
            "compatibility_analyzer": compatibility_analyzer,
            "citability_scorer": citability_scorer,
            "structure_enhancer": structure_enhancer,
            "eeat_analyzer": eeat_analyzer,
            "feature_analyzer": feature_analyzer,
            "semantic_extractor": semantic_extractor,
        }
    except Exception as e:
        logger.error(f"Ошибка при настройке сервисов: {e}")
        return {}


def analyze_content(services: Dict[str, Any], content: str, analysis_type: str) -> None:
    """
    Анализирует контент с использованием указанного типа анализа.

    Args:
        services: Словарь с сервисами
        content: Текст для анализа
        analysis_type: Тип анализа
    """
    if not services:
        logger.error("Сервисы не настроены")
        return

    try:
        # Выбираем анализатор в зависимости от типа анализа
        if analysis_type == "compatibility":
            analyzer = services["compatibility_analyzer"]
            result = analyzer.analyze_compatibility(content)

            # Выводим результаты анализа
            logger.info("=== Анализ совместимости с LLM ===")
            logger.info(f"Общая оценка: {result.get('compatibility_scores', {}).get('overall', 0)}")

            for element, score in result.get("compatibility_scores", {}).items():
                if element != "overall":
                    logger.info(f"{element}: {score}")

            logger.info("\nРекомендации по улучшению:")
            for element, improvements in result.get("suggested_improvements", {}).items():
                if improvements:
                    logger.info(f"\n{element}:")
                    for improvement in improvements[:3]:  # Выводим только первые 3 рекомендации
                        logger.info(f"- {improvement}")

            logger.info(f"\nРезюме: {result.get('summary', '')}")

        elif analysis_type == "citability":
            analyzer = services["citability_scorer"]
            result = analyzer.score_citability(content)

            # Выводим результаты анализа
            logger.info("=== Оценка цитируемости в LLM ===")
            logger.info(f"Общая оценка цитируемости: {result.get('citability_score', 0)}")

            for factor, score in result.get("factor_scores", {}).items():
                logger.info(f"{factor}: {score}")

            logger.info("\nРекомендации по улучшению цитируемости:")
            for factor, improvements in result.get("suggested_improvements", {}).items():
                if improvements:
                    logger.info(f"\n{factor}:")
                    for improvement in improvements[:3]:  # Выводим только первые 3 рекомендации
                        logger.info(f"- {improvement}")

            logger.info(f"\nРезюме: {result.get('summary', '')}")

        elif analysis_type == "structure":
            analyzer = services["structure_enhancer"]
            result = analyzer.analyze_structure(content)

            # Выводим результаты анализа
            logger.info("=== Анализ структуры для LLM ===")
            logger.info(
                f"Общая оценка структуры: {result.get('structure_scores', {}).get('overall', 0)}"
            )

            for element, score in result.get("structure_scores", {}).items():
                if element != "overall":
                    logger.info(f"{element}: {score}")

            logger.info("\nРекомендации по улучшению структуры:")
            for element, improvements in result.get("suggested_improvements", {}).items():
                if improvements:
                    logger.info(f"\n{element}:")
                    for improvement in improvements[:3]:  # Выводим только первые 3 рекомендации
                        logger.info(f"- {improvement}")

            logger.info(f"\nРезюме: {result.get('summary', '')}")

        elif analysis_type == "eeat":
            analyzer = services["eeat_analyzer"]
            result = analyzer.analyze_eeat(content)

            # Выводим результаты анализа
            logger.info("=== Анализ E-E-A-T для LLM ===")
            logger.info(f"Общая оценка E-E-A-T: {result.get('eeat_scores', {}).get('overall', 0)}")

            for component, score in result.get("eeat_scores", {}).items():
                if component != "overall":
                    logger.info(f"{component}: {score}")

            logger.info("\nРекомендации по улучшению E-E-A-T:")
            for component, improvements in result.get("suggested_improvements", {}).items():
                if improvements:
                    logger.info(f"\n{component}:")
                    for improvement in improvements[:3]:  # Выводим только первые 3 рекомендации
                        logger.info(f"- {improvement}")

            logger.info(f"\nРезюме: {result.get('summary', '')}")

        elif analysis_type == "semantic":
            analyzer = services["semantic_extractor"]
            result = analyzer.extract_semantic_structure(content)

            # Выводим результаты анализа
            logger.info("=== Извлечение семантической структуры для LLM ===")
            logger.info(f"Основные темы: {len(result.get('topics', []))}")

            for i, topic in enumerate(result.get("topics", [])[:3]):  # Выводим только первые 3 темы
                logger.info(f"\nТема {i+1}: {topic.get('name', '')}")
                logger.info(f"Ключевые слова: {', '.join(topic.get('keywords', [])[:5])}")

            logger.info(f"\nОбщие ключевые слова: {', '.join(result.get('keywords', [])[:10])}")
            logger.info(f"\nРезюме: {result.get('content_summary', '')}")

        elif analysis_type == "feature_importance":
            analyzer = services["feature_analyzer"]
            result = analyzer.analyze_feature_importance()

            # Выводим результаты анализа
            logger.info("=== Анализ важности факторов для LLM ===")
            logger.info("Важность факторов:")

            # Сортируем факторы по важности
            sorted_factors = sorted(
                result.get("factor_importance", {}).items(), key=lambda x: x[1], reverse=True
            )

            for factor, importance in sorted_factors:
                logger.info(f"{factor}: {importance}")

            logger.info(f"\nНаиболее важные факторы: {', '.join(result.get('top_factors', []))}")
            logger.info(f"\nНаименее важные факторы: {', '.join(result.get('bottom_factors', []))}")
            logger.info(f"\nРезюме: {result.get('summary', '')}")

        else:
            logger.error(f"Неизвестный тип анализа: {analysis_type}")

    except Exception as e:
        logger.error(f"Ошибка при анализе контента: {e}")


def enhance_content(services: Dict[str, Any], content: str, enhancement_type: str) -> None:
    """
    Улучшает контент с использованием указанного типа улучшения.

    Args:
        services: Словарь с сервисами
        content: Текст для улучшения
        enhancement_type: Тип улучшения
    """
    if not services:
        logger.error("Сервисы не настроены")
        return

    try:
        # Выбираем улучшатель в зависимости от типа улучшения
        if enhancement_type == "structure":
            enhancer = services["structure_enhancer"]
            result = enhancer.enhance_structure(content)

            # Выводим результаты улучшения
            logger.info("=== Улучшение структуры для LLM ===")
            logger.info("Основные изменения:")

            changes = result.get("changes", {})

            logger.info(
                f"Заголовки: {changes.get('headings', {}).get('original_count', 0)} -> {changes.get('headings', {}).get('enhanced_count', 0)}"
            )
            logger.info(
                f"Абзацы: {changes.get('paragraphs', {}).get('original_count', 0)} -> {changes.get('paragraphs', {}).get('enhanced_count', 0)}"
            )
            logger.info(
                f"Списки: {changes.get('lists', {}).get('original_count', 0)} -> {changes.get('lists', {}).get('enhanced_count', 0)}"
            )
            logger.info(
                f"Выделения: {changes.get('highlights', {}).get('original_bold', 0) + changes.get('highlights', {}).get('original_italic', 0)} -> {changes.get('highlights', {}).get('enhanced_bold', 0) + changes.get('highlights', {}).get('enhanced_italic', 0)}"
            )

            logger.info("\nУлучшенный контент:")
            logger.info("=" * 50)
            print(result.get("enhanced_content", ""))
            logger.info("=" * 50)

        elif enhancement_type == "template":
            enhancer = services["structure_enhancer"]
            topic = input("Введите тему для шаблона: ")
            result = enhancer.generate_structural_template(topic)

            # Выводим результаты генерации шаблона
            logger.info(f"=== Шаблон структуры для темы '{topic}' ===")
            logger.info("Структура шаблона:")

            structure = result.get("structure_analysis", {})
            logger.info(
                f"Заголовки: {structure.get('headings_count', 0)} (H1: {structure.get('h1_count', 0)}, H2: {structure.get('h2_count', 0)}, H3: {structure.get('h3_count', 0)})"
            )
            logger.info(f"Секции: {structure.get('sections', 0)}")
            logger.info(
                f"Списки: {structure.get('bullet_lists_count', 0) + structure.get('numbered_lists_count', 0)}"
            )

            logger.info("\nШаблон:")
            logger.info("=" * 50)
            print(result.get("template", ""))
            logger.info("=" * 50)

        else:
            logger.error(f"Неизвестный тип улучшения: {enhancement_type}")

    except Exception as e:
        logger.error(f"Ошибка при улучшении контента: {e}")


def main() -> None:
    """
    Основная функция приложения.
    """
    parser = argparse.ArgumentParser(description="Демонстрация LLM-интеграции")
    parser.add_argument("--api-key", help="API ключ OpenAI")
    parser.add_argument(
        "--action",
        choices=["analyze", "enhance"],
        default="analyze",
        help="Действие (analyze или enhance)",
    )
    parser.add_argument("--type", help="Тип анализа или улучшения")
    parser.add_argument("--file", help="Путь к файлу с контентом")

    args = parser.parse_args()

    # Настраиваем сервисы
    services = setup_services(args.api_key)

    if not services:
        logger.error("Не удалось настроить сервисы")
        return

    # Определяем типы анализа и улучшения
    analysis_types = [
        "compatibility",
        "citability",
        "structure",
        "eeat",
        "semantic",
        "feature_importance",
    ]
    enhancement_types = ["structure", "template"]

    # Запрашиваем тип анализа или улучшения, если не указан
    action_type = args.type

    if args.action == "analyze":
        if not action_type or action_type not in analysis_types:
            print(f"Выберите тип анализа ({', '.join(analysis_types)}):")
            action_type = input("> ").strip().lower()

            if action_type not in analysis_types:
                logger.error(f"Неизвестный тип анализа: {action_type}")
                return
    elif args.action == "enhance":
        if not action_type or action_type not in enhancement_types:
            print(f"Выберите тип улучшения ({', '.join(enhancement_types)}):")
            action_type = input("> ").strip().lower()

            if action_type not in enhancement_types:
                logger.error(f"Неизвестный тип улучшения: {action_type}")
                return

    # Загружаем контент
    content = ""

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Ошибка при чтении файла: {e}")
            return
    else:
        if action_type != "template":  # Для шаблона контент не нужен
            print(
                "Введите контент (завершите ввод комбинацией Ctrl+D в Unix или Ctrl+Z в Windows):"
            )
            try:
                while True:
                    line = input()
                    content += line + "\n"
            except EOFError:
                pass

    # Выполняем действие
    if args.action == "analyze":
        analyze_content(services, content, action_type)
    elif args.action == "enhance":
        enhance_content(services, content, action_type)


if __name__ == "__main__":
    main()
