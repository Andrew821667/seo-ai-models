"""Генератор рекомендаций для SEO-оптимизации."""

from typing import Dict, List, Union
from dataclasses import dataclass


@dataclass
class SuggestionPriority:
    task: str
    impact: float
    effort: float
    priority_score: float


class Suggester:
    """Генератор рекомендаций по улучшению SEO"""

    def __init__(self):
        self.priority_weights = {
            "content_length": 0.25,
            "keyword_density": 0.20,
            "readability": 0.15,
            "header_structure": 0.15,
            "meta_tags": 0.10,
            "multimedia": 0.08,
            "internal_linking": 0.07,
        }

    def generate_suggestions(
        self,
        basic_recommendations: Dict[str, List[str]],
        feature_scores: Dict[str, float],
        industry: str,
    ) -> Dict[str, List[str]]:
        """
        Генерация расширенных рекомендаций
        """
        enhanced_suggestions = {}

        # Улучшаем базовые рекомендации в зависимости от индустрии
        industry_specific = self._get_industry_specific_suggestions(industry)

        for feature, suggestions in basic_recommendations.items():
            score = feature_scores.get(feature, 0)
            enhanced = suggestions.copy()

            # Добавляем индустри-специфичные рекомендации
            if feature in industry_specific:
                enhanced.extend(industry_specific[feature])

            # Добавляем конкретные действия в зависимости от скора
            if score < 0.3:
                enhanced.extend(self._get_critical_suggestions(feature))
            elif score < 0.6:
                enhanced.extend(self._get_improvement_suggestions(feature))

            enhanced_suggestions[feature] = enhanced

        return enhanced_suggestions

    def prioritize_tasks(
        self,
        suggestions: Dict[str, List[str]],
        feature_scores: Dict[str, float],
        weighted_scores: Dict[str, float],
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Улучшенная приоритизация задач с исправленным расчетом
        """
        priorities = []

        for feature, tasks in suggestions.items():
            feature_score = feature_scores.get(feature, 0)
            weight = self.priority_weights.get(feature, 0.1)

            # Если нет weighted_scores или weighted_scores пустой, создаем значение по умолчанию
            weighted_score = weighted_scores.get(feature, feature_score * weight)
            if weighted_score == 0:
                weighted_score = feature_score * weight  # Расчет на основе веса из priority_weights

            for task in tasks:
                # Оцениваем сложность задачи
                effort = self._estimate_task_effort(task)

                # Улучшенный расчет импакта (влияния)
                impact = (1 - feature_score) * weight * 2  # Увеличиваем влияние

                # Защита от нулевых и отрицательных значений
                impact = max(impact, 0.01)  # Обеспечиваем минимальное значение > 0

                # Бонус за критически важные метрики
                if feature in ["content_length", "keyword_density"] and feature_score < 0.3:
                    impact *= 1.5

                # Считаем приоритет с учетом срочности
                # Используем ненулевое значение для effort
                effort = max(effort, 0.1)  # Гарантируем, что делитель > 0
                priority_score = (impact * 1.5) / effort

                # Дополнительное масштабирование, чтобы приоритеты не были слишком низкими
                if feature_score < 0.3:
                    priority_score *= 1.3  # Бонус за критические показатели

                priorities.append(
                    {
                        "task": task,
                        "impact": impact,
                        "effort": effort,
                        "priority_score": priority_score,
                        "feature": feature,
                    }
                )

        # Сортируем по приоритету
        return sorted(priorities, key=lambda x: x["priority_score"], reverse=True)

    def _get_industry_specific_suggestions(self, industry: str) -> Dict[str, List[str]]:
        """
        Получение рекомендаций специфичных для индустрии
        """
        suggestions = {
            "blog": {
                "content_length": [
                    "Добавьте примеры из практики",
                    "Включите экспертные мнения",
                    "Разбейте текст на тематические секции",
                ],
                "readability": [
                    "Используйте больше подзаголовков",
                    "Добавьте маркированные списки",
                    "Включите определения терминов",
                ],
            },
            "scientific_blog": {
                "content_length": [
                    "Добавьте методологию исследования",
                    "Включите статистические данные",
                    "Опишите ограничения исследования",
                ],
                "readability": [
                    "Добавьте графики и диаграммы",
                    "Включите сравнительные таблицы",
                    "Объясните сложные термины",
                ],
            },
            "ecommerce": {
                "content_length": [
                    "Расширьте описание характеристик",
                    "Добавьте сравнение с аналогами",
                    "Включите отзывы пользователей",
                ],
                "multimedia": [
                    "Добавьте больше фотографий продукта",
                    "Включите видео-обзор",
                    "Добавьте инфографику",
                ],
            },
            "finance": {
                "content_length": [
                    "Добавьте экспертные финансовые рекомендации",
                    "Включите примеры расчетов",
                    "Добавьте разбор типичных кейсов",
                ],
                "trust": [
                    "Укажите источники финансовых данных",
                    "Добавьте отказ от ответственности",
                    "Укажите квалификацию финансовых экспертов",
                ],
            },
            "health": {
                "content_length": [
                    "Добавьте актуальные медицинские исследования",
                    "Включите мнения практикующих врачей",
                    "Разъясните сложные медицинские термины",
                ],
                "trust": [
                    "Укажите медицинские источники",
                    "Включите предупреждение о консультации с врачом",
                    "Укажите дату последнего обновления информации",
                ],
            },
        }

        return suggestions.get(industry, {})

    def _get_critical_suggestions(self, feature: str) -> List[str]:
        """
        Получение критических рекомендаций для низких показателей
        """
        critical_suggestions = {
            "content_length": [
                "СРОЧНО: Увеличьте объем контента минимум в 2 раза",
                "КРИТИЧНО: Добавьте детальное описание каждого аспекта темы",
            ],
            "keyword_density": [
                "СРОЧНО: Добавьте ключевые слова в заголовки H1 и H2",
                "КРИТИЧНО: Увеличьте частоту использования ключевых слов минимум до 1-2%",
            ],
            "readability": [
                "СРОЧНО: Упростите сложные предложения, уменьшив их длину до 20 слов",
                "КРИТИЧНО: Разбейте абзацы на более короткие (2-3 предложения)",
            ],
            "header_structure": [
                "СРОЧНО: Добавьте заголовок H1 с главным ключевым словом",
                "КРИТИЧНО: Структурируйте контент с использованием H2 и H3 заголовков",
            ],
            "meta_tags": [
                "СРОЧНО: Добавьте заполненные мета-теги title и description",
                "КРИТИЧНО: Добавьте alt-теги ко всем изображениям",
            ],
            "multimedia": [
                "СРОЧНО: Добавьте как минимум 3-5 релевантных изображений",
                "КРИТИЧНО: Включите видео или инфографику по теме",
            ],
            "internal_linking": [
                "СРОЧНО: Добавьте минимум 3 внутренние ссылки",
                "КРИТИЧНО: Используйте ключевые слова в анкорах ссылок",
            ],
        }

        return critical_suggestions.get(feature, [])

    def _get_improvement_suggestions(self, feature: str) -> List[str]:
        """
        Получение рекомендаций по улучшению средних показателей
        """
        improvement_suggestions = {
            "content_length": [
                "Добавьте дополнительные примеры по каждому пункту",
                "Расширьте описание ключевых моментов с данными из исследований",
            ],
            "keyword_density": [
                "Добавьте больше LSI-ключевых слов в текст",
                "Используйте синонимы ключевых слов для разнообразия",
            ],
            "readability": [
                "Добавьте дополнительные подзаголовки через каждые 300 слов",
                "Используйте маркированные списки для перечислений",
            ],
            "header_structure": [
                "Проверьте иерархию заголовков (H1 → H2 → H3)",
                "Добавьте описательные H2 заголовки для каждого раздела",
            ],
            "meta_tags": [
                "Оптимизируйте title с учетом ключевых слов",
                "Создайте привлекательный description длиной до 160 символов",
            ],
            "multimedia": [
                "Добавьте подписи к изображениям с ключевыми словами",
                "Улучшите качество и релевантность медиа-материалов",
            ],
            "internal_linking": [
                "Добавьте ссылки на ваши наиболее авторитетные страницы",
                'Создайте секцию "Рекомендуемые материалы"',
            ],
        }

        return improvement_suggestions.get(feature, [])

    def _estimate_task_effort(self, task: str) -> float:
        """
        Оценка сложности выполнения задачи
        """
        # Улучшенная эвристика для оценки сложности
        effort = 0.5  # Базовая сложность

        if "СРОЧНО" in task:
            effort += 0.4
        elif "КРИТИЧНО" in task:
            effort += 0.3
        if "добавьте" in task.lower():
            effort += 0.2
        if "расширьте" in task.lower():
            effort += 0.2
        if "создайте" in task.lower():
            effort += 0.3

        return min(max(effort, 0.1), 1.0)  # Ограничиваем значение в диапазоне [0.1, 1.0]
