"""
Предиктор ранжирования для LLM-поисковиков.

Модуль предоставляет функционал для предсказания ранжирования контента
в LLM-поисковиках на основе различных факторов, включая структуру контента,
E-E-A-T, цитируемость и другие метрики.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from ..analyzers.citability_scorer import CitabilityScorer
from ..analyzers.llm_eeat_analyzer import LLMEEATAnalyzer
from ..serp_analysis.llm_serp_analyzer import LLMSerpAnalyzer
from ..common.utils import parse_json_response


class LLMRankPredictor:
    """
    Предиктор ранжирования для LLM-поисковиков.
    """

    def __init__(self, llm_service: LLMService, prompt_generator: PromptGenerator):
        """
        Инициализирует предиктор ранжирования для LLM-поисковиков.

        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM
            prompt_generator: Экземпляр PromptGenerator для генерации промптов
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        self.citability_scorer = CitabilityScorer(llm_service, prompt_generator)
        self.eeat_analyzer = LLMEEATAnalyzer(llm_service, prompt_generator)
        self.serp_analyzer = LLMSerpAnalyzer(llm_service, prompt_generator)

        # Настройка логгирования
        self.logger = logging.getLogger(__name__)

        # Веса факторов ранжирования по умолчанию
        self.default_ranking_weights = {
            "citability": 0.30,  # Цитируемость
            "eeat": 0.25,  # E-E-A-T
            "content_structure": 0.15,  # Структура контента
            "keyword_relevance": 0.10,  # Релевантность ключевым словам
            "uniqueness": 0.10,  # Уникальность
            "freshness": 0.05,  # Свежесть
            "authorInfo": 0.05,  # Информация об авторе
        }

        # Отраслевые корректировки весов
        self.industry_weight_adjustments = {
            "health": {
                "eeat": 0.40,  # Повышенный вес E-E-A-T
                "citability": 0.20,
                "authorInfo": 0.10,
            },
            "finance": {"eeat": 0.35, "freshness": 0.10, "citability": 0.25},
            "technology": {"freshness": 0.15, "content_structure": 0.20},  # Повышенный вес свежести
            "news": {"freshness": 0.25, "citability": 0.35},  # Приоритет свежести для новостей
        }

    def predict_ranking(
        self,
        content: str,
        query: str,
        competitors: Optional[List[Dict[str, Any]]] = None,
        industry: Optional[str] = None,
        llm_engines: Optional[List[str]] = None,
        comprehensive_analysis: bool = False,
    ) -> Dict[str, Any]:
        """
        Предсказывает ранжирование контента в LLM-поисковиках.

        Args:
            content: Контент для анализа
            query: Поисковый запрос
            competitors: Список конкурентов с их контентом (опционально)
            industry: Отрасль (опционально)
            llm_engines: Список LLM-поисковиков для анализа (опционально)
            comprehensive_analysis: Выполнять ли полный анализ контента (опционально)

        Returns:
            Dict[str, Any]: Результаты предсказания ранжирования
        """
        # Логируем начало анализа
        self.logger.info(f"Начало предсказания ранжирования по запросу: {query}")

        # Если engines не указаны, используем по умолчанию
        if not llm_engines:
            llm_engines = ["perplexity", "claude_search", "bing_copilot"]

        # Выполняем оценку цитируемости
        citability_result = self.citability_scorer.score_citability(
            content=content, queries=[query]
        )

        # Выполняем оценку E-E-A-T
        eeat_result = self.eeat_analyzer.analyze_llm_eeat(
            content=content, industry=industry or "general"
        )

        # Анализируем SERP для понимания текущей видимости
        serp_result = self.serp_analyzer.analyze_serp(
            query=query, content=content, llm_engines=llm_engines, num_samples=1
        )

        # Получаем веса факторов с учетом отрасли
        ranking_weights = self._get_ranking_weights(industry)

        # Предсказываем ранжирование
        if competitors:
            ranking_prediction = self._predict_with_competitors(
                content,
                query,
                competitors,
                ranking_weights,
                citability_result,
                eeat_result,
                serp_result,
            )
        else:
            ranking_prediction = self._predict_without_competitors(
                content, query, ranking_weights, citability_result, eeat_result, serp_result
            )

        # Если требуется полный анализ, выполняем его
        if comprehensive_analysis:
            ranking_prediction["detailed_analysis"] = self._perform_comprehensive_analysis(
                content, query, citability_result, eeat_result, serp_result
            )

        # Формируем итоговый результат
        result = {
            "query": query,
            "ranking_prediction": ranking_prediction,
            "citability_score": citability_result.get("citability_score", 0),
            "eeat_score": eeat_result.get("eeat_score", 0),
            "visibility_score": serp_result.get("visibility_score", 0),
            "citation_rate": serp_result.get("citation_rate", 0),
            "industry": industry or "general",
            "llm_engines": llm_engines,
            "ranking_weights": ranking_weights,
            "timestamp": datetime.now().isoformat(),
            "tokens": {
                "citability": citability_result.get("tokens", {}).get("total", 0),
                "eeat": eeat_result.get("tokens", {}).get("total", 0),
                "serp": serp_result.get("tokens", {}).get("total", 0),
                "total": (
                    citability_result.get("tokens", {}).get("total", 0)
                    + eeat_result.get("tokens", {}).get("total", 0)
                    + serp_result.get("tokens", {}).get("total", 0)
                ),
            },
            "cost": (
                citability_result.get("cost", 0)
                + eeat_result.get("cost", 0)
                + serp_result.get("cost", 0)
            ),
        }

        return result

    def _get_ranking_weights(self, industry: Optional[str] = None) -> Dict[str, float]:
        """
        Возвращает веса факторов ранжирования с учетом отрасли.

        Args:
            industry: Отрасль (опционально)

        Returns:
            Dict[str, float]: Веса факторов ранжирования
        """
        # Если отрасль не указана или не поддерживается, используем веса по умолчанию
        if not industry or industry not in self.industry_weight_adjustments:
            return self.default_ranking_weights.copy()

        # Применяем отраслевые корректировки
        weights = self.default_ranking_weights.copy()
        industry_adjustments = self.industry_weight_adjustments[industry]

        for factor, weight in industry_adjustments.items():
            weights[factor] = weight

        # Нормализуем веса, чтобы их сумма была равна 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _predict_with_competitors(
        self,
        content: str,
        query: str,
        competitors: List[Dict[str, Any]],
        ranking_weights: Dict[str, float],
        citability_result: Dict[str, Any],
        eeat_result: Dict[str, Any],
        serp_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Предсказывает ранжирование с учетом конкурентов.

        Args:
            content: Контент для анализа
            query: Поисковый запрос
            competitors: Список конкурентов с их контентом
            ranking_weights: Веса факторов ранжирования
            citability_result: Результат оценки цитируемости
            eeat_result: Результат оценки E-E-A-T
            serp_result: Результат анализа SERP

        Returns:
            Dict[str, Any]: Результаты предсказания ранжирования
        """
        # Оцениваем конкурентов
        competitor_scores = []

        for i, competitor in enumerate(competitors):
            comp_id = competitor.get("id", f"competitor_{i}")
            comp_name = competitor.get("name", f"Конкурент {i+1}")
            comp_content = competitor.get("content", "")

            # Пропускаем, если контент отсутствует
            if not comp_content:
                self.logger.warning(f"Пропуск конкурента {comp_name}: контент отсутствует")
                continue

            self.logger.info(f"Анализ конкурента {i+1} из {len(competitors)}: {comp_name}")

            # Оцениваем цитируемость конкурента
            comp_citability = self.citability_scorer.score_citability(
                content=comp_content, queries=[query]
            )

            # Оцениваем E-E-A-T конкурента
            comp_eeat = self.eeat_analyzer.analyze_llm_eeat(content=comp_content)

            # Анализируем SERP для конкурента
            comp_serp = self.serp_analyzer.analyze_serp(
                query=query,
                content=comp_content,
                llm_engines=serp_result.get("engines", []),
                num_samples=1,
            )

            # Рассчитываем итоговую оценку конкурента
            comp_score = self._calculate_total_score(
                citability_result=comp_citability,
                eeat_result=comp_eeat,
                serp_result=comp_serp,
                ranking_weights=ranking_weights,
            )

            # Добавляем оценку конкурента
            competitor_scores.append(
                {
                    "id": comp_id,
                    "name": comp_name,
                    "total_score": comp_score,
                    "citability_score": comp_citability.get("citability_score", 0),
                    "eeat_score": comp_eeat.get("eeat_score", 0),
                    "visibility_score": comp_serp.get("visibility_score", 0),
                    "citation_rate": comp_serp.get("citation_rate", 0),
                }
            )

        # Рассчитываем итоговую оценку нашего контента
        our_score = self._calculate_total_score(
            citability_result=citability_result,
            eeat_result=eeat_result,
            serp_result=serp_result,
            ranking_weights=ranking_weights,
        )

        # Добавляем наш контент к списку для ранжирования
        all_scores = competitor_scores + [
            {
                "id": "our_content",
                "name": "Наш контент",
                "total_score": our_score,
                "citability_score": citability_result.get("citability_score", 0),
                "eeat_score": eeat_result.get("eeat_score", 0),
                "visibility_score": serp_result.get("visibility_score", 0),
                "citation_rate": serp_result.get("citation_rate", 0),
            }
        ]

        # Сортируем по убыванию итоговой оценки
        all_scores.sort(key=lambda x: x["total_score"], reverse=True)

        # Назначаем ранги
        for i, score in enumerate(all_scores):
            score["rank"] = i + 1

        # Находим ранг нашего контента
        our_rank = next((score["rank"] for score in all_scores if score["id"] == "our_content"), -1)

        # Определяем процентиль нашего контента
        percentile = (len(all_scores) - our_rank + 1) / len(all_scores) * 100 if our_rank > 0 else 0

        # Формируем результат
        result = {
            "our_score": our_score,
            "our_rank": our_rank,
            "total_competitors": len(competitor_scores),
            "percentile": percentile,
            "ranking": all_scores,
            "score_breakdown": {
                "citability": ranking_weights["citability"]
                * citability_result.get("citability_score", 0)
                / 10,
                "eeat": ranking_weights["eeat"] * eeat_result.get("eeat_score", 0) / 10,
                "visibility": (
                    ranking_weights["content_structure"] + ranking_weights["keyword_relevance"]
                )
                * serp_result.get("visibility_score", 0),
            },
        }

        return result

    def _predict_without_competitors(
        self,
        content: str,
        query: str,
        ranking_weights: Dict[str, float],
        citability_result: Dict[str, Any],
        eeat_result: Dict[str, Any],
        serp_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Предсказывает ранжирование без учета конкурентов.

        Args:
            content: Контент для анализа
            query: Поисковый запрос
            ranking_weights: Веса факторов ранжирования
            citability_result: Результат оценки цитируемости
            eeat_result: Результат оценки E-E-A-T
            serp_result: Результат анализа SERP

        Returns:
            Dict[str, Any]: Результаты предсказания ранжирования
        """
        # Рассчитываем итоговую оценку
        total_score = self._calculate_total_score(
            citability_result=citability_result,
            eeat_result=eeat_result,
            serp_result=serp_result,
            ranking_weights=ranking_weights,
        )

        # Оцениваем потенциальную позицию
        position_estimate = self._estimate_position(total_score, query)

        # Формируем результат
        result = {
            "total_score": total_score,
            "position_estimate": position_estimate,
            "score_breakdown": {
                "citability": ranking_weights["citability"]
                * citability_result.get("citability_score", 0)
                / 10,
                "eeat": ranking_weights["eeat"] * eeat_result.get("eeat_score", 0) / 10,
                "visibility": (
                    ranking_weights["content_structure"] + ranking_weights["keyword_relevance"]
                )
                * serp_result.get("visibility_score", 0),
            },
        }

        return result

    def _calculate_total_score(
        self,
        citability_result: Dict[str, Any],
        eeat_result: Dict[str, Any],
        serp_result: Dict[str, Any],
        ranking_weights: Dict[str, float],
    ) -> float:
        """
        Рассчитывает итоговую оценку на основе различных факторов.

        Args:
            citability_result: Результат оценки цитируемости
            eeat_result: Результат оценки E-E-A-T
            serp_result: Результат анализа SERP
            ranking_weights: Веса факторов ранжирования

        Returns:
            float: Итоговая оценка
        """
        # Нормализуем оценки к единой шкале (0-1)
        citability_score = citability_result.get("citability_score", 0) / 10
        eeat_score = eeat_result.get("eeat_score", 0) / 10
        visibility_score = serp_result.get("visibility_score", 0)
        citation_rate = serp_result.get("citation_rate", 0)

        # Учитываем разные факторы с соответствующими весами
        score = (
            ranking_weights["citability"] * citability_score
            + ranking_weights["eeat"] * eeat_score
            + ranking_weights["content_structure"] * visibility_score
            + ranking_weights["keyword_relevance"] * citation_rate
        )

        # Freshness и другие факторы могут быть учтены при наличии соответствующих данных
        # Для простоты примера не учитываем их

        # Нормализуем итоговую оценку к шкале 0-10
        normalized_score = min(10, max(0, score * 10))

        return normalized_score

    def _estimate_position(self, score: float, query: str) -> Dict[str, Any]:
        """
        Оценивает потенциальную позицию контента в LLM-поисковиках.

        Args:
            score: Итоговая оценка
            query: Поисковый запрос

        Returns:
            Dict[str, Any]: Оценка позиции
        """
        # В реальном проекте здесь была бы более сложная модель
        # Для примера используем простую эвристику

        # Преобразуем оценку в диапазон 1-10 для позиции (обратная зависимость)
        position = max(1, 11 - round(score))

        # Определяем вероятность попадания в top-N
        top1_probability = max(0, min(1, (score - 8) / 2)) if score >= 8 else 0
        top3_probability = max(0, min(1, (score - 6) / 4)) if score >= 6 else 0
        top5_probability = max(0, min(1, (score - 5) / 5)) if score >= 5 else 0
        top10_probability = max(0, min(1, score / 10))

        # Формируем оценку позиции
        position_estimate = {
            "estimated_position": position,
            "probability_top1": top1_probability,
            "probability_top3": top3_probability,
            "probability_top5": top5_probability,
            "probability_top10": top10_probability,
            "confidence": min(1, max(0, score / 10)),  # Уверенность в оценке
        }

        return position_estimate

    def _perform_comprehensive_analysis(
        self,
        content: str,
        query: str,
        citability_result: Dict[str, Any],
        eeat_result: Dict[str, Any],
        serp_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Выполняет комплексный анализ контента.

        Args:
            content: Контент для анализа
            query: Поисковый запрос
            citability_result: Результат оценки цитируемости
            eeat_result: Результат оценки E-E-A-T
            serp_result: Результат анализа SERP

        Returns:
            Dict[str, Any]: Результаты комплексного анализа
        """
        # Формируем промпт для комплексного анализа
        prompt = f"""
        Ты эксперт по ранжированию контента в LLM-поисковиках (языковых моделях).
        
        Проанализируй следующий контент, учитывая запрос пользователя и предоставленные метрики.
        
        Запрос пользователя: "{query}"
        
        Метрики контента:
        - Цитируемость: {citability_result.get('citability_score', 0)}/10
        - E-E-A-T: {eeat_result.get('eeat_score', 0)}/10
        - Видимость в LLM: {serp_result.get('visibility_score', 0):.2f}
        - Частота цитирования: {serp_result.get('citation_rate', 0):.2%}
        
        Проведи комплексный анализ контента для понимания его сильных и слабых сторон с точки зрения ранжирования в LLM-поисковиках.
        
        Ответ предоставь в формате JSON со следующей структурой:
        {{
            "strengths": ["сильная сторона 1", "сильная сторона 2", ...],
            "weaknesses": ["слабая сторона 1", "слабая сторона 2", ...],
            "opportunities": ["возможность 1", "возможность 2", ...],
            "threats": ["угроза 1", "угроза 2", ...],
            "overall_analysis": "Общий анализ контента"
        }}
        
        Контент для анализа:
        {content}
        """

        # Запрос к LLM
        response = self.llm_service.query(prompt=prompt, max_tokens=1000, temperature=0.5)

        # Извлекаем анализ из ответа
        analysis_text = response.get("text", "")
        analysis_data = parse_json_response(analysis_text)

        # Если не удалось извлечь JSON, возвращаем базовый анализ
        if not analysis_data:
            analysis_data = {
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": [],
                "overall_analysis": "Не удалось выполнить комплексный анализ.",
            }

        # Добавляем метаданные анализа
        analysis_data["tokens"] = response.get("tokens", {})
        analysis_data["cost"] = response.get("cost", 0)

        return analysis_data

    def predict_impact_of_changes(
        self,
        current_content: str,
        improved_content: str,
        query: str,
        industry: Optional[str] = None,
        llm_engines: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Предсказывает влияние изменений контента на его ранжирование.

        Args:
            current_content: Текущий контент
            improved_content: Улучшенный контент
            query: Поисковый запрос
            industry: Отрасль (опционально)
            llm_engines: Список LLM-поисковиков для анализа (опционально)

        Returns:
            Dict[str, Any]: Результаты предсказания влияния изменений
        """
        # Предсказываем ранжирование для текущего и улучшенного контента
        current_ranking = self.predict_ranking(
            content=current_content, query=query, industry=industry, llm_engines=llm_engines
        )

        improved_ranking = self.predict_ranking(
            content=improved_content, query=query, industry=industry, llm_engines=llm_engines
        )

        # Рассчитываем изменения в оценках
        score_change = (
            improved_ranking["ranking_prediction"]["total_score"]
            - current_ranking["ranking_prediction"]["total_score"]
        )

        # Сравниваем факторы
        factor_changes = {}
        for factor, current_value in current_ranking["ranking_prediction"][
            "score_breakdown"
        ].items():
            improved_value = improved_ranking["ranking_prediction"]["score_breakdown"].get(
                factor, 0
            )
            factor_changes[factor] = {
                "current": current_value,
                "improved": improved_value,
                "change": improved_value - current_value,
                "percent_change": ((improved_value - current_value) / max(0.001, current_value))
                * 100,
            }

        # Сравниваем оценки позиций
        position_change = {}
        if (
            "position_estimate" in current_ranking["ranking_prediction"]
            and "position_estimate" in improved_ranking["ranking_prediction"]
        ):
            current_position = current_ranking["ranking_prediction"]["position_estimate"][
                "estimated_position"
            ]
            improved_position = improved_ranking["ranking_prediction"]["position_estimate"][
                "estimated_position"
            ]
            position_change = {
                "current_position": current_position,
                "improved_position": improved_position,
                "position_change": current_position - improved_position,
                "top10_probability_change": (
                    improved_ranking["ranking_prediction"]["position_estimate"]["probability_top10"]
                    - current_ranking["ranking_prediction"]["position_estimate"][
                        "probability_top10"
                    ]
                ),
            }

        # Формируем результат
        result = {
            "query": query,
            "score_change": score_change,
            "score_percent_change": (
                score_change / max(0.001, current_ranking["ranking_prediction"]["total_score"])
            )
            * 100,
            "factor_changes": factor_changes,
            "position_change": position_change,
            "current_ranking": current_ranking,
            "improved_ranking": improved_ranking,
            "tokens": {
                "current": current_ranking.get("tokens", {}).get("total", 0),
                "improved": improved_ranking.get("tokens", {}).get("total", 0),
                "total": (
                    current_ranking.get("tokens", {}).get("total", 0)
                    + improved_ranking.get("tokens", {}).get("total", 0)
                ),
            },
            "cost": (current_ranking.get("cost", 0) + improved_ranking.get("cost", 0)),
        }

        return result
