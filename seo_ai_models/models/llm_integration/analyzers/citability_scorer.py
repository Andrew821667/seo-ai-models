"""
Оценка вероятности цитирования контента в LLM-ответах.

Модуль предоставляет функционал для оценки вероятности цитирования
контента в ответах LLM-моделей и рекомендаций по улучшению цитируемости.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from ..service.multi_model_agent import MultiModelAgent
from ..common.utils import extract_scores_from_text, parse_json_response, chunk_text


class CitabilityScorer:
    """
    Оценка вероятности цитирования контента в LLM-ответах.
    """

    def __init__(self, llm_service: LLMService, prompt_generator: PromptGenerator):
        """
        Инициализирует оценщик цитируемости.

        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM
            prompt_generator: Экземпляр PromptGenerator для генерации промптов
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        self.multi_model_agent = MultiModelAgent(llm_service, prompt_generator)

        # Настройка логгирования
        self.logger = logging.getLogger(__name__)

        # Тематические факторы цитируемости по категориям
        self.citability_factors = {
            "general": [
                "Информативность",
                "Уникальность",
                "Достоверность",
                "Структурированность",
                "Ясность изложения",
            ],
            "science": [
                "Наличие точных данных",
                "Ссылки на исследования",
                "Методология",
                "Воспроизводимость",
                "Связь с существующими теориями",
            ],
            "tech": [
                "Актуальность технологии",
                "Технические детали",
                "Примеры кода",
                "Сравнение с альтернативами",
                "Практическое применение",
            ],
            "health": [
                "Медицинская точность",
                "Ссылки на исследования",
                "Официальные рекомендации",
                "Мнения экспертов",
                "Безопасность информации",
            ],
            "business": [
                "Рыночные данные",
                "Статистика",
                "Кейсы",
                "Анализ трендов",
                "Практические рекомендации",
            ],
        }

    def score_citability(
        self,
        content: str,
        category: Optional[str] = "general",
        queries: Optional[List[str]] = None,
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Оценивает вероятность цитирования контента в ответах LLM.

        Args:
            content: Текст для анализа
            category: Категория контента (general, science, tech, health, business)
            queries: Список запросов, для которых оценивается цитируемость (опционально)
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            Dict[str, Any]: Результат оценки цитируемости
        """
        # Если категория не поддерживается, используем общую
        if category not in self.citability_factors:
            category = "general"

        # Если контент слишком большой, разбиваем его на чанки
        if len(content) > 15000:
            return self._score_large_content(content, category, queries, budget)

        # Генерируем промпт для оценки цитируемости
        prompt = self._generate_citability_prompt(content, category, queries)

        # Используем MultiModelAgent для выбора оптимальной модели и анализа
        result = self.multi_model_agent.analyze_content(
            content, "citability", budget, use_multiple_models=False
        )

        # Обрабатываем результат анализа
        return self._process_citability_result(result, category, queries)

    def _generate_citability_prompt(
        self, content: str, category: str, queries: Optional[List[str]] = None
    ) -> str:
        """
        Генерирует промпт для оценки цитируемости.

        Args:
            content: Текст для анализа
            category: Категория контента
            queries: Список запросов, для которых оценивается цитируемость

        Returns:
            str: Промпт для анализа
        """
        # Получаем факторы цитируемости для указанной категории
        factors = self.citability_factors.get(category, self.citability_factors["general"])

        # Формируем часть промпта с факторами
        factors_prompt = "\n".join([f"- {factor}" for factor in factors])

        # Формируем часть промпта с запросами
        queries_prompt = ""
        if queries:
            queries_prompt = "\n\nОцени цитируемость для следующих запросов:\n"
            queries_prompt += "\n".join([f"- {query}" for query in queries])

        # Базовый промпт для оценки цитируемости
        base_prompt = f"""
        Ты эксперт по оптимизации контента для цитирования в ответах LLM (языковых моделей).
        
        Проанализируй следующий текст и оцени вероятность его цитирования LLM-моделями при ответе на пользовательские запросы.
        
        Оцени следующие факторы цитируемости по шкале от 1 до 10:
        {factors_prompt}
        
        Также дай общую оценку цитируемости по шкале от 1 до 10.
        {queries_prompt}
        
        Для каждого фактора предложи конкретные улучшения, которые повысят цитируемость контента.
        
        Сформируй результат в JSON формате со следующей структурой:
        {{
            "scores": {{
                "overall": 0,
                "factors": {{
                    "фактор1": 0,
                    "фактор2": 0,
                    ...
                }}
            }},
            "analysis": {{
                "overall": "Общий анализ цитируемости",
                "factors": {{
                    "фактор1": "Анализ фактора1",
                    "фактор2": "Анализ фактора2",
                    ...
                }}
            }},
            "improvements": {{
                "фактор1": ["Улучшение 1", "Улучшение 2"],
                "фактор2": ["Улучшение 1", "Улучшение 2"],
                ...
            }},
            "queries_analysis": [
                {{
                    "query": "запрос1",
                    "citability": 0,
                    "analysis": "Анализ цитируемости для запроса1"
                }},
                ...
            ],
            "summary": "Краткое резюме анализа"
        }}
        """

        # Формируем финальный промпт
        final_prompt = base_prompt + "\n\nТекст для анализа:\n" + content

        return final_prompt

    def _process_citability_result(
        self, result: Dict[str, Any], category: str, queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Обрабатывает результат оценки цитируемости.

        Args:
            result: Результат анализа от LLM
            category: Категория контента
            queries: Список запросов, для которых оценивалась цитируемость

        Returns:
            Dict[str, Any]: Обработанный результат анализа
        """
        # Извлекаем текст ответа
        text = result.get("text", "")

        # Пытаемся извлечь JSON из ответа
        citability_data = parse_json_response(text)

        # Если не удалось извлечь JSON, пытаемся структурировать ответ сами
        if not citability_data or "scores" not in citability_data:
            citability_data = {
                "scores": {"overall": 0, "factors": {}},
                "analysis": {"overall": "", "factors": {}},
                "improvements": {},
                "queries_analysis": [],
                "summary": "",
            }

            # Извлекаем общую оценку цитируемости
            overall_match = re.search(r"общ[а-я]+\s+оценка.*?(\d+)[^а-я0-9]*", text, re.IGNORECASE)
            if overall_match:
                citability_data["scores"]["overall"] = int(overall_match.group(1))

            # Извлекаем оценки факторов
            factors = self.citability_factors.get(category, self.citability_factors["general"])
            for factor in factors:
                factor_pattern = f"{factor}.*?(\d+)[^а-я0-9]*"
                factor_match = re.search(factor_pattern, text, re.IGNORECASE)
                if factor_match:
                    citability_data["scores"]["factors"][factor] = int(factor_match.group(1))

            # Пытаемся извлечь анализ и улучшения из текста
            sections = re.split(r"\n\s*#{1,3}\s+", text)

            for section in sections:
                lines = section.strip().split("\n")
                if not lines:
                    continue

                section_title = lines[0].strip().lower()
                section_content = "\n".join(lines[1:]).strip()

                # Определяем тип секции
                if "общ" in section_title or "итог" in section_title or "вывод" in section_title:
                    citability_data["analysis"]["overall"] = section_content
                    citability_data["summary"] = section_content

                # Ищем анализ факторов
                for factor in factors:
                    if factor.lower() in section_title.lower():
                        citability_data["analysis"]["factors"][factor] = section_content
                        citability_data["improvements"][factor] = re.findall(
                            r"[-*]\s*(.*?)(?:\n|$)", section_content
                        )

            # Если были указаны запросы, пытаемся извлечь анализ для них
            if queries:
                for query in queries:
                    query_pattern = f"{re.escape(query)}.*?(\d+)[^а-я0-9]*"
                    query_match = re.search(query_pattern, text, re.IGNORECASE)
                    if query_match:
                        citability_score = int(query_match.group(1))

                        # Ищем анализ для запроса
                        query_analysis = ""
                        query_section = re.search(
                            f"{re.escape(query)}.*?\n(.*?)(?:\n\n|\n#|$)",
                            text,
                            re.IGNORECASE | re.DOTALL,
                        )
                        if query_section:
                            query_analysis = query_section.group(1).strip()

                        citability_data["queries_analysis"].append(
                            {
                                "query": query,
                                "citability": citability_score,
                                "analysis": query_analysis,
                            }
                        )

            # Добавляем исходный текст ответа
            citability_data["raw_text"] = text

        # Формируем итоговый результат
        analyzed_result = {
            "citability_score": citability_data.get("scores", {}).get("overall", 0),
            "factor_scores": citability_data.get("scores", {}).get("factors", {}),
            "citability_analysis": citability_data.get("analysis", {}).get("overall", ""),
            "factor_analysis": citability_data.get("analysis", {}).get("factors", {}),
            "suggested_improvements": citability_data.get("improvements", {}),
            "queries_analysis": citability_data.get("queries_analysis", []),
            "summary": citability_data.get("summary", ""),
            "category": category,
            "provider": result.get("provider", ""),
            "model": result.get("model", ""),
            "tokens": result.get("tokens", {}),
            "cost": result.get("cost", 0),
        }

        return analyzed_result

    # Продолжение файла citability_scorer.py
    def _score_large_content(
        self,
        content: str,
        category: str,
        queries: Optional[List[str]] = None,
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Оценивает цитируемость большого контента, разбивая его на чанки.

        Args:
            content: Текст для анализа
            category: Категория контента
            queries: Список запросов, для которых оценивается цитируемость
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            Dict[str, Any]: Результат оценки цитируемости
        """
        # Разбиваем контент на чанки
        chunks = chunk_text(content, max_chunk_size=10000, overlap=200)

        # Оцениваем каждый чанк
        chunk_results = []

        # Распределяем бюджет на чанки
        chunk_budget = None
        if budget is not None:
            chunk_budget = budget / len(chunks)

        for i, chunk in enumerate(chunks):
            self.logger.info(f"Оценка чанка {i+1} из {len(chunks)}")

            # Оцениваем чанк
            chunk_result = self.score_citability(chunk, category, queries, chunk_budget)
            chunk_results.append(chunk_result)

        # Объединяем результаты оценки чанков
        return self._combine_chunk_results(chunk_results, content)

    def _combine_chunk_results(
        self, chunk_results: List[Dict[str, Any]], original_content: str
    ) -> Dict[str, Any]:
        """
        Объединяет результаты оценки чанков.

        Args:
            chunk_results: Список результатов оценки чанков
            original_content: Исходный текст для анализа

        Returns:
            Dict[str, Any]: Объединенный результат оценки
        """
        # Если нет результатов, возвращаем пустой результат
        if not chunk_results:
            return {
                "citability_score": 0,
                "factor_scores": {},
                "citability_analysis": "",
                "factor_analysis": {},
                "suggested_improvements": {},
                "queries_analysis": [],
                "summary": "Не удалось оценить цитируемость.",
                "category": "",
                "chunks_analyzed": 0,
                "original_content_length": len(original_content),
                "tokens": {"total": 0},
                "cost": 0,
            }

        # Категория контента
        category = chunk_results[0].get("category", "general")

        # Объединяем оценки (среднее значение)
        overall_scores = [result.get("citability_score", 0) for result in chunk_results]
        overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0

        # Объединяем оценки факторов
        all_factors = set()
        factor_score_sums = {}
        factor_score_counts = {}

        for result in chunk_results:
            for factor, score in result.get("factor_scores", {}).items():
                all_factors.add(factor)
                factor_score_sums[factor] = factor_score_sums.get(factor, 0) + score
                factor_score_counts[factor] = factor_score_counts.get(factor, 0) + 1

        # Вычисляем средние оценки факторов
        factor_scores = {}
        for factor in all_factors:
            if factor_score_counts.get(factor, 0) > 0:
                factor_scores[factor] = factor_score_sums[factor] / factor_score_counts[factor]

        # Объединяем анализ
        citability_analyses = [
            result.get("citability_analysis", "")
            for result in chunk_results
            if result.get("citability_analysis")
        ]
        citability_analysis = "\n\n".join(citability_analyses)

        # Объединяем анализ факторов
        factor_analysis = {}
        for factor in all_factors:
            factor_analyses = []
            for result in chunk_results:
                factor_analysis_text = result.get("factor_analysis", {}).get(factor, "")
                if factor_analysis_text:
                    factor_analyses.append(factor_analysis_text)

            if factor_analyses:
                factor_analysis[factor] = "\n\n".join(factor_analyses)

        # Объединяем предложения по улучшению
        suggested_improvements = {}
        for factor in all_factors:
            all_improvements = []
            for result in chunk_results:
                improvements = result.get("suggested_improvements", {}).get(factor, [])
                all_improvements.extend(improvements)

            # Удаляем дубликаты
            suggested_improvements[factor] = list(set(all_improvements))

        # Объединяем анализ запросов
        queries_analysis = []
        query_data = {}

        for result in chunk_results:
            for query_analysis in result.get("queries_analysis", []):
                query = query_analysis.get("query", "")
                if not query:
                    continue

                if query not in query_data:
                    query_data[query] = {"scores": [], "analyses": []}

                query_data[query]["scores"].append(query_analysis.get("citability", 0))

                analysis_text = query_analysis.get("analysis", "")
                if analysis_text:
                    query_data[query]["analyses"].append(analysis_text)

        # Формируем объединенный анализ запросов
        for query, data in query_data.items():
            avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
            combined_analysis = "\n\n".join(data["analyses"])

            queries_analysis.append(
                {"query": query, "citability": avg_score, "analysis": combined_analysis}
            )

        # Формируем общий итог
        all_summaries = [
            result.get("summary", "") for result in chunk_results if result.get("summary")
        ]
        summary = "\n\n".join(all_summaries) if all_summaries else ""

        # Если нет суммари, генерируем его из оценок
        if not summary:
            if overall_score > 7:
                summary = (
                    "Контент обладает высокой цитируемостью, но есть возможности для улучшения."
                )
            elif overall_score > 5:
                summary = (
                    "Контент обладает средней цитируемостью, требуются существенные улучшения."
                )
            else:
                summary = (
                    "Контент обладает низкой цитируемостью, требуется комплексная переработка."
                )

        # Собираем статистику по токенам и стоимости
        total_tokens = sum(result.get("tokens", {}).get("total", 0) for result in chunk_results)
        total_cost = sum(result.get("cost", 0) for result in chunk_results)

        # Формируем итоговый результат
        combined_result = {
            "citability_score": overall_score,
            "factor_scores": factor_scores,
            "citability_analysis": citability_analysis,
            "factor_analysis": factor_analysis,
            "suggested_improvements": suggested_improvements,
            "queries_analysis": queries_analysis,
            "summary": summary,
            "category": category,
            "chunks_analyzed": len(chunk_results),
            "original_content_length": len(original_content),
            "tokens": {"total": total_tokens},
            "cost": total_cost,
        }

        return combined_result

    def compare_citability(
        self,
        contents: List[str],
        queries: List[str],
        category: str = "general",
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Сравнивает цитируемость нескольких вариантов контента.

        Args:
            contents: Список вариантов текста для сравнения
            queries: Список запросов, для которых оценивается цитируемость
            category: Категория контента
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            Dict[str, Any]: Результат сравнения цитируемости
        """
        # Оцениваем цитируемость каждого варианта
        variant_results = []

        # Распределяем бюджет на варианты
        variant_budget = None
        if budget is not None:
            variant_budget = budget / len(contents)

        for i, content in enumerate(contents):
            self.logger.info(f"Оценка варианта {i+1} из {len(contents)}")

            # Оцениваем вариант
            result = self.score_citability(content, category, queries, variant_budget)
            variant_results.append(result)

        # Определяем лучший вариант по общей оценке цитируемости
        best_variant_index = max(
            range(len(variant_results)), key=lambda i: variant_results[i].get("citability_score", 0)
        )

        best_variant = variant_results[best_variant_index]

        # Определяем лучший вариант для каждого запроса
        best_for_queries = {}

        for query in queries:
            query_scores = []

            for i, result in enumerate(variant_results):
                query_analysis = next(
                    (qa for qa in result.get("queries_analysis", []) if qa.get("query") == query),
                    {"citability": 0},
                )

                query_scores.append((i, query_analysis.get("citability", 0)))

            # Сортируем по оценке
            query_scores.sort(key=lambda x: x[1], reverse=True)

            best_for_queries[query] = {
                "variant_index": query_scores[0][0] if query_scores else 0,
                "score": query_scores[0][1] if query_scores else 0,
            }

        # Собираем статистику по токенам и стоимости
        total_tokens = sum(result.get("tokens", {}).get("total", 0) for result in variant_results)
        total_cost = sum(result.get("cost", 0) for result in variant_results)

        # Формируем результат сравнения
        comparison_result = {
            "variants_count": len(contents),
            "best_variant": {
                "index": best_variant_index,
                "score": best_variant.get("citability_score", 0),
                "factor_scores": best_variant.get("factor_scores", {}),
                "summary": best_variant.get("summary", ""),
            },
            "best_for_queries": best_for_queries,
            "variant_scores": [result.get("citability_score", 0) for result in variant_results],
            "category": category,
            "queries": queries,
            "tokens": {"total": total_tokens},
            "cost": total_cost,
        }

        return comparison_result
