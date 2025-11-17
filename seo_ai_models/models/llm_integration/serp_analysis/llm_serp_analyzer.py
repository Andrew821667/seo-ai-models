"""
Анализатор результатов LLM-поисковиков.

Модуль предоставляет функционал для анализа и извлечения информации
из результатов LLM-поисковиков, сравнения с традиционными поисковыми системами
и оценки видимости и позиций в выдаче LLM.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from ..analyzers.citability_scorer import CitabilityScorer
from ..common.utils import parse_json_response


class LLMSerpAnalyzer:
    """
    Анализатор результатов LLM-поисковиков.
    """

    def __init__(self, llm_service: LLMService, prompt_generator: PromptGenerator):
        """
        Инициализирует анализатор результатов LLM-поисковиков.

        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM
            prompt_generator: Экземпляр PromptGenerator для генерации промптов
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        self.citability_scorer = CitabilityScorer(llm_service, prompt_generator)

        # Настройка логгирования
        self.logger = logging.getLogger(__name__)

        # Поддерживаемые LLM-поисковики
        self.supported_llm_engines = [
            "perplexity",
            "claude_search",
            "google_search",
            "bing_copilot",
            "you",
            "anthropic_claude",
            "kimi",
            "gigachat",
        ]

    def analyze_serp(
        self,
        query: str,
        content: str,
        llm_engines: Optional[List[str]] = None,
        num_samples: int = 3,
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Анализирует результаты LLM-поисковиков для заданного запроса и контента.

        Args:
            query: Поисковый запрос
            content: Контент, для которого проводится анализ
            llm_engines: Список LLM-поисковиков для анализа
            num_samples: Количество запросов для каждого поисковика
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            Dict[str, Any]: Результаты анализа SERP
        """
        # Если поисковики не указаны, используем поддерживаемые по умолчанию
        if not llm_engines:
            # Используем первые 3 поисковика для экономии бюджета
            llm_engines = self.supported_llm_engines[:3]
        else:
            # Проверяем, что все указанные поисковики поддерживаются
            for engine in llm_engines:
                if engine not in self.supported_llm_engines:
                    self.logger.warning(f"Поисковик {engine} не поддерживается, будет пропущен")

            # Оставляем только поддерживаемые поисковики
            llm_engines = [engine for engine in llm_engines if engine in self.supported_llm_engines]

        # Если нет поддерживаемых поисковиков, возвращаем пустой результат
        if not llm_engines:
            self.logger.error("Нет поддерживаемых поисковиков для анализа")
            return {
                "error": "Нет поддерживаемых поисковиков для анализа",
                "query": query,
                "engines": [],
                "samples": 0,
                "citation_rate": 0,
                "visibility_score": 0,
                "engines_results": {},
                "tokens": {"total": 0},
                "cost": 0,
            }

        # Распределяем бюджет между поисковиками
        engine_budget = None
        if budget is not None:
            engine_budget = budget / len(llm_engines)

        # Выполняем анализ для каждого поисковика
        engines_results = {}
        total_tokens = 0
        total_cost = 0
        citation_count = 0
        visibility_scores = []

        for engine in llm_engines:
            self.logger.info(f"Анализ SERP для поисковика {engine}")

            # Анализируем результаты поисковика
            engine_result = self._analyze_engine_serp(
                query, content, engine, num_samples, engine_budget
            )

            engines_results[engine] = engine_result

            # Обновляем статистику
            total_tokens += engine_result.get("tokens", {}).get("total", 0)
            total_cost += engine_result.get("cost", 0)
            citation_count += engine_result.get("citation_count", 0)
            visibility_scores.append(engine_result.get("visibility_score", 0))

        # Вычисляем общие метрики
        total_samples = len(llm_engines) * num_samples
        citation_rate = citation_count / total_samples if total_samples > 0 else 0
        avg_visibility_score = (
            sum(visibility_scores) / len(visibility_scores) if visibility_scores else 0
        )

        # Формируем итоговый результат
        result = {
            "query": query,
            "engines": llm_engines,
            "samples": num_samples,
            "total_samples": total_samples,
            "citation_count": citation_count,
            "citation_rate": citation_rate,
            "visibility_score": avg_visibility_score,
            "engines_results": engines_results,
            "tokens": {"total": total_tokens},
            "cost": total_cost,
        }

        return result

    def _analyze_engine_serp(
        self,
        query: str,
        content: str,
        engine: str,
        num_samples: int,
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Анализирует результаты конкретного LLM-поисковика.

        Args:
            query: Поисковый запрос
            content: Контент, для которого проводится анализ
            engine: Название LLM-поисковика
            num_samples: Количество запросов к поисковику
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            Dict[str, Any]: Результаты анализа SERP для поисковика
        """
        # Распределяем бюджет между запросами
        sample_budget = None
        if budget is not None:
            sample_budget = budget / num_samples

        # Генерируем промпт для симуляции поисковика
        engine_prompt = self._generate_engine_prompt(query, engine)

        # Собираем результаты для всех образцов
        samples_results = []
        citation_count = 0
        total_tokens = 0
        total_cost = 0

        for i in range(num_samples):
            self.logger.info(f"Анализ образца {i+1} из {num_samples} для поисковика {engine}")

            # Получаем результат от LLM
            response = self.llm_service.query(
                engine_prompt, max_tokens=1000, temperature=0.7, budget=sample_budget
            )

            # Обрабатываем результат
            sample_result = self._process_engine_response(response, content, engine)
            samples_results.append(sample_result)

            # Обновляем статистику
            citation_count += 1 if sample_result.get("contains_citation", False) else 0
            total_tokens += response.get("tokens", {}).get("total", 0)
            total_cost += response.get("cost", 0)

        # Рассчитываем видимость контента в результатах поисковика
        visibility_score = self._calculate_visibility_score(samples_results)

        # Формируем итоговый результат для поисковика
        engine_result = {
            "engine": engine,
            "samples": num_samples,
            "citation_count": citation_count,
            "citation_rate": citation_count / num_samples if num_samples > 0 else 0,
            "visibility_score": visibility_score,
            "samples_results": samples_results,
            "tokens": {"total": total_tokens},
            "cost": total_cost,
        }

        return engine_result

    def _generate_engine_prompt(self, query: str, engine: str) -> str:
        """
        Генерирует промпт для симуляции LLM-поисковика.

        Args:
            query: Поисковый запрос
            engine: Название LLM-поисковика

        Returns:
            str: Промпт для симуляции поисковика
        """
        # Базовый промпт для симуляции поисковика
        base_prompt = f"""
        Ты опытный аналитик поисковых систем, который помогает симулировать ответы LLM-поисковиков.
        
        Представь, что ты поисковик {engine}, отвечающий на вопрос пользователя.
        
        Вопрос пользователя: "{query}"
        
        Создай реалистичный ответ, который мог бы дать этот поисковик. 
        Учти следующие особенности данного поисковика:
        """

        # Добавляем особенности конкретного поисковика
        engine_specifics = {
            "perplexity": """
            - Perplexity обычно дает краткие, прямые ответы
            - Всегда включает источники в виде ссылок
            - Часто представляет разные точки зрения
            - Отвечает в нейтральном тоне
            """,
            "claude_search": """
            - Claude Search дает подробные, обстоятельные ответы
            - Часто приводит цитаты из источников
            - Хорошо структурирует информацию
            - Отвечает в дружелюбном, разговорном тоне
            """,
            "google_search": """
            - Google AI (SGE) дает лаконичные ответы
            - Ограничивает длину ответа
            - Стремится дать самую релевантную информацию
            - Часто добавляет варианты дополнительных запросов
            """,
            "bing_copilot": """
            - Bing Copilot дает развернутые ответы
            - Часто включает списки и структурированную информацию
            - Приводит ссылки на источники
            - Иногда добавляет изображения (которые ты должен описать текстом)
            """,
            "you": """
            - You.com дает сбалансированные ответы средней длины
            - Хорошо структурирует информацию в виде параграфов
            - Включает разнообразные источники
            - Предлагает дополнительные вопросы по теме
            """,
            "anthropic_claude": """
            - Claude дает подробные, хорошо структурированные ответы
            - Часто делает оговорки и уточнения
            - Использует вежливый, подробный стиль общения
            - Избегает чрезмерного упрощения сложных вопросов
            """,
            "kimi": """
            - Kimi дает ответы в неформальном стиле
            - Стремится к краткости и ясности
            - Предпочитает конкретные примеры
            - Использует простой язык
            """,
            "gigachat": """
            - GigaChat дает подробные ответы на русском языке
            - Хорошо работает с российскими источниками
            - Использует официальный, но доступный стиль
            - Часто структурирует ответ в виде разделов
            """,
        }

        engine_prompt = base_prompt + engine_specifics.get(engine, "")

        # Добавляем инструкции по формату ответа
        engine_prompt += """
        
        Ответь в следующем формате JSON:
        {
            "answer": "Полный ответ поисковика",
            "sources": ["источник1", "источник2", ...],
            "sections": [
                {"title": "Название раздела 1", "content": "Содержание раздела 1"},
                {"title": "Название раздела 2", "content": "Содержание раздела 2"},
                ...
            ]
        }
        """

        return engine_prompt

    def _process_engine_response(
        self, response: Dict[str, Any], content: str, engine: str
    ) -> Dict[str, Any]:
        """
        Обрабатывает ответ LLM, симулирующий результаты поисковика.

        Args:
            response: Ответ от LLM
            content: Контент, для которого проводится анализ
            engine: Название LLM-поисковика

        Returns:
            Dict[str, Any]: Обработанный результат анализа
        """
        # Извлекаем текст ответа
        response_text = response.get("text", "")

        # Пытаемся извлечь JSON из ответа
        engine_data = parse_json_response(response_text)

        # Если не удалось извлечь JSON, пытаемся структурировать ответ сами
        if not engine_data or "answer" not in engine_data:
            engine_data = {"answer": response_text, "sources": [], "sections": []}

            # Пытаемся извлечь источники из текста
            sources_match = re.search(
                r"источники:?\s*\n(.*?)(?:\n\n|\n#|$)", response_text, re.IGNORECASE | re.DOTALL
            )
            if sources_match:
                sources_text = sources_match.group(1)
                engine_data["sources"] = re.findall(r"[-*]\s*(.*?)(?:\n|$)", sources_text)

            # Пытаемся извлечь разделы из текста
            sections = re.split(r"\n\s*#{1,3}\s+", response_text)

            for section in sections[1:]:  # Пропускаем первый раздел (введение)
                lines = section.strip().split("\n")
                if not lines:
                    continue

                section_title = lines[0].strip()
                section_content = "\n".join(lines[1:]).strip()

                engine_data["sections"].append({"title": section_title, "content": section_content})

        # Проверяем наличие цитирования контента в ответе
        contains_citation = self._check_content_citation(engine_data, content)

        # Оцениваем релевантность ответа
        relevance_score = self._evaluate_relevance(engine_data, content)

        # Определяем позицию цитирования контента
        citation_position = self._determine_citation_position(engine_data, content)

        # Формируем результат анализа
        result = {
            "engine": engine,
            "answer": engine_data.get("answer", ""),
            "sources": engine_data.get("sources", []),
            "sections_count": len(engine_data.get("sections", [])),
            "contains_citation": contains_citation,
            "citation_position": citation_position,
            "relevance_score": relevance_score,
            "provider": response.get("provider", ""),
            "model": response.get("model", ""),
            "raw_response": response_text,
        }

        return result

    def _check_content_citation(self, engine_data: Dict[str, Any], content: str) -> bool:
        """
        Проверяет наличие цитирования контента в ответе.

        Args:
            engine_data: Данные ответа поисковика
            content: Контент, для которого проводится анализ

        Returns:
            bool: True если контент цитируется, иначе False
        """
        # Извлекаем ключевые фразы из контента (например, первые 3 предложения)
        content_sentences = re.split(r"[.!?]", content)
        content_sentences = [s.strip() for s in content_sentences if s.strip()]
        key_phrases = content_sentences[:3]

        # Извлекаем также последние 3 предложения
        last_phrases = content_sentences[-3:]

        # Объединяем ключевые фразы
        all_key_phrases = key_phrases + last_phrases

        # Получаем ответ поисковика
        answer = engine_data.get("answer", "")

        # Проверяем наличие ключевых фраз в ответе
        for phrase in all_key_phrases:
            # Игнорируем слишком короткие фразы
            if len(phrase) < 10:
                continue

            # Нормализуем фразу для поиска
            normalized_phrase = re.sub(r"\s+", " ", phrase.lower())
            normalized_answer = re.sub(r"\s+", " ", answer.lower())

            # Если фраза найдена в ответе, считаем что контент цитируется
            if normalized_phrase in normalized_answer:
                return True

        # Проверяем наличие ссылок на источники, которые могут соответствовать контенту
        for source in engine_data.get("sources", []):
            # Проверяем, содержит ли источник ключевые слова из контента
            source_lower = source.lower()

            # Извлекаем ключевые слова из первого параграфа контента
            first_paragraph = " ".join(content_sentences[:5])
            key_words = re.findall(r"\b\w{5,}\b", first_paragraph.lower())

            # Считаем количество совпадений
            matches = sum(1 for word in key_words if word in source_lower)

            # Если более 3 ключевых слов найдены в источнике, считаем что контент цитируется
            if matches >= 3:
                return True

        # Если цитирование не найдено
        return False

    def _evaluate_relevance(self, engine_data: Dict[str, Any], content: str) -> float:
        """
        Оценивает релевантность ответа поисковика контенту.

        Args:
            engine_data: Данные ответа поисковика
            content: Контент, для которого проводится анализ

        Returns:
            float: Оценка релевантности от 0 до 1
        """
        # Получаем ответ поисковика
        answer = engine_data.get("answer", "")

        # Простой способ оценки релевантности - сравнение ключевых слов
        # Извлекаем ключевые слова из контента
        content_words = set(re.findall(r"\b\w{5,}\b", content.lower()))

        # Извлекаем ключевые слова из ответа
        answer_words = set(re.findall(r"\b\w{5,}\b", answer.lower()))

        # Определяем пересечение ключевых слов
        common_words = content_words.intersection(answer_words)

        # Рассчитываем релевантность как отношение общих слов к общему количеству уникальных слов
        if not content_words or not answer_words:
            return 0.0

        # Используем коэффициент Жаккара
        relevance = len(common_words) / (len(content_words.union(answer_words)))

        return relevance

    def _determine_citation_position(
        self, engine_data: Dict[str, Any], content: str
    ) -> Dict[str, Any]:
        """
        Определяет позицию цитирования контента в ответе.

        Args:
            engine_data: Данные ответа поисковика
            content: Контент, для которого проводится анализа

        Returns:
            Dict[str, Any]: Информация о позиции цитирования
        """
        # Позиция по умолчанию - отсутствие цитирования
        position_info = {
            "is_cited": False,
            "position": "none",
            "position_index": -1,
            "section_title": "",
            "context": "",
        }

        # Если ответ не содержит цитирования, возвращаем позицию по умолчанию
        if not self._check_content_citation(engine_data, content):
            return position_info

        # Получаем ответ поисковика
        answer = engine_data.get("answer", "")

        # Извлекаем ключевые фразы из контента
        content_sentences = re.split(r"[.!?]", content)
        content_sentences = [s.strip() for s in content_sentences if s.strip()]
        key_phrases = content_sentences[:3] + content_sentences[-3:]

        # Нормализуем фразы для поиска
        normalized_phrases = [
            re.sub(r"\s+", " ", phrase.lower()) for phrase in key_phrases if len(phrase) >= 10
        ]
        normalized_answer = re.sub(r"\s+", " ", answer.lower())

        # Ищем позицию цитирования
        citation_positions = []

        for phrase in normalized_phrases:
            pos = normalized_answer.find(phrase)
            if pos >= 0:
                citation_positions.append(pos)

        # Если цитирование найдено
        if citation_positions:
            # Берем первую позицию цитирования
            first_pos = min(citation_positions)

            # Определяем относительную позицию в ответе
            relative_pos = first_pos / len(normalized_answer)

            # Определяем текстовую позицию
            if relative_pos < 0.33:
                position = "начало"
            elif relative_pos < 0.66:
                position = "середина"
            else:
                position = "конец"

            # Определяем, в каком разделе находится цитирование
            section_title = ""
            for i, section in enumerate(engine_data.get("sections", [])):
                section_content = section.get("content", "").lower()

                for phrase in normalized_phrases:
                    if phrase in section_content:
                        section_title = section.get("title", f"Раздел {i+1}")
                        break

                if section_title:
                    break

            # Извлекаем контекст цитирования (50 символов до и после первой найденной фразы)
            start_idx = max(0, first_pos - 50)
            end_idx = min(len(normalized_answer), first_pos + len(normalized_phrases[0]) + 50)
            context = normalized_answer[start_idx:end_idx]

            # Обновляем информацию о позиции
            position_info = {
                "is_cited": True,
                "position": position,
                "position_index": first_pos,
                "relative_position": relative_pos,
                "section_title": section_title,
                "context": context,
            }

        return position_info

    def _calculate_visibility_score(self, samples_results: List[Dict[str, Any]]) -> float:
        """
        Рассчитывает оценку видимости контента в результатах поисковика.

        Args:
            samples_results: Список результатов анализа образцов

        Returns:
            float: Оценка видимости от 0 до 1
        """
        if not samples_results:
            return 0.0

        # Количество образцов с цитированием
        citation_count = sum(
            1 for result in samples_results if result.get("contains_citation", False)
        )

        # Расчет базовой видимости - доля образцов с цитированием
        base_visibility = citation_count / len(samples_results)

        # Учитываем позицию цитирования - цитирование в начале ценнее
        position_weight = 0
        for result in samples_results:
            if not result.get("contains_citation", False):
                continue

            position_info = result.get("citation_position", {})
            position = position_info.get("position", "none")

            # Назначаем вес в зависимости от позиции
            if position == "начало":
                position_weight += 1.0
            elif position == "середина":
                position_weight += 0.7
            elif position == "конец":
                position_weight += 0.4

        # Нормализуем вес позиции
        if citation_count > 0:
            position_weight /= citation_count

        # Учитываем релевантность
        avg_relevance = sum(result.get("relevance_score", 0) for result in samples_results) / len(
            samples_results
        )

        # Рассчитываем итоговую оценку видимости (60% - наличие цитирования, 25% - позиция, 15% - релевантность)
        visibility_score = (
            0.6 * base_visibility + 0.25 * position_weight * base_visibility + 0.15 * avg_relevance
        )

        return visibility_score

    def compare_serp_results(
        self,
        query: str,
        contents: List[str],
        llm_engines: Optional[List[str]] = None,
        num_samples: int = 2,
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Сравнивает результаты LLM-поисковиков для нескольких вариантов контента.

        Args:
            query: Поисковый запрос
            contents: Список вариантов контента для сравнения
            llm_engines: Список LLM-поисковиков для анализа
            num_samples: Количество запросов для каждого поисковика
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            Dict[str, Any]: Результаты сравнения SERP
        """
        # Оцениваем результаты SERP для каждого варианта контента
        variants_results = []

        # Распределяем бюджет между вариантами
        variant_budget = None
        if budget is not None:
            variant_budget = budget / len(contents)

        for i, content in enumerate(contents):
            self.logger.info(f"Анализ варианта {i+1} из {len(contents)}")

            # Анализируем SERP для варианта
            result = self.analyze_serp(query, content, llm_engines, num_samples, variant_budget)
            variants_results.append(result)

        # Определяем лучший вариант по видимости
        best_variant_index = max(
            range(len(variants_results)),
            key=lambda i: variants_results[i].get("visibility_score", 0),
        )

        best_variant = variants_results[best_variant_index]

        # Собираем статистику по токенам и стоимости
        total_tokens = sum(result.get("tokens", {}).get("total", 0) for result in variants_results)
        total_cost = sum(result.get("cost", 0) for result in variants_results)

        # Формируем результат сравнения
        comparison_result = {
            "query": query,
            "variants_count": len(contents),
            "engines": llm_engines,
            "samples_per_engine": num_samples,
            "best_variant": {
                "index": best_variant_index,
                "visibility_score": best_variant.get("visibility_score", 0),
                "citation_rate": best_variant.get("citation_rate", 0),
            },
            "visibility_scores": [result.get("visibility_score", 0) for result in variants_results],
            "citation_rates": [result.get("citation_rate", 0) for result in variants_results],
            "variants_results": variants_results,
            "tokens": {"total": total_tokens},
            "cost": total_cost,
        }

        return comparison_result
