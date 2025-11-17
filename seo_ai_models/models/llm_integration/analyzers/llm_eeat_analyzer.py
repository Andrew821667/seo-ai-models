"""
Адаптация анализа E-E-A-T для LLM-оптимизации.

Модуль предоставляет функционал для анализа контента на соответствие
принципам E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)
с учетом особенностей LLM-поисковиков.
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


class LLMEEATAnalyzer:
    """
    Адаптация анализа E-E-A-T для LLM-оптимизации.
    """

    def __init__(self, llm_service: LLMService, prompt_generator: PromptGenerator):
        """
        Инициализирует анализатор E-E-A-T для LLM.

        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM
            prompt_generator: Экземпляр PromptGenerator для генерации промптов
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        self.multi_model_agent = MultiModelAgent(llm_service, prompt_generator)

        # Настройка логгирования
        self.logger = logging.getLogger(__name__)

        # YMYL категории
        self.ymyl_categories = [
            "health",
            "finance",
            "safety",
            "legal",
            "news",
            "shopping",
            "civic_information",
        ]

    def analyze_eeat(
        self, content: str, category: Optional[str] = None, budget: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Анализирует контент на соответствие принципам E-E-A-T для LLM.

        Args:
            content: Текст для анализа
            category: Категория контента (опционально)
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            Dict[str, Any]: Результат анализа E-E-A-T
        """
        # Если категория не указана, пытаемся определить ее
        if not category:
            category = self._detect_category(content, budget * 0.1 if budget else None)

        # Если контент слишком большой, разбиваем его на чанки
        if len(content) > 15000:
            return self._analyze_large_content_eeat(content, category, budget)

        # Генерируем промпт для анализа E-E-A-T
        prompt = self._generate_eeat_prompt(content, category)

        # Используем MultiModelAgent для выбора оптимальной модели и анализа
        result = self.multi_model_agent.analyze_content(
            content, "eeat", budget, use_multiple_models=False
        )

        # Обрабатываем результат анализа
        return self._process_eeat_result(result, category)

    def _detect_category(self, content: str, budget: Optional[float] = None) -> str:
        """
        Определяет категорию контента.

        Args:
            content: Текст для анализа
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            str: Определенная категория контента
        """
        # Усекаем контент для определения категории (используем начало и конец)
        max_sample_size = 5000
        if len(content) > max_sample_size:
            # Берем первую и последнюю часть контента
            start = content[: max_sample_size // 2]
            end = content[-max_sample_size // 2 :]
            sample = start + "\n...\n" + end
        else:
            sample = content

        # Выбираем оптимальную модель для определения категории
        provider, model = self.multi_model_agent.select_optimal_model(
            "classification", len(sample), budget
        )

        # Промпт для определения категории
        prompt = f"""
        Определи категорию контента из следующего списка:
        - general (общая информация)
        - health (здоровье, медицина)
        - finance (финансы, инвестиции)
        - safety (безопасность)
        - legal (юридическая информация)
        - news (новости, текущие события)
        - shopping (покупки, обзоры продуктов)
        - civic_information (гражданская информация)
        - tech (технологии)
        - education (образование)
        - entertainment (развлечения)
        
        Верни только название категории (например, "health") без дополнительных комментариев.
        
        Текст для анализа:
        {sample}
        """

        # Генерируем ответ
        result = self.llm_service.generate(prompt, provider)

        # Извлекаем категорию из текста ответа
        category_text = result.get("text", "").strip().lower()

        # Проверяем, соответствует ли ответ одной из категорий
        for category in [
            "general",
            "health",
            "finance",
            "safety",
            "legal",
            "news",
            "shopping",
            "civic_information",
            "tech",
            "education",
            "entertainment",
        ]:
            if category in category_text:
                return category

        # Если категория не определена, используем "general"
        return "general"

    def _generate_eeat_prompt(self, content: str, category: str) -> str:
        """
        Генерирует промпт для анализа E-E-A-T.

        Args:
            content: Текст для анализа
            category: Категория контента

        Returns:
            str: Промпт для анализа
        """
        # Определяем, является ли категория YMYL
        is_ymyl = category in self.ymyl_categories

        # Дополнительные инструкции для YMYL контента
        ymyl_instructions = ""
        if is_ymyl:
            ymyl_instructions = f"""
            Данный контент относится к категории YMYL ({category}), поэтому требует особого внимания 
            к достоверности, авторитетности и надежности информации.
            
            Для контента в категории YMYL особое внимание обрати на:
            - Наличие ссылок на авторитетные источники
            - Квалификацию автора в данной области
            - Точность и актуальность информации
            - Полноту раскрытия темы
            - Отсутствие потенциально вредных советов
            """

        # Базовый промпт для анализа E-E-A-T
        base_prompt = f"""
        Ты эксперт по E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness) в контексте LLM-оптимизации.
        
        Проанализируй следующий текст на соответствие принципам E-E-A-T, но с учетом специфики LLM-поисковиков.
        
        Оцени каждый компонент E-E-A-T по шкале от 1 до 10:
        
        1. Experience (Опыт): наличие признаков реального опыта в данной теме
           - Личный опыт автора
           - Примеры из практики
           - Конкретные кейсы и ситуации
        
        2. Expertise (Экспертиза): демонстрация экспертных знаний в области
           - Глубина и точность информации
           - Использование профессиональной терминологии
           - Понимание нюансов темы
        
        3. Authoritativeness (Авторитетность): насколько текст воспринимается как авторитетный источник
           - Ссылки на исследования и достоверные источники
           - Упоминание экспертов и авторитетов в области
           - Профессиональная подача материала
        
        4. Trustworthiness (Надежность): надежность, точность и достоверность информации
           - Отсутствие противоречий и неточностей
           - Сбалансированность мнений
           - Прозрачность и честность в представлении информации
        
        Также дай общую оценку E-E-A-T по шкале от 1 до 10.
        
        {ymyl_instructions}
        
        Для каждого компонента E-E-A-T предложи конкретные улучшения, которые повысят соответствие контента этому принципу.
        
        Проанализируй, как LLM-модели (вроде ChatGPT, Perplexity) воспримут этот контент с точки зрения E-E-A-T.
        
        Сформируй результат в JSON формате со следующей структурой:
        {{
            "scores": {{
                "experience": 0,
                "expertise": 0,
                "authoritativeness": 0,
                "trustworthiness": 0,
                "overall": 0
            }},
            "analysis": {{
                "experience": "Анализ компонента Experience",
                "expertise": "Анализ компонента Expertise",
                "authoritativeness": "Анализ компонента Authoritativeness",
                "trustworthiness": "Анализ компонента Trustworthiness",
                "overall": "Общий анализ E-E-A-T",
                "llm_perception": "Анализ восприятия LLM-моделями"
            }},
            "improvements": {{
                "experience": ["Улучшение 1", "Улучшение 2"],
                "expertise": ["Улучшение 1", "Улучшение 2"],
                "authoritativeness": ["Улучшение 1", "Улучшение 2"],
                "trustworthiness": ["Улучшение 1", "Улучшение 2"]
            }},
            "summary": "Краткое резюме анализа"
        }}
        """

        # Формируем финальный промпт
        final_prompt = base_prompt + "\n\nТекст для анализа:\n" + content

        return final_prompt

    def _process_eeat_result(self, result: Dict[str, Any], category: str) -> Dict[str, Any]:
        """
        Обрабатывает результат анализа E-E-A-T.

        Args:
            result: Результат анализа от LLM
            category: Категория контента

        Returns:
            Dict[str, Any]: Обработанный результат анализа
        """
        # Извлекаем текст ответа
        text = result.get("text", "")

        # Пытаемся извлечь JSON из ответа
        eeat_data = parse_json_response(text)

        # Если не удалось извлечь JSON, пытаемся структурировать ответ сами
        if not eeat_data or "scores" not in eeat_data:
            eeat_data = {
                "scores": {
                    "experience": 0,
                    "expertise": 0,
                    "authoritativeness": 0,
                    "trustworthiness": 0,
                    "overall": 0,
                },
                "analysis": {
                    "experience": "",
                    "expertise": "",
                    "authoritativeness": "",
                    "trustworthiness": "",
                    "overall": "",
                    "llm_perception": "",
                },
                # Продолжение файла llm_eeat_analyzer.py
                "improvements": {
                    "experience": [],
                    "expertise": [],
                    "authoritativeness": [],
                    "trustworthiness": [],
                },
                "summary": "",
            }

            # Извлекаем оценки компонентов E-E-A-T
            components = [
                "experience",
                "expertise",
                "authoritativeness",
                "trustworthiness",
                "overall",
            ]

            for component in components:
                component_pattern = f"{component}.*?(\d+)[^а-я0-9]*"
                component_match = re.search(component_pattern, text, re.IGNORECASE)
                if component_match:
                    eeat_data["scores"][component] = int(component_match.group(1))

            # Пытаемся извлечь анализ и улучшения из текста
            sections = re.split(r"\n\s*#{1,3}\s+", text)

            for section in sections:
                lines = section.strip().split("\n")
                if not lines:
                    continue

                section_title = lines[0].strip().lower()
                section_content = "\n".join(lines[1:]).strip()

                # Определяем тип секции
                for component in [
                    "experience",
                    "expertise",
                    "authoritativeness",
                    "trustworthiness",
                    "overall",
                ]:
                    if component in section_title.lower():
                        # Добавляем анализ
                        eeat_data["analysis"][component] = section_content

                        # Извлекаем улучшения, если это не общий анализ
                        if component != "overall":
                            eeat_data["improvements"][component] = re.findall(
                                r"[-*]\s*(.*?)(?:\n|$)", section_content
                            )

                # Также ищем анализ восприятия LLM
                if "llm" in section_title.lower() or "восприят" in section_title.lower():
                    eeat_data["analysis"]["llm_perception"] = section_content

            # Если нет суммари, формируем его из общего анализа
            if not eeat_data["summary"] and eeat_data["analysis"]["overall"]:
                eeat_data["summary"] = eeat_data["analysis"]["overall"]

            # Добавляем исходный текст ответа
            eeat_data["raw_text"] = text

        # Определяем, является ли категория YMYL
        is_ymyl = category in self.ymyl_categories

        # Формируем итоговый результат
        analyzed_result = {
            "eeat_scores": eeat_data.get("scores", {}),
            "eeat_analysis": eeat_data.get("analysis", {}),
            "suggested_improvements": eeat_data.get("improvements", {}),
            "summary": eeat_data.get("summary", ""),
            "category": category,
            "is_ymyl": is_ymyl,
            "provider": result.get("provider", ""),
            "model": result.get("model", ""),
            "tokens": result.get("tokens", {}),
            "cost": result.get("cost", 0),
        }

        return analyzed_result

    def _analyze_large_content_eeat(
        self, content: str, category: str, budget: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Анализирует E-E-A-T для большого контента, разбивая его на чанки.

        Args:
            content: Текст для анализа
            category: Категория контента
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            Dict[str, Any]: Результат анализа E-E-A-T
        """
        # Разбиваем контент на чанки
        chunks = chunk_text(content, max_chunk_size=10000, overlap=200)

        # Анализируем каждый чанк
        chunk_results = []

        # Распределяем бюджет на чанки
        chunk_budget = None
        if budget is not None:
            chunk_budget = budget / len(chunks)

        for i, chunk in enumerate(chunks):
            self.logger.info(f"Анализ E-E-A-T чанка {i+1} из {len(chunks)}")

            # Анализируем чанк
            chunk_result = self.analyze_eeat(chunk, category, chunk_budget)
            chunk_results.append(chunk_result)

        # Объединяем результаты анализа чанков
        return self._combine_eeat_chunk_results(chunk_results, content)

    def _combine_eeat_chunk_results(
        self, chunk_results: List[Dict[str, Any]], original_content: str
    ) -> Dict[str, Any]:
        """
        Объединяет результаты анализа E-E-A-T чанков.

        Args:
            chunk_results: Список результатов анализа чанков
            original_content: Исходный текст для анализа

        Returns:
            Dict[str, Any]: Объединенный результат анализа
        """
        # Если нет результатов, возвращаем пустой результат
        if not chunk_results:
            return {
                "eeat_scores": {},
                "eeat_analysis": {},
                "suggested_improvements": {},
                "summary": "Не удалось проанализировать E-E-A-T.",
                "category": "",
                "is_ymyl": False,
                "chunks_analyzed": 0,
                "original_content_length": len(original_content),
                "tokens": {"total": 0},
                "cost": 0,
            }

        # Категория контента и YMYL статус
        category = chunk_results[0].get("category", "")
        is_ymyl = chunk_results[0].get("is_ymyl", False)

        # Объединяем оценки (среднее значение)
        eeat_scores = {}

        for result in chunk_results:
            scores = result.get("eeat_scores", {})
            for key, value in scores.items():
                if key not in eeat_scores:
                    eeat_scores[key] = []
                eeat_scores[key].append(value)

        # Вычисляем средние оценки
        averaged_scores = {}
        for key, values in eeat_scores.items():
            if values:
                averaged_scores[key] = sum(values) / len(values)

        # Объединяем анализ
        eeat_analysis = {}

        for key in [
            "experience",
            "expertise",
            "authoritativeness",
            "trustworthiness",
            "overall",
            "llm_perception",
        ]:
            analyses = []
            for result in chunk_results:
                analysis = result.get("eeat_analysis", {}).get(key, "")
                if analysis:
                    analyses.append(analysis)

            if analyses:
                eeat_analysis[key] = "\n\n".join(analyses)

        # Объединяем предложения по улучшению
        suggested_improvements = {}

        for key in ["experience", "expertise", "authoritativeness", "trustworthiness"]:
            all_improvements = []
            for result in chunk_results:
                improvements = result.get("suggested_improvements", {}).get(key, [])
                all_improvements.extend(improvements)

            # Удаляем дубликаты
            suggested_improvements[key] = list(set(all_improvements))

        # Формируем общий итог
        all_summaries = [
            result.get("summary", "") for result in chunk_results if result.get("summary")
        ]
        summary = "\n\n".join(all_summaries) if all_summaries else ""

        # Если нет суммари, генерируем его из оценок
        if not summary:
            overall_score = averaged_scores.get("overall", 0)
            if overall_score > 7:
                summary = "Контент хорошо соответствует принципам E-E-A-T для LLM, но есть возможности для улучшения."
            elif overall_score > 5:
                summary = "Контент умеренно соответствует принципам E-E-A-T для LLM, требуются существенные улучшения."
            else:
                summary = "Контент слабо соответствует принципам E-E-A-T для LLM, требуется комплексная переработка."

        # Собираем статистику по токенам и стоимости
        total_tokens = sum(result.get("tokens", {}).get("total", 0) for result in chunk_results)
        total_cost = sum(result.get("cost", 0) for result in chunk_results)

        # Формируем итоговый результат
        combined_result = {
            "eeat_scores": averaged_scores,
            "eeat_analysis": eeat_analysis,
            "suggested_improvements": suggested_improvements,
            "summary": summary,
            "category": category,
            "is_ymyl": is_ymyl,
            "chunks_analyzed": len(chunk_results),
            "original_content_length": len(original_content),
            "tokens": {"total": total_tokens},
            "cost": total_cost,
        }

        return combined_result

    def compare_eeat(
        self, contents: List[str], category: Optional[str] = None, budget: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Сравнивает E-E-A-T для нескольких вариантов контента.

        Args:
            contents: Список вариантов текста для сравнения
            category: Категория контента (опционально)
            budget: Максимальный бюджет в рублях (опционально)

        Returns:
            Dict[str, Any]: Результат сравнения E-E-A-T
        """
        # Если категория не указана, определяем ее на основе первого варианта
        if not category:
            sample_content = contents[0]
            category = self._detect_category(sample_content, budget * 0.1 if budget else None)

        # Анализируем E-E-A-T для каждого варианта
        variant_results = []

        # Распределяем бюджет на варианты
        variant_budget = None
        if budget is not None:
            variant_budget = budget / len(contents)

        for i, content in enumerate(contents):
            self.logger.info(f"Анализ E-E-A-T варианта {i+1} из {len(contents)}")

            # Анализируем вариант
            result = self.analyze_eeat(content, category, variant_budget)
            variant_results.append(result)

        # Определяем лучший вариант по общей оценке E-E-A-T
        best_variant_index = max(
            range(len(variant_results)),
            key=lambda i: variant_results[i].get("eeat_scores", {}).get("overall", 0),
        )

        best_variant = variant_results[best_variant_index]

        # Определяем лучший вариант для каждого компонента E-E-A-T
        best_for_components = {}
        components = ["experience", "expertise", "authoritativeness", "trustworthiness"]

        for component in components:
            component_scores = []

            for i, result in enumerate(variant_results):
                score = result.get("eeat_scores", {}).get(component, 0)
                component_scores.append((i, score))

            # Сортируем по оценке
            component_scores.sort(key=lambda x: x[1], reverse=True)

            best_for_components[component] = {
                "variant_index": component_scores[0][0] if component_scores else 0,
                "score": component_scores[0][1] if component_scores else 0,
            }

        # Собираем статистику по токенам и стоимости
        total_tokens = sum(result.get("tokens", {}).get("total", 0) for result in variant_results)
        total_cost = sum(result.get("cost", 0) for result in variant_results)

        # Формируем результат сравнения
        comparison_result = {
            "variants_count": len(contents),
            "best_variant": {
                "index": best_variant_index,
                "overall_score": best_variant.get("eeat_scores", {}).get("overall", 0),
                "component_scores": {
                    component: best_variant.get("eeat_scores", {}).get(component, 0)
                    for component in components
                },
                "summary": best_variant.get("summary", ""),
            },
            "best_for_components": best_for_components,
            "variant_overall_scores": [
                result.get("eeat_scores", {}).get("overall", 0) for result in variant_results
            ],
            "category": category,
            "is_ymyl": category in self.ymyl_categories,
            "tokens": {"total": total_tokens},
            "cost": total_cost,
        }

        return comparison_result
