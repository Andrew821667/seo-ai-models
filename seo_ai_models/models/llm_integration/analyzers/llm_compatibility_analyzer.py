"""
Анализатор совместимости контента с требованиями LLM.

Модуль предоставляет функционал для оценки совместимости контента
с требованиями различных LLM-систем и рекомендаций по улучшению.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from ..service.multi_model_agent import MultiModelAgent
from ..common.utils import (
    extract_scores_from_text,
    parse_json_response,
    chunk_text
)


class LLMCompatibilityAnalyzer:
    """
    Анализатор совместимости контента с требованиями LLM.
    """
    
    def __init__(self, llm_service: LLMService, prompt_generator: PromptGenerator):
        """
        Инициализирует анализатор совместимости.
        
        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM
            prompt_generator: Экземпляр PromptGenerator для генерации промптов
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        self.multi_model_agent = MultiModelAgent(llm_service, prompt_generator)
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
    
    def analyze_compatibility(self, content: str, llm_types: Optional[List[str]] = None, 
                            budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Анализирует совместимость контента с требованиями LLM.
        
        Args:
            content: Текст для анализа
            llm_types: Список типов LLM для анализа (опционально)
            budget: Максимальный бюджет в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Результат анализа совместимости
        """
        # Если типы LLM не указаны, используем стандартный набор
        if llm_types is None:
            llm_types = ["generic", "chat", "search"]
        
        # Если контент слишком большой, разбиваем его на чанки
        if len(content) > 15000:
            return self._analyze_large_content(content, llm_types, budget)
        
        # Генерируем промпт для анализа совместимости
        prompt = self._generate_compatibility_prompt(content, llm_types)
        
        # Используем MultiModelAgent для выбора оптимальной модели и анализа
        result = self.multi_model_agent.analyze_content(
            content, "compatibility", budget, use_multiple_models=False
        )
        
        # Обрабатываем результат анализа
        return self._process_compatibility_result(result, llm_types)
    
    def _generate_compatibility_prompt(self, content: str, llm_types: List[str]) -> str:
        """
        Генерирует промпт для анализа совместимости.
        
        Args:
            content: Текст для анализа
            llm_types: Список типов LLM для анализа
            
        Returns:
            str: Промпт для анализа
        """
        # Базовый промпт для анализа совместимости
        base_prompt = """
        Ты эксперт по оптимизации контента для работы с LLM (языковыми моделями).
        
        Проанализируй следующий текст на совместимость с LLM и оцени следующие аспекты по шкале от 1 до 10:
        
        1. Ясность и структурированность (насколько легко LLM может понять структуру)
        2. Информативность (насколько текст содержит конкретную, полезную информацию)
        3. Точность формулировок (насколько точно и однозначно сформулированы утверждения)
        4. Релевантность заявленной теме (насколько содержание соответствует заголовкам/теме)
        5. Цитируемость (насколько вероятно, что LLM будет цитировать этот контент в ответах)
        
        Также дай общую оценку совместимости с LLM по шкале от 1 до 10.
        
        Для каждого аспекта предложи конкретные улучшения, которые повысят совместимость с LLM.
        
        Сформируй результат в JSON формате со следующей структурой:
        {
            "scores": {
                "clarity": 0,
                "informativeness": 0,
                "precision": 0,
                "relevance": 0,
                "citability": 0,
                "overall": 0
            },
            "analysis": {
                "clarity": "Анализ ясности и структурированности",
                "informativeness": "Анализ информативности",
                "precision": "Анализ точности формулировок",
                "relevance": "Анализ релевантности",
                "citability": "Анализ цитируемости",
                "overall": "Общий анализ"
            },
            "improvements": {
                "clarity": ["Улучшение 1", "Улучшение 2"],
                "informativeness": ["Улучшение 1", "Улучшение 2"],
                "precision": ["Улучшение 1", "Улучшение 2"],
                "relevance": ["Улучшение 1", "Улучшение 2"],
                "citability": ["Улучшение 1", "Улучшение 2"]
            },
            "summary": "Краткое резюме анализа"
        }
        """
        
        # Дополнительные инструкции в зависимости от типов LLM
        llm_specific_instructions = ""
        
        if "chat" in llm_types:
            llm_specific_instructions += """
            Для чат-моделей особое внимание обрати на:
            - Простоту языка и отсутствие сложных конструкций
            - Четкость примеров и иллюстраций
            - Наличие ключевой информации в начале абзацев
            """
        
        if "search" in llm_types:
            llm_specific_instructions += """
            Для поисковых LLM (Perplexity, Claude Instant и др.) особое внимание обрати на:
            - Наличие четких ответов на потенциальные запросы пользователей
            - Структурированность информации (заголовки, списки, таблицы)
            - Фактическую точность и наличие цитируемых данных
            - Уникальность информации, которую трудно найти в других источниках
            """
        
        # Формируем финальный промпт
        final_prompt = base_prompt + llm_specific_instructions + "\n\nТекст для анализа:\n" + content
        
        return final_prompt
    
    def _process_compatibility_result(self, result: Dict[str, Any], 
                                    llm_types: List[str]) -> Dict[str, Any]:
        """
        Обрабатывает результат анализа совместимости.
        
        Args:
            result: Результат анализа от LLM
            llm_types: Список типов LLM для анализа
            
        Returns:
            Dict[str, Any]: Обработанный результат анализа
        """
        # Извлекаем текст ответа
        text = result.get("text", "")
        
        # Пытаемся извлечь JSON из ответа
        compatibility_data = parse_json_response(text)
        
        # Если не удалось извлечь JSON, пытаемся структурировать ответ сами
        if not compatibility_data or "scores" not in compatibility_data:
            compatibility_data = {
                "scores": extract_scores_from_text(text),
                "analysis": {},
                "improvements": {},
                "summary": ""
            }
            
            # Пытаемся извлечь анализ и улучшения из текста
            sections = re.split(r"\n\s*#{1,3}\s+", text)
            
            for section in sections:
                lines = section.strip().split("\n")
                if not lines:
                    continue
                
                section_title = lines[0].strip().lower()
                section_content = "\n".join(lines[1:]).strip()
                
                # Определяем тип секции (анализ или улучшения)
                if "ясност" in section_title or "структур" in section_title:
                    compatibility_data["analysis"]["clarity"] = section_content
                    compatibility_data["improvements"]["clarity"] = re.findall(r"[-*]\s*(.*?)(?:\n|$)", section_content)
                
                elif "информати" in section_title:
                    compatibility_data["analysis"]["informativeness"] = section_content
                    compatibility_data["improvements"]["informativeness"] = re.findall(r"[-*]\s*(.*?)(?:\n|$)", section_content)
                
                elif "точност" in section_title or "формулир" in section_title:
                    compatibility_data["analysis"]["precision"] = section_content
                    compatibility_data["improvements"]["precision"] = re.findall(r"[-*]\s*(.*?)(?:\n|$)", section_content)
                
                elif "релевант" in section_title:
                    compatibility_data["analysis"]["relevance"] = section_content
                    compatibility_data["improvements"]["relevance"] = re.findall(r"[-*]\s*(.*?)(?:\n|$)", section_content)
                
                elif "цитируе" in section_title:
                    compatibility_data["analysis"]["citability"] = section_content
                    compatibility_data["improvements"]["citability"] = re.findall(r"[-*]\s*(.*?)(?:\n|$)", section_content)
                
                elif "общ" in section_title or "итог" in section_title or "вывод" in section_title:
                    compatibility_data["analysis"]["overall"] = section_content
                    compatibility_data["summary"] = section_content
            
            # Добавляем исходный текст ответа
            compatibility_data["raw_text"] = text
        
        # Формируем итоговый результат
        analyzed_result = {
            "compatibility_scores": compatibility_data.get("scores", {}),
            "compatibility_analysis": compatibility_data.get("analysis", {}),
            "suggested_improvements": compatibility_data.get("improvements", {}),
            "summary": compatibility_data.get("summary", ""),
            "llm_types": llm_types,
            "provider": result.get("provider", ""),
            "model": result.get("model", ""),
            "tokens": result.get("tokens", {}),
            "cost": result.get("cost", 0)
        }
        
        return analyzed_result
    
    def _analyze_large_content(self, content: str, llm_types: List[str], 
                             budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Анализирует большой контент, разбивая его на чанки.
        
        Args:
            content: Текст для анализа
            llm_types: Список типов LLM для анализа
            budget: Максимальный бюджет в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Результат анализа совместимости
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
            self.logger.info(f"Анализ чанка {i+1} из {len(chunks)}")
            
            # Анализируем чанк
            chunk_result = self.analyze_compatibility(chunk, llm_types, chunk_budget)
            chunk_results.append(chunk_result)
        
        # Объединяем результаты анализа чанков
        return self._combine_chunk_results(chunk_results, content)
    
    def _combine_chunk_results(self, chunk_results: List[Dict[str, Any]], 
                             original_content: str) -> Dict[str, Any]:
        """
        Объединяет результаты анализа чанков.
        
        Args:
            chunk_results: Список результатов анализа чанков
            original_content: Исходный текст для анализа
            
        Returns:
            Dict[str, Any]: Объединенный результат анализа
        """
        # Объединяем оценки (среднее значение)
        combined_scores = {}
        
        # Списки для анализа и улучшений
        combined_analysis = {
            "clarity": [],
            "informativeness": [],
            "precision": [],
            "relevance": [],
            "citability": [],
            "overall": []
        }
        
        combined_improvements = {
            "clarity": [],
            "informativeness": [],
            "precision": [],
            "relevance": [],
            "citability": []
        }
        
        # Объединяем результаты чанков
        for result in chunk_results:
            # Объединяем оценки
            for key, value in result.get("compatibility_scores", {}).items():
                if key not in combined_scores:
                    combined_scores[key] = []
                combined_scores[key].append(value)
            
            # Объединяем анализ
            for key, value in result.get("compatibility_analysis", {}).items():
                if key in combined_analysis and value:
                    combined_analysis[key].append(value)
            
            # Объединяем улучшения
            for key, values in result.get("suggested_improvements", {}).items():
                if key in combined_improvements:
                    combined_improvements[key].extend(values)
        
        # Вычисляем средние оценки
        averaged_scores = {}
        for key, values in combined_scores.items():
            if values:
                averaged_scores[key] = sum(values) / len(values)
        
        # Удаляем дубликаты в улучшениях
        for key in combined_improvements:
            combined_improvements[key] = list(set(combined_improvements[key]))
        
        # Объединяем анализ в один текст
        unified_analysis = {}
        for key, values in combined_analysis.items():
            if values:
                unified_analysis[key] = "\n\n".join(values)
        
        # Формируем общий итог
        overall_summaries = combined_analysis.get("overall", [])
        summary = "\n\n".join(overall_summaries) if overall_summaries else ""
        
        # Если нет суммари, генерируем его из оценок
        if not summary and averaged_scores:
            overall_score = averaged_scores.get("overall", 0)
            if overall_score > 7:
                summary = "Контент хорошо оптимизирован для LLM, но есть возможности для улучшения."
            elif overall_score > 5:
                summary = "Контент умеренно оптимизирован для LLM, требуются существенные улучшения."
            else:
                summary = "Контент слабо оптимизирован для LLM, требуется комплексная переработка."
        
        # Собираем статистику по токенам и стоимости
        total_tokens = sum(result.get("tokens", {}).get("total", 0) for result in chunk_results)
        total_cost = sum(result.get("cost", 0) for result in chunk_results)
        
        # Формируем итоговый результат
        combined_result = {
            "compatibility_scores": averaged_scores,
            "compatibility_analysis": unified_analysis,
            "suggested_improvements": combined_improvements,
            "summary": summary,
            "llm_types": chunk_results[0].get("llm_types", []) if chunk_results else [],
            "chunks_analyzed": len(chunk_results),
            "original_content_length": len(original_content),
            "tokens": {"total": total_tokens},
            "cost": total_cost
        }
        
        return combined_result
