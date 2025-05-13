"""
Улучшение структуры контента для повышения цитируемости в LLM.

Модуль предоставляет функционал для анализа и улучшения структуры
контента с целью повышения его цитируемости в ответах LLM-моделей.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from ..service.multi_model_agent import MultiModelAgent
from ..common.utils import (
    parse_json_response,
    chunk_text
)


class ContentStructureEnhancer:
    """
    Улучшение структуры контента для повышения цитируемости в LLM.
    """
    
    def __init__(self, llm_service: LLMService, prompt_generator: PromptGenerator):
        """
        Инициализирует улучшитель структуры контента.
        
        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM
            prompt_generator: Экземпляр PromptGenerator для генерации промптов
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        self.multi_model_agent = MultiModelAgent(llm_service, prompt_generator)
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
    
    def analyze_structure(self, content: str, 
                        content_type: str = "article",
                        budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Анализирует структуру контента и предлагает улучшения.
        
        Args:
            content: Текст для анализа
            content_type: Тип контента (article, blog_post, product_description, etc.)
            budget: Максимальный бюджет в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Результат анализа структуры
        """
        # Если контент слишком большой, разбиваем его на чанки
        if len(content) > 15000:
            return self._analyze_large_content_structure(content, content_type, budget)
        
        # Генерируем промпт для анализа структуры
        prompt = self._generate_structure_analysis_prompt(content, content_type)
        
        # Используем MultiModelAgent для выбора оптимальной модели и анализа
        result = self.multi_model_agent.analyze_content(
            content, "structure", budget, use_multiple_models=False
        )
        
        # Обрабатываем результат анализа
        return self._process_structure_analysis_result(result, content_type)
    
    def enhance_structure(self, content: str, 
                        content_type: str = "article",
                        target_improvement: Optional[str] = None,
                        budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Улучшает структуру контента для повышения цитируемости.
        
        Args:
            content: Текст для улучшения
            content_type: Тип контента (article, blog_post, product_description, etc.)
            target_improvement: Целевое улучшение (headings, paragraphs, lists, tables, etc.)
            budget: Максимальный бюджет в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Результат улучшения структуры
        """
        # Сначала анализируем структуру
        analysis_result = self.analyze_structure(content, content_type, budget * 0.3 if budget else None)
        
        # Генерируем промпт для улучшения структуры
        prompt = self._generate_structure_enhancement_prompt(
            content, content_type, target_improvement, analysis_result
        )
        
        # Используем оптимальную модель для улучшения
        provider, model = self.multi_model_agent.select_optimal_model(
            "generation", len(content), budget * 0.7 if budget else None
        )
        
        # Генерируем улучшенный контент
        result = self.llm_service.generate(prompt, provider)
        
        # Обрабатываем результат улучшения
        return self._process_structure_enhancement_result(result, content, content_type, provider, model)
    
    def _generate_structure_analysis_prompt(self, content: str, content_type: str) -> str:
        """
        Генерирует промпт для анализа структуры.
        
        Args:
            content: Текст для анализа
            content_type: Тип контента
            
        Returns:
            str: Промпт для анализа
        """
        # Базовый промпт для анализа структуры
        base_prompt = f"""
        Ты эксперт по структурированию контента для максимальной эффективности в эру LLM-поисковиков.
        
        Проанализируй структуру представленного {content_type} и оцени следующие элементы по шкале от 1 до 10:
        
        1. Заголовки (наличие, иерархия, информативность)
        2. Абзацы (длина, связность, распределение ключевой информации)
        3. Списки (использование маркированных и нумерованных списков)
        4. Таблицы и структурированные данные (если применимо)
        5. Выделение ключевых моментов (наличие ключевых выводов, цитат, важной информации)
        
        Также дай общую оценку структуры по шкале от 1 до 10.
        
        Для каждого элемента предложи конкретные улучшения, которые сделают контент более структурированным и цитируемым.
        
        Сформируй результат в JSON формате со следующей структурой:
        {{
            "scores": {{
                "headings": 0,
                "paragraphs": 0,
                "lists": 0,
                "tables": 0,
                "highlights": 0,
                "overall": 0
            }},
            "analysis": {{
                "headings": "Анализ заголовков",
                "paragraphs": "Анализ абзацев",
                "lists": "Анализ списков",
                "tables": "Анализ таблиц и структурированных данных",
                "highlights": "Анализ выделения ключевых моментов",
                "overall": "Общий анализ структуры"
            }},
            "improvements": {{
                "headings": ["Улучшение 1", "Улучшение 2"],
                "paragraphs": ["Улучшение 1", "Улучшение 2"],
                "lists": ["Улучшение 1", "Улучшение 2"],
                "tables": ["Улучшение 1", "Улучшение 2"],
                "highlights": ["Улучшение 1", "Улучшение 2"]
            }},
            "summary": "Краткое резюме анализа"
        }}
        """
        
        # Формируем финальный промпт
        final_prompt = base_prompt + "\n\nТекст для анализа:\n" + content
        
        return final_prompt
    
    def _generate_structure_enhancement_prompt(self, content: str, content_type: str, 
                                            target_improvement: Optional[str],
                                            analysis_result: Dict[str, Any]) -> str:
        """
        Генерирует промпт для улучшения структуры.
        
        Args:
            content: Текст для улучшения
            content_type: Тип контента
            target_improvement: Целевое улучшение
            analysis_result: Результат анализа структуры
            
        Returns:
            str: Промпт для улучшения
        """
        # Определяем приоритетные улучшения на основе анализа
        improvements = []
        
        # Если указано целевое улучшение, добавляем его первым
        if target_improvement:
            improvements.append(target_improvement)
        
        # Добавляем остальные улучшения в порядке возрастания оценки
        scores = analysis_result.get("structure_scores", {})
        sorted_elements = sorted(scores.items(), key=lambda x: x[1])
        
        for element, _ in sorted_elements:
            if element != "overall" and element != target_improvement:
                improvements.append(element)
        
        # Формируем часть промпта с улучшениями
        improvements_prompt = "\n".join([
            f"{i+1}. {element.capitalize()}: {', '.join(analysis_result.get('suggested_improvements', {}).get(element, []))}"
            for i, element in enumerate(improvements)
        ])
        
        # Базовый промпт для улучшения структуры
        base_prompt = f"""
        Ты эксперт по структурированию контента для максимальной эффективности в эру LLM-поисковиков.
        
        Реструктурируй представленный {content_type}, чтобы сделать его более цитируемым и полезным для LLM-моделей.
        
        На основе проведенного анализа, необходимо внести следующие улучшения:
        
        {improvements_prompt}
        
        Сохрани весь важный смысл и информацию из оригинального контента. Внеси только структурные изменения, не добавляя новой информации от себя.
        
        Верни полную, улучшенную версию контента, сохраняя его изначальную тему и назначение.
        """
        
        # Формируем финальный промпт
        final_prompt = base_prompt + "\n\nОригинальный текст:\n" + content
        
        return final_prompt
    
    def _process_structure_analysis_result(self, result: Dict[str, Any], 
                                        content_type: str) -> Dict[str, Any]:
        """
        Обрабатывает результат анализа структуры.
        
        Args:
            result: Результат анализа от LLM
            content_type: Тип контента
            
        Returns:
            Dict[str, Any]: Обработанный результат анализа
        """
        # Извлекаем текст ответа
        text = result.get("text", "")
        
        # Пытаемся извлечь JSON из ответа
        structure_data = parse_json_response(text)
        
        # Если не удалось извлечь JSON, пытаемся структурировать ответ сами
        if not structure_data or "scores" not in structure_data:
            structure_data = {
                "scores": {
                    "headings": 0,
                    "paragraphs": 0,
                    "lists": 0,
                    "tables": 0,
                    "highlights": 0,
                    "overall": 0
                },
                "analysis": {
                    "headings": "",
                    "paragraphs": "",
                    "lists": "",
                    "tables": "",
                    "highlights": "",
                    "overall": ""
                },
                "improvements": {
                    "headings": [],
                    "paragraphs": [],
                    "lists": [],
                    "tables": [],
                    "highlights": []
                },
                "summary": ""
            }
            
            # Пытаемся извлечь оценки из текста
            for element in ["headings", "paragraphs", "lists", "tables", "highlights", "overall"]:
                pattern = f"{element}.*?(\d+)[^а-я0-9]*"
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    structure_data["scores"][element] = int(match.group(1))
            
            # Пытаемся извлечь анализ и улучшения из текста
            sections = re.split(r"\n\s*#{1,3}\s+", text)
            
            for section in sections:
                lines = section.strip().split("\n")
                if not lines:
                    continue
                
                section_title = lines[0].strip().lower()
                section_content = "\n".join(lines[1:]).strip()
                
                # Определяем тип секции
                for element in ["headings", "заголовки", "paragraphs", "абзацы", 
                               "lists", "списки", "tables", "таблицы", 
                               "highlights", "выделения", "overall", "общий"]:
                    if element in section_title.lower():
                        # Определяем, к какому элементу относится секция
                        if "заголов" in element:
                            element_key = "headings"
                        elif "абзац" in element:
                            element_key = "paragraphs"
                        elif "спис" in element:
                            element_key = "lists"
                        elif "табли" in element:
                            element_key = "tables"
                        elif "выдел" in element or "ключев" in element:
                            element_key = "highlights"
                        elif "общ" in element or "overall" in element:
                            element_key = "overall"
                        else:
                            continue
                        
                        # Добавляем анализ
                        structure_data["analysis"][element_key] = section_content
                        
                        # Извлекаем улучшения, если это не общий анализ
                        if element_key != "overall":
                            structure_data["improvements"][element_key] = re.findall(r"[-*]\s*(.*?)(?:\n|$)", section_content)
            
            # Если нет суммари, формируем его из общего анализа
            if not structure_data["summary"] and structure_data["analysis"]["overall"]:
                structure_data["summary"] = structure_data["analysis"]["overall"]
            
            # Добавляем исходный текст ответа
            structure_data["raw_text"] = text
        
        # Формируем итоговый результат
        analyzed_result = {
            "structure_scores": structure_data.get("scores", {}),
            "structure_analysis": structure_data.get("analysis", {}),
            "suggested_improvements": structure_data.get("improvements", {}),
            "summary": structure_data.get("summary", ""),
            "content_type": content_type,
            "provider": result.get("provider", ""),
            "model": result.get("model", ""),
            "tokens": result.get("tokens", {}),
            "cost": result.get("cost", 0)
        }
        
        return analyzed_result
    
    def _process_structure_enhancement_result(self, result: Dict[str, Any], 
                                           original_content: str,
                                           content_type: str,
                                           provider: str,
                                           model: str) -> Dict[str, Any]:
        """
        Обрабатывает результат улучшения структуры.
        
        Args:
            result: Результат генерации от LLM
            original_content: Исходный текст
            content_type: Тип контента
            provider: Провайдер LLM
            model: Модель LLM
            
        Returns:
            Dict[str, Any]: Обработанный результат улучшения
        """
        # Извлекаем текст ответа
        enhanced_content = result.get("text", "")
        
        # Очищаем от лишних комментариев и разметки
        clean_content = self._clean_enhanced_content(enhanced_content)
        
        # Анализируем изменения
        changes = self._analyze_structural_changes(original_content, clean_content)
        
        # Формируем итоговый результат
        enhanced_result = {
            "original_content": original_content,
            "enhanced_content": clean_content,
            "changes": changes,
            "content_type": content_type,
            "provider": provider,
            "model": model,
            "tokens": result.get("tokens", {}),
            "cost": result.get("cost", 0)
        }
        
        return enhanced_result
    
    def _clean_enhanced_content(self, content: str) -> str:
        """
        Очищает улучшенный контент от лишних комментариев и разметки.
        
        Args:
            content: Текст улучшенного контента
            
        Returns:
            str: Очищенный контент
        """
        # Удаляем вступительные и заключительные комментарии
        clean_content = re.sub(r"^(Вот улучшенная версия|Улучшенная структура|Вот реструктурированный).*?\n\n", "", content, flags=re.IGNORECASE)
        clean_content = re.sub(r"\n\n(Я сохранил|Я внес следующие изменения|Теперь контент).*?$", "", clean_content, flags=re.IGNORECASE)
        
        return clean_content.strip()
    
    def _analyze_structural_changes(self, original: str, enhanced: str) -> Dict[str, Any]:
        """
        Анализирует структурные изменения между исходным и улучшенным контентом.
        
        Args:
            original: Исходный текст
            enhanced: Улучшенный текст
            
        Returns:
            Dict[str, Any]: Анализ изменений
        """
        # Анализируем заголовки
        original_headings = re.findall(r"(^|\n)#{1,6}\s+(.+?)($|\n)", original)
        enhanced_headings = re.findall(r"(^|\n)#{1,6}\s+(.+?)($|\n)", enhanced)
        
        headings_changes = {
            "original_count": len(original_headings),
            "enhanced_count": len(enhanced_headings),
            "difference": len(enhanced_headings) - len(original_headings)
        }
        
        # Анализируем абзацы
        original_paragraphs = [p for p in re.split(r"\n\s*\n", original) if p.strip()]
        enhanced_paragraphs = [p for p in re.split(r"\n\s*\n", enhanced) if p.strip()]
        
        paragraphs_changes = {
            "original_count": len(original_paragraphs),
            "enhanced_count": len(enhanced_paragraphs),
            "difference": len(enhanced_paragraphs) - len(original_paragraphs),
            # Продолжение файла content_structure_enhancer.py
            "avg_original_length": sum(len(p) for p in original_paragraphs) / max(1, len(original_paragraphs)),
            "avg_enhanced_length": sum(len(p) for p in enhanced_paragraphs) / max(1, len(enhanced_paragraphs)),
        }
        
        # Анализируем списки
        original_lists = len(re.findall(r"(^|\n)[*-]\s+", original))
        enhanced_lists = len(re.findall(r"(^|\n)[*-]\s+", enhanced))
        
        original_numbered_lists = len(re.findall(r"(^|\n)\d+\.\s+", original))
        enhanced_numbered_lists = len(re.findall(r"(^|\n)\d+\.\s+", enhanced))
        
        lists_changes = {
            "original_count": original_lists + original_numbered_lists,
            "enhanced_count": enhanced_lists + enhanced_numbered_lists,
            "difference": (enhanced_lists + enhanced_numbered_lists) - (original_lists + original_numbered_lists),
            "original_bullet_lists": original_lists,
            "enhanced_bullet_lists": enhanced_lists,
            "original_numbered_lists": original_numbered_lists,
            "enhanced_numbered_lists": enhanced_numbered_lists
        }
        
        # Анализируем таблицы
        original_tables = len(re.findall(r"\n\|.+\|.+\|\n\|[-:]+\|[-:]+\|", original))
        enhanced_tables = len(re.findall(r"\n\|.+\|.+\|\n\|[-:]+\|[-:]+\|", enhanced))
        
        tables_changes = {
            "original_count": original_tables,
            "enhanced_count": enhanced_tables,
            "difference": enhanced_tables - original_tables
        }
        
        # Анализируем выделения
        original_bold = len(re.findall(r"\*\*(.+?)\*\*", original))
        enhanced_bold = len(re.findall(r"\*\*(.+?)\*\*", enhanced))
        
        original_italic = len(re.findall(r"\*(.+?)\*", original)) - original_bold * 2
        enhanced_italic = len(re.findall(r"\*(.+?)\*", enhanced)) - enhanced_bold * 2
        
        original_blockquotes = len(re.findall(r"(^|\n)>", original))
        enhanced_blockquotes = len(re.findall(r"(^|\n)>", enhanced))
        
        highlights_changes = {
            "original_bold": original_bold,
            "enhanced_bold": enhanced_bold,
            "original_italic": original_italic,
            "enhanced_italic": enhanced_italic,
            "original_blockquotes": original_blockquotes,
            "enhanced_blockquotes": enhanced_blockquotes,
            "difference": (enhanced_bold + enhanced_italic + enhanced_blockquotes) - 
                         (original_bold + original_italic + original_blockquotes)
        }
        
        # Общая статистика
        word_pattern = r"\b\w+\b"
        original_words = len(re.findall(word_pattern, original))
        enhanced_words = len(re.findall(word_pattern, enhanced))
        
        overall_changes = {
            "original_length": len(original),
            "enhanced_length": len(enhanced),
            "length_difference": len(enhanced) - len(original),
            "original_words": original_words,
            "enhanced_words": enhanced_words,
            "words_difference": enhanced_words - original_words,
            "structure_elements_added": (headings_changes["difference"] > 0) + 
                                      (paragraphs_changes["difference"] > 0) + 
                                      (lists_changes["difference"] > 0) + 
                                      (tables_changes["difference"] > 0) + 
                                      (highlights_changes["difference"] > 0)
        }
        
        return {
            "headings": headings_changes,
            "paragraphs": paragraphs_changes,
            "lists": lists_changes,
            "tables": tables_changes,
            "highlights": highlights_changes,
            "overall": overall_changes
        }
    
    def _analyze_large_content_structure(self, content: str, content_type: str, 
                                       budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Анализирует структуру большого контента, разбивая его на чанки.
        
        Args:
            content: Текст для анализа
            content_type: Тип контента
            budget: Максимальный бюджет в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Результат анализа структуры
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
            self.logger.info(f"Анализ структуры чанка {i+1} из {len(chunks)}")
            
            # Анализируем чанк
            chunk_result = self.analyze_structure(chunk, content_type, chunk_budget)
            chunk_results.append(chunk_result)
        
        # Объединяем результаты анализа чанков
        return self._combine_structure_chunk_results(chunk_results, content)
    
    def _combine_structure_chunk_results(self, chunk_results: List[Dict[str, Any]], 
                                      original_content: str) -> Dict[str, Any]:
        """
        Объединяет результаты анализа структуры чанков.
        
        Args:
            chunk_results: Список результатов анализа чанков
            original_content: Исходный текст для анализа
            
        Returns:
            Dict[str, Any]: Объединенный результат анализа
        """
        # Если нет результатов, возвращаем пустой результат
        if not chunk_results:
            return {
                "structure_scores": {},
                "structure_analysis": {},
                "suggested_improvements": {},
                "summary": "Не удалось проанализировать структуру.",
                "content_type": "",
                "chunks_analyzed": 0,
                "original_content_length": len(original_content),
                "tokens": {"total": 0},
                "cost": 0
            }
        
        # Тип контента
        content_type = chunk_results[0].get("content_type", "")
        
        # Объединяем оценки (среднее значение)
        structure_scores = {}
        
        for result in chunk_results:
            scores = result.get("structure_scores", {})
            for key, value in scores.items():
                if key not in structure_scores:
                    structure_scores[key] = []
                structure_scores[key].append(value)
        
        # Вычисляем средние оценки
        averaged_scores = {}
        for key, values in structure_scores.items():
            if values:
                averaged_scores[key] = sum(values) / len(values)
        
        # Объединяем анализ
        structure_analysis = {}
        
        for key in ["headings", "paragraphs", "lists", "tables", "highlights", "overall"]:
            analyses = []
            for result in chunk_results:
                analysis = result.get("structure_analysis", {}).get(key, "")
                if analysis:
                    analyses.append(analysis)
            
            if analyses:
                structure_analysis[key] = "\n\n".join(analyses)
        
        # Объединяем предложения по улучшению
        suggested_improvements = {}
        
        for key in ["headings", "paragraphs", "lists", "tables", "highlights"]:
            all_improvements = []
            for result in chunk_results:
                improvements = result.get("suggested_improvements", {}).get(key, [])
                all_improvements.extend(improvements)
            
            # Удаляем дубликаты
            suggested_improvements[key] = list(set(all_improvements))
        
        # Формируем общий итог
        all_summaries = [result.get("summary", "") for result in chunk_results if result.get("summary")]
        summary = "\n\n".join(all_summaries) if all_summaries else ""
        
        # Собираем статистику по токенам и стоимости
        total_tokens = sum(result.get("tokens", {}).get("total", 0) for result in chunk_results)
        total_cost = sum(result.get("cost", 0) for result in chunk_results)
        
        # Формируем итоговый результат
        combined_result = {
            "structure_scores": averaged_scores,
            "structure_analysis": structure_analysis,
            "suggested_improvements": suggested_improvements,
            "summary": summary,
            "content_type": content_type,
            "chunks_analyzed": len(chunk_results),
            "original_content_length": len(original_content),
            "tokens": {"total": total_tokens},
            "cost": total_cost
        }
        
        return combined_result
    
    def generate_structural_template(self, topic: str, content_type: str = "article", 
                                   keywords: Optional[List[str]] = None,
                                   length: str = "medium",
                                   budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Генерирует структурный шаблон для создания оптимизированного контента.
        
        Args:
            topic: Тема контента
            content_type: Тип контента (article, blog_post, product_description, etc.)
            keywords: Список ключевых слов (опционально)
            length: Длина контента (short, medium, long)
            budget: Максимальный бюджет в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Структурный шаблон
        """
        # Выбираем оптимальную модель для генерации
        provider, model = self.multi_model_agent.select_optimal_model(
            "generation", len(topic) * 10, budget
        )
        
        # Формируем промпт для генерации шаблона
        prompt = self._generate_template_prompt(topic, content_type, keywords, length)
        
        # Генерируем шаблон
        result = self.llm_service.generate(prompt, provider)
        
        # Обрабатываем результат генерации
        return self._process_template_result(result, topic, content_type, keywords, provider, model)
    
    def _generate_template_prompt(self, topic: str, content_type: str, 
                               keywords: Optional[List[str]] = None,
                               length: str = "medium") -> str:
        """
        Генерирует промпт для создания структурного шаблона.
        
        Args:
            topic: Тема контента
            content_type: Тип контента
            keywords: Список ключевых слов
            length: Длина контента
            
        Returns:
            str: Промпт для генерации шаблона
        """
        # Формируем часть промпта с ключевыми словами
        keywords_prompt = ""
        if keywords:
            keywords_prompt = f"\nКлючевые слова: {', '.join(keywords)}"
        
        # Определяем длину контента
        length_guide = {
            "short": "Краткий контент (до 500 слов)",
            "medium": "Средний по длине контент (500-1500 слов)",
            "long": "Объемный контент (более 1500 слов)"
        }.get(length, "Средний по длине контент (500-1500 слов)")
        
        # Базовый промпт для генерации шаблона
        base_prompt = f"""
        Ты эксперт по структурированию контента для максимальной эффективности в эру LLM-поисковиков.
        
        Создай структурный шаблон для {content_type} на тему "{topic}".
        {keywords_prompt}
        Длина: {length_guide}
        
        Шаблон должен содержать:
        
        1. Заголовок и подзаголовки (с указанием иерархии — H1, H2, H3)
        2. Структуру разделов с указанием, что должно содержаться в каждом разделе
        3. Рекомендации по организации абзацев
        4. Места для списков, таблиц и других структурных элементов
        5. Рекомендации по выделению ключевых моментов
        
        Структура должна быть оптимизирована для максимальной цитируемости в ответах LLM-моделей.
        
        Создай шаблон в Markdown, используя соответствующую разметку.
        """
        
        return base_prompt
    
    def _process_template_result(self, result: Dict[str, Any], 
                               topic: str, content_type: str, 
                               keywords: Optional[List[str]],
                               provider: str, model: str) -> Dict[str, Any]:
        """
        Обрабатывает результат генерации структурного шаблона.
        
        Args:
            result: Результат генерации от LLM
            topic: Тема контента
            content_type: Тип контента
            keywords: Список ключевых слов
            provider: Провайдер LLM
            model: Модель LLM
            
        Returns:
            Dict[str, Any]: Обработанный результат генерации
        """
        # Извлекаем текст шаблона
        template = result.get("text", "")
        
        # Очищаем от комментариев и разметки
        clean_template = self._clean_template(template)
        
        # Анализируем структуру шаблона
        structure_analysis = self._analyze_template_structure(clean_template)
        
        # Формируем итоговый результат
        template_result = {
            "template": clean_template,
            "structure_analysis": structure_analysis,
            "topic": topic,
            "content_type": content_type,
            "keywords": keywords or [],
            "provider": provider,
            "model": model,
            "tokens": result.get("tokens", {}),
            "cost": result.get("cost", 0)
        }
        
        return template_result
    
    def _clean_template(self, template: str) -> str:
        """
        Очищает шаблон от лишних комментариев и разметки.
        
        Args:
            template: Текст шаблона
            
        Returns:
            str: Очищенный шаблон
        """
        # Удаляем вступительные и заключительные комментарии
        clean_template = re.sub(r"^(Вот структурный шаблон|Шаблон для|Структурный шаблон).*?\n\n", "", template, flags=re.IGNORECASE)
        clean_template = re.sub(r"\n\n(Этот шаблон|Данный шаблон|Шаблон оптимизирован).*?$", "", clean_template, flags=re.IGNORECASE)
        
        return clean_template.strip()
    
    def _analyze_template_structure(self, template: str) -> Dict[str, Any]:
        """
        Анализирует структуру сгенерированного шаблона.
        
        Args:
            template: Текст шаблона
            
        Returns:
            Dict[str, Any]: Анализ структуры шаблона
        """
        # Анализируем заголовки
        headings = re.findall(r"(^|\n)#{1,6}\s+(.+?)($|\n)", template)
        h1_headings = re.findall(r"(^|\n)#\s+(.+?)($|\n)", template)
        h2_headings = re.findall(r"(^|\n)##\s+(.+?)($|\n)", template)
        h3_headings = re.findall(r"(^|\n)###\s+(.+?)($|\n)", template)
        
        # Анализируем комментарии в шаблоне
        comments = re.findall(r"(^|\n)>\s+(.+?)($|\n)", template)
        
        # Анализируем списки
        bullet_lists = re.findall(r"(^|\n)[*-]\s+(.+?)($|\n)", template)
        numbered_lists = re.findall(r"(^|\n)\d+\.\s+(.+?)($|\n)", template)
        
        # Анализируем таблицы
        tables = re.findall(r"\n\|.+\|.+\|\n\|[-:]+\|[-:]+\|", template)
        
        # Анализируем плейсхолдеры
        placeholders = re.findall(r"\[(.+?)\]", template)
        
        return {
            "headings_count": len(headings),
            "h1_count": len(h1_headings),
            "h2_count": len(h2_headings),
            "h3_count": len(h3_headings),
            "comments_count": len(comments),
            "bullet_lists_count": len(bullet_lists),
            "numbered_lists_count": len(numbered_lists),
            "tables_count": len(tables),
            "placeholders_count": len(placeholders),
            "template_length": len(template),
            "sections": len(h2_headings) + len(h3_headings)
        }
