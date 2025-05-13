"""
Извлечение семантической структуры для анализа LLM.

Модуль предоставляет функционал для извлечения семантической структуры
контента, что позволяет анализировать его для LLM-оптимизации.
"""

import re
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD, NMF
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from ..service.multi_model_agent import MultiModelAgent
from ..common.utils import (
    parse_json_response,
    chunk_text
)


class SemanticStructureExtractor:
    """
    Извлечение семантической структуры для анализа LLM.
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None, 
               prompt_generator: Optional[PromptGenerator] = None):
        """
        Инициализирует экстрактор семантической структуры.
        
        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM (опционально)
            prompt_generator: Экземпляр PromptGenerator для генерации промптов (опционально)
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        
        if llm_service and prompt_generator:
            self.multi_model_agent = MultiModelAgent(llm_service, prompt_generator)
        else:
            self.multi_model_agent = None
        
        # Проверяем наличие scikit-learn
        if not HAS_SKLEARN:
            self.logger.warning("scikit-learn не установлен. Функциональность будет ограничена.")
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
    
    def extract_semantic_structure(self, content: str, 
                                 method: str = "hybrid",
                                 n_topics: int = 5,
                                 budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Извлекает семантическую структуру из контента.
        
        Args:
            content: Текст для анализа
            method: Метод извлечения (hybrid, llm, statistical)
            n_topics: Количество тем для извлечения
            budget: Максимальный бюджет в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Семантическая структура контента
        """
        # Проверяем доступность выбранного метода
        if method == "llm" and self.multi_model_agent is None:
            self.logger.warning("LLM метод недоступен. Используем статистический метод.")
            method = "statistical"
        
        if method == "statistical" and not HAS_SKLEARN:
            self.logger.warning("Статистический метод недоступен. Используем LLM метод.")
            if self.multi_model_agent is None:
                raise RuntimeError("Ни один из методов не доступен. Установите scikit-learn или предоставьте LLMService.")
            method = "llm"
        
        # Если контент слишком большой, разбиваем его на чанки
        if len(content) > 15000 and method in ["hybrid", "llm"]:
            return self._extract_large_content_structure(content, method, n_topics, budget)
        
        # Используем выбранный метод
        if method == "llm":
            return self._extract_structure_llm(content, n_topics, budget)
        elif method == "statistical":
            return self._extract_structure_statistical(content, n_topics)
        else:  # hybrid
            # Комбинируем результаты обоих методов
            llm_result = self._extract_structure_llm(content, n_topics, budget * 0.7 if budget else None)
            stat_result = self._extract_structure_statistical(content, n_topics)
            
            return self._combine_extraction_results(llm_result, stat_result)
    
    def _extract_structure_llm(self, content: str, n_topics: int, 
                            budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Извлекает семантическую структуру с помощью LLM.
        
        Args:
            content: Текст для анализа
            n_topics: Количество тем для извлечения
            budget: Максимальный бюджет в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Семантическая структура контента
        """
        # Генерируем промпт для извлечения структуры
        prompt = self._generate_structure_extraction_prompt(content, n_topics)
        
        # Используем MultiModelAgent для выбора оптимальной модели и анализа
        result = self.multi_model_agent.analyze_content(
            content, "semantic", budget, use_multiple_models=False
        )
        
        # Обрабатываем результат анализа
        return self._process_structure_extraction_result(result, content)
    
    def _extract_structure_statistical(self, content: str, n_topics: int) -> Dict[str, Any]:
        """
        Извлекает семантическую структуру с помощью статистических методов.
        
        Args:
            content: Текст для анализа
            n_topics: Количество тем для извлечения
            
        Returns:
            Dict[str, Any]: Семантическая структура контента
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn не установлен. Невозможно использовать статистический метод.")
        
        # Разбиваем текст на абзацы
        paragraphs = [p for p in re.split(r"\n\s*\n", content) if p.strip()]
        
        # Если абзацев слишком мало, разбиваем текст на предложения
        if len(paragraphs) < n_topics * 2:
            sentences = re.split(r'(?<=[.!?])\s+', content)
            paragraphs = sentences
        
        # Если текст слишком короткий, возвращаем весь текст как одну тему
        if len(paragraphs) < 3:
            return {
                "topics": [{"name": "Основная тема", "keywords": [], "paragraphs": [content]}],
                "keywords": [],
                "semantic_clusters": [],
                "content_summary": content[:200] + "..." if len(content) > 200 else content,
                "extraction_method": "statistical",
                "topic_coherence": 0.0
            }
        
        # Создаем TF-IDF векторизатор
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',  # Для русского текста нужно будет использовать другие стоп-слова
            ngram_range=(1, 2)
        )
        
        # Трансформируем текст в TF-IDF матрицу
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        
        # Извлекаем темы с помощью NMF
        nmf_model = NMF(n_components=n_topics, random_state=42)
        nmf_topics = nmf_model.fit_transform(tfidf_matrix)
        
        # Получаем наиболее важные слова для каждой темы
        feature_names = vectorizer.get_feature_names_out()
        topic_keywords = []
        
        for topic_idx, topic in enumerate(nmf_model.components_):
            top_keywords_idx = topic.argsort()[:-11:-1]
            top_keywords = [feature_names[i] for i in top_keywords_idx]
            topic_keywords.append(top_keywords)
        
        # Определяем принадлежность абзацев к темам
        paragraph_topics = nmf_topics.argmax(axis=1)
        
        # Формируем темы
        topics = []
        for topic_idx in range(n_topics):
            topic_paragraphs = [paragraphs[i] for i in range(len(paragraphs)) if paragraph_topics[i] == topic_idx]
            
            # Если для темы нет абзацев, пропускаем ее
            if not topic_paragraphs:
                continue
            
            # Генерируем название темы на основе ключевых слов
            topic_name = " & ".join(topic_keywords[topic_idx][:3])
            
            topics.append({
                "name": topic_name,
                "keywords": topic_keywords[topic_idx],
                "paragraphs": topic_paragraphs
            })
        
        # Вычисляем когерентность тем
        topic_coherence = self._calculate_topic_coherence(nmf_topics)
        
        # Извлекаем общие ключевые слова
        all_keywords = set()
        for keywords in topic_keywords:
            all_keywords.update(keywords)
        
        # Формируем семантические кластеры
        semantic_clusters = []
        for i, keywords in enumerate(topic_keywords):
            semantic_clusters.append({
                "name": f"Кластер {i+1}",
                "keywords": keywords,
                "related_clusters": []
            })
        
        # Находим связанные кластеры
        topic_similarities = cosine_similarity(nmf_model.components_)
        
        for i in range(n_topics):
            # Находим наиболее похожие темы
            similar_topics = topic_similarities[i].argsort()[:-1][::-1]
            related_clusters = []
            
            for topic_idx in similar_topics[:2]:  # Берем 2 наиболее похожие темы
                if topic_idx != i:
                    related_clusters.append({
                        "name": f"Кластер {topic_idx+1}",
                        "similarity": topic_similarities[i, topic_idx]
                    })
            
            semantic_clusters[i]["related_clusters"] = related_clusters
        
        # Формируем краткое резюме контента
        content_summary = " ".join(paragraphs[:3]) if len(paragraphs) > 3 else content
        if len(content_summary) > 200:
            content_summary = content_summary[:197] + "..."
        
        return {
            "topics": topics,
            "keywords": list(all_keywords)[:50],  # Ограничиваем 50 ключевыми словами
            "semantic_clusters": semantic_clusters,
            "content_summary": content_summary,
            "extraction_method": "statistical",
            "topic_coherence": topic_coherence
        }
    
    def _calculate_topic_coherence(self, topic_document_matrix: np.ndarray) -> float:
        """
        Вычисляет когерентность тем.
        
        Args:
            topic_document_matrix: Матрица тем-документов
            
        Returns:
            float: Когерентность тем
        """
        # Вычисляем энтропию распределения тем
        topic_probs = topic_document_matrix / topic_document_matrix.sum(axis=1, keepdims=True)
        topic_entropy = -np.sum(topic_probs * np.log2(topic_probs + 1e-9)) / topic_probs.shape[0]
        
        # Нормализуем к [0, 1], где 1 - высокая когерентность
        coherence = 1.0 - topic_entropy / np.log2(topic_probs.shape[1])
        
        return float(coherence)
    
    def _generate_structure_extraction_prompt(self, content: str, n_topics: int) -> str:
        """
        Генерирует промпт для извлечения семантической структуры.
        
        Args:
            content: Текст для анализа
            n_topics: Количество тем для извлечения
            
        Returns:
            str: Промпт для извлечения
        """
        # Базовый промпт для извлечения семантической структуры
        base_prompt = f"""
        Ты эксперт по семантическому анализу в контексте поисковой оптимизации для LLM.
        
        Проведи глубокий семантический анализ представленного текста:
        
        1. Выдели {n_topics} основных тематических кластеров в тексте
        2. Для каждого кластера определи:
           - Название кластера
           - 5-10 ключевых слов/фраз, характеризующих кластер
           - Основные абзацы, относящиеся к этому кластеру
        
        3. Определи основные семантические сущности и их взаимосвязи
        4. Оцени семантическую плотность и релевантность основной теме
        5. Выяви потенциальные семантические пробелы
        
        Сформируй результат в JSON формате со следующей структурой:
        {{
            "topics": [
                {{
                    "name": "Название темы 1",
                    "keywords": ["ключевое слово 1", "ключевое слово 2", ...],
                    "paragraphs": ["текст абзаца 1", "текст абзаца 2", ...]
                }},
                ...
            ],
            "keywords": ["общее ключевое слово 1", "общее ключевое слово 2", ...],
            "semantic_clusters": [
                {{
                    "name": "Название кластера 1",
                    "keywords": ["ключевое слово 1", "ключевое слово 2", ...],
                    "related_clusters": [
                        {{"name": "Название связанного кластера", "relation": "тип связи"}}
                    ]
                }},
                ...
            ],
            "content_summary": "Краткое резюме содержания",
            "semantic_gaps": ["семантический пробел 1", "семантический пробел 2", ...]
        }}
        """
        
        # Формируем финальный промпт
        final_prompt = base_prompt + "\n\nТекст для анализа:\n" + content
        
        return final_prompt
    
    def _process_structure_extraction_result(self, result: Dict[str, Any], 
                                          content: str) -> Dict[str, Any]:
        """
        Обрабатывает результат извлечения семантической структуры.
        
        Args:
            result: Результат анализа от LLM
            content: Исходный текст для анализа
            
        Returns:
            Dict[str, Any]: Обработанный результат анализа
        """
        # Извлекаем текст ответа
        text = result.get("text", "")
        
        # Пытаемся извлечь JSON из ответа
        structure_data = parse_json_response(text)
        
        # Если не удалось извлечь JSON, пытаемся структурировать ответ сами
        if not structure_data or "topics" not in structure_data:
            # Разбиваем текст на абзацы
            paragraphs = [p for p in re.split(r"\n\s*\n", content) if p.strip()]
            
            structure_data = {
                "topics": [],
                "keywords": [],
                "semantic_clusters": [],
                "content_summary": "",
                "semantic_gaps": []
            }
            
            # Пытаемся извлечь темы из текста
            topic_sections = re.findall(r"(?:Тема|Кластер|Тематика)\s+\d+:?\s+(.+?)(?:\n|$)", text)
            
            for i, topic_title in enumerate(topic_sections):
                topic = {
                    "name": topic_title.strip(),
                    "keywords": [],
                    "paragraphs": []
                }
                
                # Ищем ключевые слова для темы
                keyword_section = re.search(r"(?:Ключевые слова|Keywords).*?:(.*?)(?:\n\n|\n[А-Я]|$)", text[text.find(topic_title):], re.DOTALL)
                
                if keyword_section:
                    keywords_text = keyword_section.group(1).strip()
                    keywords = re.findall(r"[«\"](.+?)[»\"]|[\w\-]+", keywords_text)
                    topic["keywords"] = keywords
                
                # Добавляем тему в список
                structure_data["topics"].append(topic)
                
                # Если тем слишком мало, распределяем абзацы поровну
                if len(topic_sections) <= 2:
                    chunk_size = len(paragraphs) // max(1, len(topic_sections))
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size if i < len(topic_sections) - 1 else len(paragraphs)
                    topic["paragraphs"] = paragraphs[start_idx:end_idx]
            
            # Если не нашли ни одной темы, создаем одну общую
            if not structure_data["topics"]:
                structure_data["topics"].append({
                    "name": "Основная тема",
                    "keywords": [],
                    "paragraphs": paragraphs
                })
            
            # Извлекаем общие ключевые слова
            keyword_section = re.search(r"(?:Общие ключевые слова|Keywords|Ключевые слова).*?:(.*?)(?:\n\n|\n[А-Я]|$)", text, re.DOTALL)
            
            if keyword_section:
                keywords_text = keyword_section.group(1).strip()
                keywords = re.findall(r"[«\"](.+?)[»\"]|[\w\-]+", keywords_text)
                structure_data["keywords"] = keywords
            
            # Извлекаем семантические кластеры
            cluster_sections = re.findall(r"(?:Семантический кластер|Cluster)\s+\d+:?\s+(.+?)(?:\n|$)", text)
            
            for cluster_title in cluster_sections:
                cluster = {
                    "name": cluster_title.strip(),
                    "keywords": [],
                    "related_clusters": []
                }
                
                # Ищем ключевые слова для кластера
                keyword_section = re.search(r"(?:Ключевые слова|Keywords).*?:(.*?)(?:\n\n|\n[А-Я]|$)", text[text.find(cluster_title):], re.DOTALL)
                
                if keyword_section:
                    keywords_text = keyword_section.group(1).strip()
                    keywords = re.findall(r"[«\"](.+?)[»\"]|[\w\-]+", keywords_text)
                    cluster["keywords"] = keywords
                
                # Добавляем кластер в список
                structure_data["semantic_clusters"].append(cluster)
            
            # Если не нашли ни одного кластера, используем темы
            if not structure_data["semantic_clusters"]:
                for i, topic in enumerate(structure_data["topics"]):
                    structure_data["semantic_clusters"].append({
                        "name": topic["name"],
                        "keywords": topic["keywords"],
                        "related_clusters": []
                    })
            
            # Извлекаем семантические пробелы
            gaps_section = re.search(r"(?:Семантические пробелы|Gaps).*?:(.*?)(?:\n\n|\n[А-Я]|$)", text, re.DOTALL)
            
            if gaps_section:
                gaps_text = gaps_section.group(1).strip()
                gaps = re.findall(r"[*-]\s*(.+?)(?:\n|$)", gaps_text)
                structure_data["semantic_gaps"] = gaps
            
            # Извлекаем резюме контента
            summary_section = re.search(r"(?:Резюме|Summary).*?:(.*?)(?:\n\n|\n[А-Я]|$)", text, re.DOTALL)
            
            if summary_section:
                structure_data["content_summary"] = summary_section.group(1).strip()
            else:
                # Если не нашли резюме, используем начало текста
                structure_data["content_summary"] = content[:200] + "..." if len(content) > 200 else content
            
            # Добавляем исходный текст ответа
            structure_data["raw_text"] = text
        
        # Добавляем метод извлечения и другие метаданные
        extraction_result = {
            "topics": structure_data.get("topics", []),
            "keywords": structure_data.get("keywords", []),
            "semantic_clusters": structure_data.get("semantic_clusters", []),
            "content_summary": structure_data.get("content_summary", ""),
            "semantic_gaps": structure_data.get("semantic_gaps", []),
            "extraction_method": "llm",
            "provider": result.get("provider", ""),
            "model": result.get("model", ""),
            "tokens": result.get("tokens", {}),
            "cost": result.get("cost", 0)
        }
        
        return extraction_result
    
    def _extract_large_content_structure(self, content: str, method: str, 
                                       n_topics: int,
                                       budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Извлекает семантическую структуру из большого контента, разбивая его на чанки.
        
        Args:
            content: Текст для анализа
            method: Метод извлечения
            n_topics: Количество тем для извлечения
            budget: Максимальный бюджет в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Семантическая структура контента
        """
        # Если метод статистический, используем его напрямую
        if method == "statistical":
            return self._extract_structure_statistical(content, n_topics)
        
        # Разбиваем контент на чанки
        chunks = chunk_text(content, max_chunk_size=10000, overlap=200)
        
        # Извлекаем структуру из каждого чанка
        chunk_results = []
        
        # Распределяем бюджет на чанки
        chunk_budget = None
        if budget is not None:
            chunk_budget = budget / len(chunks)
        
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Извлечение структуры из чанка {i+1} из {len(chunks)}")
            
            # Извлекаем структуру из чанка с помощью LLM
            result = self._extract_structure_llm(chunk, n_topics // 2, chunk_budget)
            chunk_results.append(result)
        
        # Объединяем результаты анализа чанков
        combined_llm_result = self._combine_extraction_chunk_results(chunk_results, content)
        
        # Если метод гибридный, дополняем статистическим анализом
        if method == "hybrid":
            stat_result = self._extract_structure_statistical(content, n_topics)
            return self._combine_extraction_results(combined_llm_result, stat_result)
        
        return combined_llm_result
    
    def _combine_extraction_chunk_results(self, chunk_results: List[Dict[str, Any]], 
                                       original_content: str) -> Dict[str, Any]:
        """
        Объединяет результаты извлечения структуры из чанков.
        
        Args:
            chunk_results: Список результатов анализа чанков
            original_content: Исходный текст для анализа
            
        Returns:
            Dict[str, Any]: Объединенный результат анализа
        """
        # Если нет результатов, возвращаем пустой результат
        if not chunk_results:
            return {
                "topics": [],
                "keywords": [],
                "semantic_clusters": [],
                "content_summary": "",
                "semantic_gaps": [],
                "extraction_method": "llm",
                "chunks_analyzed": 0,
                "original_content_length": len(original_content),
                "tokens": {"total": 0},
                "cost": 0
            }
        
        # Объединяем темы
        all_topics = []
        for result in chunk_results:
            all_topics.extend(result.get("topics", []))
        
        # Объединяем похожие темы
        combined_topics = []
        processed_indices = set()
        
        for i, topic in enumerate(all_topics):
            if i in processed_indices:
                continue
            
            # Находим похожие темы
            similar_topics = [topic]
            for j, other_topic in enumerate(all_topics):
                if j == i or j in processed_indices:
                    continue
                
                # Проверяем сходство по названию и ключевым словам
                name_similarity = self._text_similarity(topic["name"], other_topic["name"])
                keyword_similarity = self._keyword_similarity(
                    topic.get("keywords", []), 
                    other_topic.get("keywords", [])
                )
                
                # Если темы похожи, объединяем их
                if name_similarity > 0.5 or keyword_similarity > 0.3:
                    similar_topics.append(other_topic)
                    processed_indices.add(j)
            
            # Объединяем похожие темы
            combined_topic = self._merge_topics(similar_topics)
            combined_topics.append(combined_topic)
        
        # Ограничиваем количество тем
        max_topics = min(10, len(combined_topics))
        combined_topics = combined_topics[:max_topics]
        
        # Объединяем ключевые слова
        all_keywords = []
        for result in chunk_results:
            all_keywords.extend(result.get("keywords", []))
        
        # Удаляем дубликаты и ограничиваем количество ключевых слов
        unique_keywords = list(set(all_keywords))
        keywords = unique_keywords[:50]  # Ограничиваем 50 ключевыми словами
        
        # Объединяем семантические кластеры
        all_clusters = []
        for result in chunk_results:
            all_clusters.extend(result.get("semantic_clusters", []))
        
        # Объединяем похожие кластеры
        combined_clusters = []
        processed_indices = set()
        
        for i, cluster in enumerate(all_clusters):
            if i in processed_indices:
                continue
            
            # Находим похожие кластеры
            similar_clusters = [cluster]
            for j, other_cluster in enumerate(all_clusters):
                if j == i or j in processed_indices:
                    continue
                
                # Проверяем сходство по названию и ключевым словам
                name_similarity = self._text_similarity(cluster["name"], other_cluster["name"])
                keyword_similarity = self._keyword_similarity(
                    cluster.get("keywords", []), 
                    other_cluster.get("keywords", [])
                )
                
                # Если кластеры похожи, объединяем их
                if name_similarity > 0.5 or keyword_similarity > 0.3:
                    similar_clusters.append(other_cluster)
                    processed_indices.add(j)
            
            # Объединяем похожие кластеры
            combined_cluster = self._merge_clusters(similar_clusters)
            combined_clusters.append(combined_cluster)
        
        # Ограничиваем количество кластеров
        max_clusters = min(10, len(combined_clusters))
        combined_clusters = combined_clusters[:max_clusters]
        
        # Обновляем связи между кластерами
        for cluster in combined_clusters:
            related = []
            for related_cluster in cluster.get("related_clusters", []):
                for target_cluster in combined_clusters:
                    if self._text_similarity(related_cluster.get("name", ""), target_cluster["name"]) > 0.5:
                        related.append({
                            "name": target_cluster["name"],
                            "relation": related_cluster.get("relation", "связан")
                        })
                        break
            
            cluster["related_clusters"] = related
        
        # Объединяем семантические пробелы
        all_gaps = []
        for result in chunk_results:
            all_gaps.extend(result.get("semantic_gaps", []))
        
        # Удаляем дубликаты
        unique_gaps = []
        for gap in all_gaps:
            is_duplicate = False
            for existing_gap in unique_gaps:
                if self._text_similarity(gap, existing_gap) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_gaps.append(gap)
        
        # Ограничиваем количество пробелов
        max_gaps = min(10, len(unique_gaps))
        semantic_gaps = unique_gaps[:max_gaps]
        
        # Формируем резюме контента
        content_summaries = [result.get("content_summary", "") for result in chunk_results if result.get("content_summary")]
        content_summary = " ".join(content_summaries)
        if len(content_summary) > 200:
            content_summary = content_summary[:197] + "..."
        
        # Собираем статистику по токенам и стоимости
        # Продолжение файла semantic_structure_extractor.py
        # Собираем статистику по токенам и стоимости
        total_tokens = sum(result.get("tokens", {}).get("total", 0) for result in chunk_results)
        total_cost = sum(result.get("cost", 0) for result in chunk_results)
        
        # Формируем итоговый результат
        combined_result = {
            "topics": combined_topics,
            "keywords": keywords,
            "semantic_clusters": combined_clusters,
            "content_summary": content_summary,
            "semantic_gaps": semantic_gaps,
            "extraction_method": "llm",
            "chunks_analyzed": len(chunk_results),
            "original_content_length": len(original_content),
            "tokens": {"total": total_tokens},
            "cost": total_cost
        }
        
        return combined_result
    
    def _combine_extraction_results(self, llm_result: Dict[str, Any], 
                                 stat_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Объединяет результаты извлечения структуры разными методами.
        
        Args:
            llm_result: Результат анализа с помощью LLM
            stat_result: Результат анализа с помощью статистических методов
            
        Returns:
            Dict[str, Any]: Объединенный результат анализа
        """
        # Объединяем темы
        llm_topics = llm_result.get("topics", [])
        stat_topics = stat_result.get("topics", [])
        
        # Ищем соответствия между темами
        matched_topics = []
        unmatched_llm_topics = []
        unmatched_stat_topics = []
        
        for llm_topic in llm_topics:
            found_match = False
            
            for stat_topic in stat_topics:
                name_similarity = self._text_similarity(llm_topic["name"], stat_topic["name"])
                keyword_similarity = self._keyword_similarity(
                    llm_topic.get("keywords", []), 
                    stat_topic.get("keywords", [])
                )
                
                # Если темы похожи, объединяем их
                if name_similarity > 0.5 or keyword_similarity > 0.3:
                    # Объединяем тему
                    matched_topic = {
                        "name": llm_topic["name"],  # Предпочитаем название от LLM
                        "keywords": list(set(llm_topic.get("keywords", []) + stat_topic.get("keywords", []))),
                        "paragraphs": list(set(llm_topic.get("paragraphs", []) + stat_topic.get("paragraphs", [])))
                    }
                    
                    matched_topics.append(matched_topic)
                    found_match = True
                    break
            
            if not found_match:
                unmatched_llm_topics.append(llm_topic)
        
        # Добавляем неcопоставленные статистические темы
        for stat_topic in stat_topics:
            found_match = False
            
            for llm_topic in llm_topics:
                name_similarity = self._text_similarity(stat_topic["name"], llm_topic["name"])
                keyword_similarity = self._keyword_similarity(
                    stat_topic.get("keywords", []), 
                    llm_topic.get("keywords", [])
                )
                
                if name_similarity > 0.5 or keyword_similarity > 0.3:
                    found_match = True
                    break
            
            if not found_match:
                unmatched_stat_topics.append(stat_topic)
        
        # Объединяем все темы
        all_topics = matched_topics + unmatched_llm_topics + unmatched_stat_topics
        
        # Ограничиваем количество тем
        max_topics = min(10, len(all_topics))
        combined_topics = all_topics[:max_topics]
        
        # Объединяем ключевые слова
        llm_keywords = llm_result.get("keywords", [])
        stat_keywords = stat_result.get("keywords", [])
        
        # Объединяем и удаляем дубликаты
        combined_keywords = list(set(llm_keywords + stat_keywords))
        
        # Ограничиваем количество ключевых слов
        max_keywords = min(50, len(combined_keywords))
        keywords = combined_keywords[:max_keywords]
        
        # Объединяем семантические кластеры
        llm_clusters = llm_result.get("semantic_clusters", [])
        stat_clusters = stat_result.get("semantic_clusters", [])
        
        # Ищем соответствия между кластерами
        matched_clusters = []
        unmatched_llm_clusters = []
        unmatched_stat_clusters = []
        
        for llm_cluster in llm_clusters:
            found_match = False
            
            for stat_cluster in stat_clusters:
                name_similarity = self._text_similarity(llm_cluster["name"], stat_cluster["name"])
                keyword_similarity = self._keyword_similarity(
                    llm_cluster.get("keywords", []), 
                    stat_cluster.get("keywords", [])
                )
                
                # Если кластеры похожи, объединяем их
                if name_similarity > 0.5 or keyword_similarity > 0.3:
                    # Объединяем кластер
                    matched_cluster = {
                        "name": llm_cluster["name"],  # Предпочитаем название от LLM
                        "keywords": list(set(llm_cluster.get("keywords", []) + stat_cluster.get("keywords", []))),
                        "related_clusters": llm_cluster.get("related_clusters", [])
                    }
                    
                    matched_clusters.append(matched_cluster)
                    found_match = True
                    break
            
            if not found_match:
                unmatched_llm_clusters.append(llm_cluster)
        
        # Добавляем неcопоставленные статистические кластеры
        for stat_cluster in stat_clusters:
            found_match = False
            
            for llm_cluster in llm_clusters:
                name_similarity = self._text_similarity(stat_cluster["name"], llm_cluster["name"])
                keyword_similarity = self._keyword_similarity(
                    stat_cluster.get("keywords", []), 
                    llm_cluster.get("keywords", [])
                )
                
                if name_similarity > 0.5 or keyword_similarity > 0.3:
                    found_match = True
                    break
            
            if not found_match:
                unmatched_stat_clusters.append(stat_cluster)
        
        # Объединяем все кластеры
        all_clusters = matched_clusters + unmatched_llm_clusters + unmatched_stat_clusters
        
        # Ограничиваем количество кластеров
        max_clusters = min(10, len(all_clusters))
        combined_clusters = all_clusters[:max_clusters]
        
        # Сохраняем семантические пробелы от LLM
        semantic_gaps = llm_result.get("semantic_gaps", [])
        
        # Предпочитаем резюме от LLM
        content_summary = llm_result.get("content_summary", "")
        
        # Собираем статистику по токенам и стоимости
        total_tokens = llm_result.get("tokens", {}).get("total", 0)
        total_cost = llm_result.get("cost", 0)
        
        # Формируем итоговый результат
        combined_result = {
            "topics": combined_topics,
            "keywords": keywords,
            "semantic_clusters": combined_clusters,
            "content_summary": content_summary,
            "semantic_gaps": semantic_gaps,
            "extraction_method": "hybrid",
            "tokens": {"total": total_tokens},
            "cost": total_cost
        }
        
        return combined_result
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Вычисляет сходство между двумя текстами.
        
        Args:
            text1: Первый текст
            text2: Второй текст
            
        Returns:
            float: Сходство между текстами (от 0 до 1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Простая мера сходства на основе пересечения слов
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        # Избегаем деления на ноль
        if not words1 or not words2:
            return 0.0
        
        # Используем коэффициент Жаккара
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union
    
    def _keyword_similarity(self, keywords1: List[str], keywords2: List[str]) -> float:
        """
        Вычисляет сходство между двумя наборами ключевых слов.
        
        Args:
            keywords1: Первый набор ключевых слов
            keywords2: Второй набор ключевых слов
            
        Returns:
            float: Сходство между наборами (от 0 до 1)
        """
        if not keywords1 or not keywords2:
            return 0.0
        
        # Преобразуем в множества для удобства
        set1 = set(k.lower() for k in keywords1)
        set2 = set(k.lower() for k in keywords2)
        
        # Избегаем деления на ноль
        if not set1 or not set2:
            return 0.0
        
        # Используем коэффициент Жаккара
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union
    
    def _merge_topics(self, topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Объединяет несколько похожих тем.
        
        Args:
            topics: Список тем для объединения
            
        Returns:
            Dict[str, Any]: Объединенная тема
        """
        if not topics:
            return {
                "name": "",
                "keywords": [],
                "paragraphs": []
            }
        
        # Если только одна тема, возвращаем ее
        if len(topics) == 1:
            return topics[0]
        
        # Объединяем темы
        all_names = [topic["name"] for topic in topics]
        all_keywords = []
        all_paragraphs = []
        
        for topic in topics:
            all_keywords.extend(topic.get("keywords", []))
            all_paragraphs.extend(topic.get("paragraphs", []))
        
        # Удаляем дубликаты
        unique_keywords = list(set(all_keywords))
        
        # Для абзацев нужно более сложное сравнение
        unique_paragraphs = []
        for paragraph in all_paragraphs:
            is_duplicate = False
            for existing_paragraph in unique_paragraphs:
                if self._text_similarity(paragraph, existing_paragraph) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_paragraphs.append(paragraph)
        
        # Выбираем название темы (предпочитаем более длинное)
        name = max(all_names, key=len) if all_names else ""
        
        return {
            "name": name,
            "keywords": unique_keywords,
            "paragraphs": unique_paragraphs
        }
    
    def _merge_clusters(self, clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Объединяет несколько похожих семантических кластеров.
        
        Args:
            clusters: Список кластеров для объединения
            
        Returns:
            Dict[str, Any]: Объединенный кластер
        """
        if not clusters:
            return {
                "name": "",
                "keywords": [],
                "related_clusters": []
            }
        
        # Если только один кластер, возвращаем его
        if len(clusters) == 1:
            return clusters[0]
        
        # Объединяем кластеры
        all_names = [cluster["name"] for cluster in clusters]
        all_keywords = []
        all_related = []
        
        for cluster in clusters:
            all_keywords.extend(cluster.get("keywords", []))
            all_related.extend(cluster.get("related_clusters", []))
        
        # Удаляем дубликаты ключевых слов
        unique_keywords = list(set(all_keywords))
        
        # Удаляем дубликаты связанных кластеров
        unique_related = []
        for related in all_related:
            related_name = related.get("name", "")
            is_duplicate = False
            
            for existing_related in unique_related:
                if self._text_similarity(related_name, existing_related.get("name", "")) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_related.append(related)
        
        # Выбираем название кластера (предпочитаем более длинное)
        name = max(all_names, key=len) if all_names else ""
        
        return {
            "name": name,
            "keywords": unique_keywords,
            "related_clusters": unique_related
        }
