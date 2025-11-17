"""
Режим работы без постоянного доступа к API.

Модуль предоставляет функционал для анализа контента в офлайн-режиме,
используя локальные модели и кэшированные данные, что позволяет
работать без постоянного подключения к облачным API.
"""

import os
import json
import logging
import re
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from ..service.cost_estimator import CostEstimator
from .local_llm_manager import LocalLLMManager
from .intelligent_cache import IntelligentCache
from .hybrid_processing_pipeline import HybridProcessingPipeline


class OfflineAnalysisMode:
    """
    Режим работы без постоянного доступа к API.
    """

    def __init__(
        self,
        local_llm_manager: LocalLLMManager,
        cache: IntelligentCache,
        cost_estimator: Optional[CostEstimator] = None,
    ):
        """
        Инициализирует режим офлайн-анализа.

        Args:
            local_llm_manager: Экземпляр LocalLLMManager для работы с локальными LLM
            cache: Экземпляр IntelligentCache для кэширования
            cost_estimator: Экземпляр CostEstimator для оценки затрат (опционально)
        """
        self.local_llm_manager = local_llm_manager
        self.cache = cache
        self.cost_estimator = cost_estimator or CostEstimator()

        # Настройка логгирования
        self.logger = logging.getLogger(__name__)

        # Флаг офлайн-режима
        self.offline_mode = False

        # Настройки режима
        self.default_quality = "medium"  # Качество по умолчанию (low, medium, high)
        self.fallback_to_cached = True  # Использовать кэш при отсутствии локальных моделей
        self.cache_all_results = True  # Кэшировать все результаты
        self.prefetch_enabled = False  # Предварительная загрузка результатов

    def enable_offline_mode(self) -> None:
        """
        Включает офлайн-режим.
        """
        self.offline_mode = True
        self.logger.info("Офлайн-режим включен")

    def disable_offline_mode(self) -> None:
        """
        Выключает офлайн-режим.
        """
        self.offline_mode = False
        self.logger.info("Офлайн-режим выключен")

    def is_offline_mode_enabled(self) -> bool:
        """
        Проверяет, включен ли офлайн-режим.

        Returns:
            bool: True если офлайн-режим включен, иначе False
        """
        return self.offline_mode

    def configure(
        self,
        default_quality: Optional[str] = None,
        fallback_to_cached: Optional[bool] = None,
        cache_all_results: Optional[bool] = None,
        prefetch_enabled: Optional[bool] = None,
    ) -> None:
        """
        Настраивает параметры офлайн-режима.

        Args:
            default_quality: Качество по умолчанию ('low', 'medium', 'high')
            fallback_to_cached: Использовать кэш при отсутствии локальных моделей
            cache_all_results: Кэшировать все результаты
            prefetch_enabled: Предварительная загрузка результатов
        """
        if default_quality is not None:
            if default_quality not in ["low", "medium", "high"]:
                self.logger.warning(
                    f"Качество {default_quality} не поддерживается, используем 'medium'"
                )
                default_quality = "medium"

            self.default_quality = default_quality

        if fallback_to_cached is not None:
            self.fallback_to_cached = fallback_to_cached

        if cache_all_results is not None:
            self.cache_all_results = cache_all_results

        if prefetch_enabled is not None:
            self.prefetch_enabled = prefetch_enabled

    def query(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        local_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Выполняет запрос к модели в офлайн-режиме.

        Args:
            prompt: Промпт для модели
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации (от 0 до 1)
            local_model: Локальная модель для использования (опционально)

        Returns:
            Dict[str, Any]: Результат запроса
        """
        # Проверяем, включен ли офлайн-режим
        if not self.offline_mode:
            self.logger.warning(
                "Офлайн-режим не включен, но запрос выполняется через OfflineAnalysisMode"
            )

        # Пытаемся получить результат из кэша
        cached_result = self.cache.get_query_from_cache(
            prompt=prompt,
            model=local_model or "offline_model",
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if cached_result:
            self.logger.info("Найден результат в кэше")
            return cached_result

        # Если кэш не найден, используем локальную модель
        try:
            # Выбираем локальную модель, если не указана
            selected_model = local_model
            if selected_model is None:
                selected_model = self.local_llm_manager.select_optimal_model(
                    prompt=prompt, required_quality=self.default_quality
                )

            # Запрос к локальной модели
            result = self.local_llm_manager.query_model(
                prompt=prompt,
                model_name=selected_model,
                max_tokens=max_tokens,
                temperature=temperature,
                use_cache=True,
            )

            # Проверяем наличие ошибки
            if "error" in result:
                raise Exception(result["error"])

            # Если включено кэширование всех результатов, сохраняем
            if self.cache_all_results:
                self.cache.save_query_to_cache(
                    prompt=prompt,
                    model=selected_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    result=result,
                )

            return result

        except Exception as e:
            self.logger.error(f"Ошибка при запросе к локальной модели: {str(e)}")

            # Если включен fallback к кэшу и есть похожие запросы в кэше, используем их
            if self.fallback_to_cached:
                self.logger.info("Пытаемся найти похожий запрос в кэше")

                similar_result = self._find_similar_query_in_cache(
                    prompt=prompt, max_tokens=max_tokens, temperature=temperature
                )

                if similar_result:
                    self.logger.info("Найден похожий запрос в кэше")
                    similar_result["fallback_used"] = True
                    similar_result["fallback_reason"] = "Ошибка при запросе к локальной модели"
                    similar_result["original_error"] = str(e)
                    return similar_result

            # Если ничего не помогло, возвращаем ошибку
            return {
                "error": f"Ошибка при запросе в офлайн-режиме: {str(e)}",
                "text": "Ошибка при выполнении запроса в офлайн-режиме. Попробуйте другую модель или включите онлайн-режим.",
                "provider": "offline",
                "model": local_model,
                "tokens": {"prompt": 0, "completion": 0, "total": 0},
                "cost": 0,
            }

    def _find_similar_query_in_cache(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> Optional[Dict[str, Any]]:
        """
        Находит похожий запрос в кэше.

        Сейчас это заглушка, в реальной реализации здесь был бы алгоритм
        поиска похожих запросов в кэше на основе векторных представлений.

        Args:
            prompt: Промпт для модели
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации

        Returns:
            Optional[Dict[str, Any]]: Результат похожего запроса или None
        """
        # Заглушка, в реальной реализации здесь был бы более сложный алгоритм
        return None

    def analyze_content(
        self, content: str, analysis_type: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Анализирует контент в офлайн-режиме.

        Args:
            content: Контент для анализа
            analysis_type: Тип анализа
            params: Дополнительные параметры анализа (опционально)

        Returns:
            Dict[str, Any]: Результат анализа
        """
        # Проверяем, включен ли офлайн-режим
        if not self.offline_mode:
            self.logger.warning(
                "Офлайн-режим не включен, но анализ выполняется через OfflineAnalysisMode"
            )

        # Пытаемся получить результат из кэша
        cached_result = self.cache.get_analysis_from_cache(
            content=content, analysis_type=analysis_type, params=params
        )

        if cached_result:
            self.logger.info(f"Найден результат анализа {analysis_type} в кэше")
            return cached_result

        # Если кэш не найден, выполняем анализ локально
        try:
            # Проверяем тип анализа и выполняем соответствующий анализ
            if analysis_type == "citability":
                # Анализ цитируемости
                result = self._analyze_citability(content, params)
            elif analysis_type == "content_structure":
                # Анализ структуры контента
                result = self._analyze_content_structure(content, params)
            elif analysis_type == "keyword_analysis":
                # Анализ ключевых слов
                result = self._analyze_keywords(content, params)
            elif analysis_type == "eeat":
                # Анализ E-E-A-T
                result = self._analyze_eeat(content, params)
            else:
                # Если тип анализа не поддерживается, возвращаем ошибку
                return {
                    "error": f"Тип анализа {analysis_type} не поддерживается в офлайн-режиме",
                    "provider": "offline",
                    "cost": 0,
                }

            # Если включено кэширование всех результатов, сохраняем
            if self.cache_all_results:
                self.cache.save_analysis_to_cache(
                    content=content, analysis_type=analysis_type, result=result, params=params
                )

            return result

        except Exception as e:
            self.logger.error(f"Ошибка при анализе {analysis_type} в офлайн-режиме: {str(e)}")

            # Если включен fallback к кэшу, попробуем найти похожий анализ
            if self.fallback_to_cached:
                self.logger.info("Пытаемся найти похожий анализ в кэше")

                similar_result = self._find_similar_analysis_in_cache(
                    content=content, analysis_type=analysis_type, params=params
                )

                if similar_result:
                    self.logger.info(f"Найден похожий анализ {analysis_type} в кэше")
                    similar_result["fallback_used"] = True
                    similar_result["fallback_reason"] = (
                        f"Ошибка при анализе {analysis_type} в офлайн-режиме"
                    )
                    similar_result["original_error"] = str(e)
                    return similar_result

            # Если ничего не помогло, возвращаем ошибку
            return {
                "error": f"Ошибка при анализе {analysis_type} в офлайн-режиме: {str(e)}",
                "provider": "offline",
                "cost": 0,
            }

    def _find_similar_analysis_in_cache(
        self, content: str, analysis_type: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Находит похожий анализ в кэше.

        Сейчас это заглушка, в реальной реализации здесь был бы алгоритм
        поиска похожих анализов в кэше на основе сходства контента.

        Args:
            content: Контент для анализа
            analysis_type: Тип анализа
            params: Дополнительные параметры анализа (опционально)

        Returns:
            Optional[Dict[str, Any]]: Результат похожего анализа или None
        """
        # Заглушка, в реальной реализации здесь был бы более сложный алгоритм
        return None

    def _analyze_citability(
        self, content: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Анализирует цитируемость контента в офлайн-режиме.

        Args:
            content: Контент для анализа
            params: Дополнительные параметры анализа (опционально)

        Returns:
            Dict[str, Any]: Результат анализа цитируемости
        """
        # Получаем параметры
        category = params.get("category", "general") if params else "general"
        queries = params.get("queries", []) if params else []

        # Формируем промпт для анализа цитируемости
        prompt = f"""
        Проанализируй следующий текст и оцени вероятность его цитирования языковыми моделями (LLM) при ответе на пользовательские запросы.
        
        Оцени следующие факторы цитируемости по шкале от 1 до 10:
        - Информативность
        - Уникальность
        - Достоверность
        - Структурированность
        - Ясность изложения
        
        Также дай общую оценку цитируемости по шкале от 1 до 10.
        
        {f"Оцени цитируемость для следующих запросов: {', '.join(queries)}" if queries else ""}
        
        Для каждого фактора предложи конкретные улучшения, которые повысят цитируемость контента.
        
        Текст для анализа:
        {content}
        """

        # Запрос к локальной модели
        model_result = self.local_llm_manager.query_model(
            prompt=prompt, max_tokens=1000, temperature=0.7, use_cache=True
        )

        # Извлекаем оценки факторов из ответа модели
        response_text = model_result.get("text", "")

        # Извлекаем общую оценку цитируемости
        overall_match = re.search(
            r"общ[а-я]+\s+оценка.*?(\d+)[^а-я0-9]*", response_text, re.IGNORECASE
        )
        overall_score = int(overall_match.group(1)) if overall_match else 5

        # Извлекаем оценки факторов
        factor_scores = {}
        factors = [
            "Информативность",
            "Уникальность",
            "Достоверность",
            "Структурированность",
            "Ясность изложения",
        ]

        for factor in factors:
            factor_match = re.search(f"{factor}.*?(\d+)[^а-я0-9]*", response_text, re.IGNORECASE)
            factor_scores[factor] = int(factor_match.group(1)) if factor_match else 5

        # Формируем результат
        result = {
            "citability_score": overall_score,
            "factor_scores": factor_scores,
            "citability_analysis": "Анализ выполнен в офлайн-режиме",
            "factor_analysis": {
                factor: f"Оценка: {score}/10" for factor, score in factor_scores.items()
            },
            "suggested_improvements": {},
            "provider": "offline",
            "model": model_result.get("model", "offline_model"),
            "tokens": model_result.get("tokens", {"total": 0}),
            "cost": model_result.get("cost", 0),
        }

        return result

    def _analyze_content_structure(
        self, content: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Анализирует структуру контента в офлайн-режиме.

        Args:
            content: Контент для анализа
            params: Дополнительные параметры анализа (опционально)

        Returns:
            Dict[str, Any]: Результат анализа структуры контента
        """
        # Простой анализ структуры контента
        lines = content.split("\n")
        paragraphs = [p for p in content.split("\n\n") if p.strip()]

        # Подсчет заголовков
        headers = {
            "h1": len([l for l in lines if l.strip().startswith("# ")]),
            "h2": len([l for l in lines if l.strip().startswith("## ")]),
            "h3": len([l for l in lines if l.strip().startswith("### ")]),
            "total": len([l for l in lines if re.match(r"^#{1,6}\s", l.strip())]),
        }

        # Подсчет списков
        bullet_lists = len([l for l in lines if re.match(r"^\s*[-*+]\s", l.strip())])
        numbered_lists = len([l for l in lines if re.match(r"^\s*\d+\.\s", l.strip())])

        # Подсчет предложений (приблизительно)
        sentences = len(re.split(r"[.!?]+", content))

        # Формируем результат
        result = {
            "content_length": len(content),
            "paragraphs_count": len(paragraphs),
            "sentences_count": sentences,
            "headers": headers,
            "lists": {
                "bullet_lists": bullet_lists,
                "numbered_lists": numbered_lists,
                "total": bullet_lists + numbered_lists,
            },
            "avg_paragraph_length": len(content) / len(paragraphs) if paragraphs else 0,
            "avg_sentence_length": len(content) / sentences if sentences > 0 else 0,
            "provider": "offline",
            "cost": 0,
        }

        return result

    def _analyze_keywords(
        self, content: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Анализирует ключевые слова в контенте в офлайн-режиме.

        Args:
            content: Контент для анализа
            params: Дополнительные параметры анализа (опционально)

        Returns:
            Dict[str, Any]: Результат анализа ключевых слов
        """
        # Получаем параметры
        target_keywords = params.get("target_keywords", []) if params else []

        # Простой анализ ключевых слов
        # В реальной реализации здесь был бы более сложный алгоритм

        # Преобразуем контент в нижний регистр
        content_lower = content.lower()

        # Подсчитываем вхождения ключевых слов
        keyword_counts = {}

        for keyword in target_keywords:
            keyword_lower = keyword.lower()
            keyword_counts[keyword] = content_lower.count(keyword_lower)

        # Общее количество слов
        words = re.findall(r"\b\w+\b", content_lower)
        total_words = len(words)

        # Вычисляем плотность ключевых слов
        keyword_density = {}

        for keyword, count in keyword_counts.items():
            keyword_density[keyword] = count / total_words if total_words > 0 else 0

        # Формируем результат
        result = {
            "target_keywords": target_keywords,
            "keyword_counts": keyword_counts,
            "keyword_density": keyword_density,
            "total_words": total_words,
            "provider": "offline",
            "cost": 0,
        }

        return result

    def _analyze_eeat(
        self, content: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Анализирует E-E-A-T контента в офлайн-режиме.

        Args:
            content: Контент для анализа
            params: Дополнительные параметры анализа (опционально)

        Returns:
            Dict[str, Any]: Результат анализа E-E-A-T
        """
        # Формируем промпт для анализа E-E-A-T
        prompt = f"""
        Проанализируй следующий текст на соответствие принципам E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness).
        
        Оцени каждый компонент E-E-A-T по шкале от 1 до 10:
        - Experience (Опыт): Насколько в тексте демонстрируется личный опыт автора?
        - Expertise (Экспертиза): Насколько текст демонстрирует экспертные знания?
        - Authoritativeness (Авторитетность): Насколько текст и его автор представляются авторитетными?
        - Trustworthiness (Надежность): Насколько информация в тексте заслуживает доверия?
        
        Также дай общую оценку E-E-A-T по шкале от 1 до 10.
        
        Для каждого компонента предложи конкретные улучшения.
        
        Текст для анализа:
        {content}
        """

        # Запрос к локальной модели
        model_result = self.local_llm_manager.query_model(
            prompt=prompt, max_tokens=1000, temperature=0.7, use_cache=True
        )

        # Извлекаем оценки компонентов из ответа модели
        response_text = model_result.get("text", "")

        # Извлекаем общую оценку E-E-A-T
        overall_match = re.search(
            r"общ[а-я]+\s+оценка.*?(\d+)[^а-я0-9]*", response_text, re.IGNORECASE
        )
        overall_score = int(overall_match.group(1)) if overall_match else 5

        # Извлекаем оценки компонентов
        component_scores = {}
        components = ["Experience", "Expertise", "Authoritativeness", "Trustworthiness"]

        for component in components:
            component_match = re.search(
                f"{component}.*?(\d+)[^а-я0-9]*", response_text, re.IGNORECASE
            )
            component_scores[component] = int(component_match.group(1)) if component_match else 5

        # Формируем результат
        result = {
            "eeat_score": overall_score,
            "component_scores": component_scores,
            "eeat_analysis": "Анализ E-E-A-T выполнен в офлайн-режиме",
            "component_analysis": {
                component: f"Оценка: {score}/10" for component, score in component_scores.items()
            },
            "provider": "offline",
            "model": model_result.get("model", "offline_model"),
            "tokens": model_result.get("tokens", {"total": 0}),
            "cost": model_result.get("cost", 0),
        }

        return result

    def prefetch_analysis(self, content: str, analysis_types: List[str]) -> Dict[str, Any]:
        """
        Предварительно загружает результаты анализа в кэш.

        Args:
            content: Контент для анализа
            analysis_types: Типы анализа для предварительной загрузки

        Returns:
            Dict[str, Any]: Результаты предварительной загрузки
        """
        # Проверяем, включена ли предварительная загрузка
        if not self.prefetch_enabled:
            self.logger.warning("Предварительная загрузка отключена")
            return {"prefetch_enabled": False, "results": {}}

        # Результаты предварительной загрузки
        prefetch_results = {}

        # Выполняем анализ для каждого типа
        for analysis_type in analysis_types:
            self.logger.info(f"Предварительная загрузка анализа {analysis_type}")

            try:
                # Выполняем анализ
                result = self.analyze_content(content, analysis_type)

                # Сохраняем результат
                prefetch_results[analysis_type] = {"success": True, "cost": result.get("cost", 0)}
            except Exception as e:
                self.logger.error(
                    f"Ошибка при предварительной загрузке анализа {analysis_type}: {str(e)}"
                )

                prefetch_results[analysis_type] = {"success": False, "error": str(e)}

        # Формируем итоговый результат
        result = {
            "prefetch_enabled": True,
            "content_hash": self.cache._hash_content(content),
            "results": prefetch_results,
            "total_cost": sum(
                r.get("cost", 0) for r in prefetch_results.values() if r.get("success", False)
            ),
        }

        return result
