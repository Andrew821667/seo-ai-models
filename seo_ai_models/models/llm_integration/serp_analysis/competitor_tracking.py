"""
Отслеживание цитируемости конкурентов в ответах LLM-поисковиков.

Модуль предоставляет функционал для отслеживания и сравнения цитируемости
контента конкурентов в ответах LLM-поисковиков, выявления стратегий
оптимизации, используемых конкурентами, и анализа трендов.
"""

import re
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from .llm_serp_analyzer import LLMSerpAnalyzer
from .citation_analyzer import CitationAnalyzer
from ..common.utils import parse_json_response


class CompetitorTracking:
    """
    Отслеживание цитируемости конкурентов в ответах LLM-поисковиков.
    """
    
    def __init__(self, llm_service: LLMService, prompt_generator: PromptGenerator):
        """
        Инициализирует трекер конкурентов.
        
        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM
            prompt_generator: Экземпляр PromptGenerator для генерации промптов
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        self.serp_analyzer = LLMSerpAnalyzer(llm_service, prompt_generator)
        self.citation_analyzer = CitationAnalyzer(llm_service, prompt_generator)
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
        
        # Инициализация хранилища данных конкурентов
        self.competitors_data = {}
    
    def track_competitors(self, query: str, competitors: List[Dict[str, Any]], 
                        our_content: Optional[str] = None,
                        llm_engines: Optional[List[str]] = None,
                        num_samples: int = 2, budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Отслеживает цитируемость контента конкурентов.
        
        Args:
            query: Поисковый запрос
            competitors: Список конкурентов с их контентом
                [{"id": "competitor_id", "name": "Competitor Name", "content": "Content"}]
            our_content: Наш контент для сравнения (опционально)
            llm_engines: Список LLM-поисковиков для анализа
            num_samples: Количество запросов для каждого поисковика
            budget: Максимальный бюджет в рублях (опционально)
            
        Returns:
            Dict[str, Any]: Результаты отслеживания конкурентов
        """
        # Определяем общее количество контента для анализа
        total_content_count = len(competitors)
        if our_content:
            total_content_count += 1
        
        # Распределяем бюджет
        content_budget = None
        if budget is not None:
            content_budget = budget / total_content_count
        
        # Анализируем контент конкурентов
        competitors_results = []
        total_tokens = 0
        total_cost = 0
        
        for i, competitor in enumerate(competitors):
            comp_id = competitor.get("id", f"competitor_{i}")
            comp_name = competitor.get("name", f"Конкурент {i+1}")
            comp_content = competitor.get("content", "")
            
            # Пропускаем, если контент отсутствует
            if not comp_content:
                self.logger.warning(f"Пропуск конкурента {comp_name}: контент отсутствует")
                continue
            
            self.logger.info(f"Анализ конкурента {i+1} из {len(competitors)}: {comp_name}")
            
            # Анализируем SERP для контента конкурента
            serp_result = self.serp_analyzer.analyze_serp(
                query, comp_content, llm_engines, num_samples, content_budget
            )
            
            # Создаем результат для конкурента
            competitor_result = {
                "id": comp_id,
                "name": comp_name,
                "content_hash": self._generate_content_hash(comp_content),
                "content_length": len(comp_content),
                "citation_rate": serp_result.get("citation_rate", 0),
                "visibility_score": serp_result.get("visibility_score", 0),
                "serp_details": {
                    "engines": serp_result.get("engines", []),
                    "engines_results": serp_result.get("engines_results", {})
                },
                "tokens": serp_result.get("tokens", {}),
                "cost": serp_result.get("cost", 0)
            }
            
            # Добавляем результат конкурента
            competitors_results.append(competitor_result)
            
            # Обновляем статистику
            total_tokens += serp_result.get("tokens", {}).get("total", 0)
            total_cost += serp_result.get("cost", 0)
            
            # Сохраняем данные конкурента
            self._save_competitor_data(comp_id, comp_name, comp_content, query, serp_result)
        
        # Если есть наш контент, анализируем его для сравнения
        our_result = None
        if our_content:
            self.logger.info("Анализ нашего контента для сравнения")
            
            # Анализируем SERP для нашего контента
            our_serp_result = self.serp_analyzer.analyze_serp(
                query, our_content, llm_engines, num_samples, content_budget
            )
            
            # Создаем результат для нашего контента
            our_result = {
                "id": "our_content",
                "name": "Наш контент",
                "content_hash": self._generate_content_hash(our_content),
                "content_length": len(our_content),
                "citation_rate": our_serp_result.get("citation_rate", 0),
                "visibility_score": our_serp_result.get("visibility_score", 0),
                "serp_details": {
                    "engines": our_serp_result.get("engines", []),
                    "engines_results": our_serp_result.get("engines_results", {})
                },
                "tokens": our_serp_result.get("tokens", {}),
                "cost": our_serp_result.get("cost", 0)
            }
            
            # Обновляем статистику
            total_tokens += our_serp_result.get("tokens", {}).get("total", 0)
            total_cost += our_serp_result.get("cost", 0)
        
        # Анализируем стратегии конкурентов
        competitor_strategies = self._analyze_competitor_strategies(competitors_results, our_result)
        
        # Формируем итоговый результат
        result = {
            "query": query,
            "competitors_count": len(competitors_results),
            "has_our_content": our_result is not None,
            "ranking": self._rank_competitors(competitors_results, our_result),
            "competitor_strategies": competitor_strategies,
            "competitors_results": competitors_results,
            "our_result": our_result,
            "engines": llm_engines,
            "samples_per_engine": num_samples,
            "tokens": {"total": total_tokens},
            "cost": total_cost,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _generate_content_hash(self, content: str) -> str:
        """
        Генерирует хэш контента.
        
        Args:
            content: Контент для хэширования
            
        Returns:
            str: Хэш контента
        """
        # Используем MD5 для получения хэша (только для идентификации контента, не для криптографии)
        md5_hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
        return md5_hash[:10]  # Используем только первые 10 символов для краткости
    
    def _save_competitor_data(self, competitor_id: str, competitor_name: str,
                           content: str, query: str, serp_result: Dict[str, Any]) -> None:
        """
        Сохраняет данные о конкуренте.
        
        Args:
            competitor_id: Идентификатор конкурента
            competitor_name: Название конкурента
            content: Контент конкурента
            query: Поисковый запрос
            serp_result: Результат анализа SERP
        """
        # Генерируем хэш контента
        content_hash = self._generate_content_hash(content)
        
        # Создаем запись для конкурента, если ее еще нет
        if competitor_id not in self.competitors_data:
            self.competitors_data[competitor_id] = {
                "id": competitor_id,
                "name": competitor_name,
                "content_hashes": {},
                "last_updated": datetime.now().isoformat()
            }
        
        # Обновляем имя конкурента
        self.competitors_data[competitor_id]["name"] = competitor_name
        
        # Создаем запись для контента, если ее еще нет
        if content_hash not in self.competitors_data[competitor_id]["content_hashes"]:
            self.competitors_data[competitor_id]["content_hashes"][content_hash] = {
                "content_length": len(content),
                "queries": {},
                "first_seen": datetime.now().isoformat()
            }
        
        # Обновляем данные для запроса
        if query not in self.competitors_data[competitor_id]["content_hashes"][content_hash]["queries"]:
            self.competitors_data[competitor_id]["content_hashes"][content_hash]["queries"][query] = []
        
        # Добавляем новую запись
        self.competitors_data[competitor_id]["content_hashes"][content_hash]["queries"][query].append({
            "citation_rate": serp_result.get("citation_rate", 0),
            "visibility_score": serp_result.get("visibility_score", 0),
            "engines": serp_result.get("engines", []),
            "timestamp": datetime.now().isoformat()
        })
        
        # Обновляем время последнего обновления
        self.competitors_data[competitor_id]["last_updated"] = datetime.now().isoformat()
    
    def _rank_competitors(self, competitors_results: List[Dict[str, Any]],
                        our_result: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Ранжирует конкурентов по цитируемости.
        
        Args:
            competitors_results: Результаты анализа конкурентов
            our_result: Результат анализа нашего контента
            
        Returns:
            List[Dict[str, Any]]: Ранжированный список конкурентов
        """
        # Создаем список всех результатов
        all_results = list(competitors_results)
        if our_result:
            all_results.append(our_result)
        
        # Сортируем по видимости (основной показатель)
        all_results.sort(key=lambda x: x.get("visibility_score", 0), reverse=True)
        
        # Присваиваем ранги
        ranked_results = []
        for i, result in enumerate(all_results):
            ranked_results.append({
                "rank": i + 1,
                "id": result.get("id", ""),
                "name": result.get("name", ""),
                "is_our_content": result.get("id") == "our_content",
                "visibility_score": result.get("visibility_score", 0),
                "citation_rate": result.get("citation_rate", 0),
                "content_length": result.get("content_length", 0)
            })
        
        return ranked_results
    
    def _analyze_competitor_strategies(self, competitors_results: List[Dict[str, Any]],
                                     our_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Анализирует стратегии конкурентов.
        
        Args:
            competitors_results: Результаты анализа конкурентов
            our_result: Результат анализа нашего контента
            
        Returns:
            Dict[str, Any]: Анализ стратегий конкурентов
        """
        # Если нет результатов конкурентов, возвращаем пустой анализ
        if not competitors_results:
            return {
                "top_competitor": None,
                "content_length_correlation": 0,
                "content_length_analysis": "Недостаточно данных для анализа",
                "citation_insights": "Недостаточно данных для анализа",
                "recommendations": []
            }
        
        # Создаем список всех результатов
        all_results = list(competitors_results)
        if our_result:
            all_results.append(our_result)
        
        # Определяем топового конкурента (с наивысшей видимостью)
        top_competitor = max(competitors_results, key=lambda x: x.get("visibility_score", 0))
        
        # Анализируем корреляцию между длиной контента и видимостью
        content_lengths = [result.get("content_length", 0) for result in all_results]
        visibility_scores = [result.get("visibility_score", 0) for result in all_results]
        
        # Упрощенный расчет корреляции
        content_length_correlation = self._calculate_correlation(content_lengths, visibility_scores)
        
        # Анализируем длину контента
        avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
        top_content_length = top_competitor.get("content_length", 0)
        
        # Формируем анализ длины контента
        if content_length_correlation > 0.6:
            content_length_analysis = f"Наблюдается сильная корреляция между длиной контента и видимостью в LLM-поисковиках. Средняя длина контента составляет {avg_content_length:.0f} символов, а у лидера - {top_content_length} символов."
        elif content_length_correlation > 0.3:
            content_length_analysis = f"Наблюдается умеренная корреляция между длиной контента и видимостью. Средняя длина контента составляет {avg_content_length:.0f} символов."
        else:
            content_length_analysis = f"Корреляция между длиной контента и видимостью слабая. Видимо, другие факторы имеют большее значение."
        
        # Анализируем инсайты по цитированию
        citation_insights = self._generate_citation_insights(all_results, our_result)
        
        # Формируем рекомендации
        recommendations = self._generate_recommendations(
            all_results, our_result, content_length_correlation, top_competitor
        )
        
        # Формируем итоговый анализ
        strategies_analysis = {
            "top_competitor": {
                "id": top_competitor.get("id", ""),
                "name": top_competitor.get("name", ""),
                "visibility_score": top_competitor.get("visibility_score", 0),
                "citation_rate": top_competitor.get("citation_rate", 0),
                "content_length": top_competitor.get("content_length", 0)
            },
            "average_metrics": {
                "visibility_score": sum(visibility_scores) / len(visibility_scores) if visibility_scores else 0,
                "citation_rate": sum(result.get("citation_rate", 0) for result in all_results) / len(all_results) if all_results else 0,
                "content_length": avg_content_length
            },
            "content_length_correlation": content_length_correlation,
            "content_length_analysis": content_length_analysis,
            "citation_insights": citation_insights,
            "recommendations": recommendations
        }
        
        return strategies_analysis
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """
        Рассчитывает корреляцию между двумя списками значений.
        
        Args:
            x_values: Список значений X
            y_values: Список значений Y
            
        Returns:
            float: Коэффициент корреляции
        """
        # Проверяем длину списков
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        # Рассчитываем средние значения
        mean_x = sum(x_values) / len(x_values)
        mean_y = sum(y_values) / len(y_values)
        
        # Рассчитываем ковариацию и дисперсии
        covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        variance_x = sum((x - mean_x) ** 2 for x in x_values)
        variance_y = sum((y - mean_y) ** 2 for y in y_values)
        
        # Проверяем деление на ноль
        if variance_x == 0 or variance_y == 0:
            return 0.0
        
        # Рассчитываем корреляцию
        correlation = covariance / ((variance_x * variance_y) ** 0.5)
        
        return correlation
    
    def _generate_citation_insights(self, all_results: List[Dict[str, Any]],
                                  our_result: Optional[Dict[str, Any]] = None) -> str:
        """
        Генерирует инсайты по цитированию.
        
        Args:
            all_results: Все результаты анализа
            our_result: Результат анализа нашего контента
            
        Returns:
            str: Инсайты по цитированию
        """
        # Если нет достаточно результатов, возвращаем базовый инсайт
        if len(all_results) < 2:
            return "Недостаточно данных для формирования инсайтов по цитированию."
        
        # Собираем данные по цитированию
        citation_rates = [result.get("citation_rate", 0) for result in all_results]
        avg_citation_rate = sum(citation_rates) / len(citation_rates)
        max_citation_rate = max(citation_rates)
        
        # Формируем базовые инсайты
        insights = f"Средняя частота цитирования составляет {avg_citation_rate:.1%}, максимальная - {max_citation_rate:.1%}. "
        
        # Если есть наш результат, сравниваем его со средними показателями
        if our_result:
            our_citation_rate = our_result.get("citation_rate", 0)
            our_visibility_score = our_result.get("visibility_score", 0)
            
            # Сравниваем наш результат со средними показателями
            if our_citation_rate > avg_citation_rate * 1.2:
                insights += "Наш контент цитируется чаще, чем у конкурентов. "
            elif our_citation_rate < avg_citation_rate * 0.8:
                insights += "Наш контент цитируется реже, чем у конкурентов. "
            else:
                insights += "Частота цитирования нашего контента примерно соответствует среднему уровню конкурентов. "
            
            # Анализируем видимость
            avg_visibility = sum(result.get("visibility_score", 0) for result in all_results) / len(all_results)
            
            if our_visibility_score > avg_visibility * 1.2:
                insights += "При этом видимость нашего контента выше средней. "
            elif our_visibility_score < avg_visibility * 0.8:
                insights += "При этом видимость нашего контента ниже средней. "
        
        # Добавляем инсайты по распределению цитирования
        max_result = max(all_results, key=lambda x: x.get("citation_rate", 0))
        min_result = min(all_results, key=lambda x: x.get("citation_rate", 0))
        
        insights += f"Наибольшая частота цитирования у {max_result.get('name', 'неизвестного конкурента')} - {max_result.get('citation_rate', 0):.1%}, "
        insights += f"наименьшая у {min_result.get('name', 'неизвестного конкурента')} - {min_result.get('citation_rate', 0):.1%}."
        
        return insights
    
    def _generate_recommendations(self, all_results: List[Dict[str, Any]],
                               our_result: Optional[Dict[str, Any]],
                               content_length_correlation: float,
                               top_competitor: Dict[str, Any]) -> List[str]:
        """
        Генерирует рекомендации на основе анализа конкурентов.
        
        Args:
            all_results: Все результаты анализа
            our_result: Результат анализа нашего контента
            content_length_correlation: Корреляция между длиной контента и видимостью
            top_competitor: Данные о топовом конкуренте
            
        Returns:
            List[str]: Список рекомендаций
        """
        recommendations = []
        
        # Если нет нашего результата, возвращаем общие рекомендации
        if not our_result:
            recommendations.append(f"Проанализируйте контент лидера - {top_competitor.get('name', 'топ-конкурента')} - для понимания стратегии оптимизации.")
            
            if content_length_correlation > 0.5:
                avg_length = sum(result.get("content_length", 0) for result in all_results) / len(all_results)
                recommendations.append(f"Оптимальная длина контента для данной тематики составляет примерно {avg_length:.0f} символов.")
            
            recommendations.append("Отслеживайте изменения в контенте конкурентов для выявления стратегий оптимизации.")
            return recommendations
        
        # Анализируем наш контент относительно конкурентов
        our_citation_rate = our_result.get("citation_rate", 0)
        our_visibility_score = our_result.get("visibility_score", 0)
        our_content_length = our_result.get("content_length", 0)
        
        # Сравниваем с топовым конкурентом
        top_citation_rate = top_competitor.get("citation_rate", 0)
        top_visibility_score = top_competitor.get("visibility_score", 0)
        top_content_length = top_competitor.get("content_length", 0)
        
        # Формируем рекомендации на основе сравнения
        if our_citation_rate < top_citation_rate * 0.8:
            recommendations.append(f"Повысьте цитируемость контента, изучив стратегию лидера ({top_competitor.get('name', 'топ-конкурента')}).")
        
        if our_visibility_score < top_visibility_score * 0.8:
            recommendations.append("Улучшите видимость контента в LLM-поисковиках, добавив больше уникальной и полезной информации.")
        
        # Рекомендации по длине контента
        if content_length_correlation > 0.5:
            if our_content_length < top_content_length * 0.8:
                recommendations.append(f"Увеличьте длину контента примерно до {top_content_length} символов, добавив релевантную информацию.")
            elif our_content_length > top_content_length * 1.2:
                recommendations.append(f"Сократите и структурируйте контент, сделав его более лаконичным (примерно до {top_content_length} символов).")
        
        # Общие рекомендации
        recommendations.append("Регулярно отслеживайте цитируемость контента конкурентов для выявления новых трендов и стратегий.")
        
        # Если рекомендаций мало, добавляем общие
        if len(recommendations) < 3:
            recommendations.append("Улучшите структуру контента, добавив подзаголовки и маркированные списки для лучшего восприятия LLM.")
            recommendations.append("Добавьте больше фактической информации, статистику и данные исследований для повышения авторитетности.")
        
        return recommendations
    
    def get_competitor_history(self, competitor_id: str, 
                             time_period: Optional[int] = None) -> Dict[str, Any]:
        """
        Получает историю данных о конкуренте.
        
        Args:
            competitor_id: Идентификатор конкурента
            time_period: Период в днях для анализа истории (опционально)
            
        Returns:
            Dict[str, Any]: История данных о конкуренте
        """
        # Проверяем наличие конкурента в данных
        if competitor_id not in self.competitors_data:
            return {
                "id": competitor_id,
                "name": "",
                "history_available": False,
                "message": f"История данных для конкурента {competitor_id} отсутствует"
            }
        
        # Получаем данные о конкуренте
        competitor_data = self.competitors_data[competitor_id]
        
        # Если указан период, фильтруем данные
        if time_period is not None:
            cutoff_date = datetime.now() - timedelta(days=time_period)
            cutoff_date_str = cutoff_date.isoformat()
            
            # Фильтруем данные по дате
            filtered_content_hashes = {}
            
            for content_hash, content_data in competitor_data.get("content_hashes", {}).items():
                # Пропускаем контент, который впервые был замечен после даты отсечения
                if content_data.get("first_seen", "") >= cutoff_date_str:
                    continue
                
                # Фильтруем запросы по дате
                filtered_queries = {}
                
                for query, records in content_data.get("queries", {}).items():
                    filtered_records = [
                        record for record in records 
                        if record.get("timestamp", "") >= cutoff_date_str
                    ]
                    
                    if filtered_records:
                        filtered_queries[query] = filtered_records
                
                # Добавляем контент только если есть отфильтрованные запросы
                if filtered_queries:
                    filtered_content_data = dict(content_data)
                    filtered_content_data["queries"] = filtered_queries
                    filtered_content_hashes[content_hash] = filtered_content_data
            
            # Обновляем данные о конкуренте
            filtered_competitor_data = dict(competitor_data)
            filtered_competitor_data["content_hashes"] = filtered_content_hashes
            filtered_competitor_data["time_period"] = time_period
            filtered_competitor_data["history_available"] = bool(filtered_content_hashes)
            
            return filtered_competitor_data
        
        # Возвращаем полные данные о конкуренте
        return {
            "id": competitor_data.get("id", competitor_id),
            "name": competitor_data.get("name", ""),
            "content_hashes": competitor_data.get("content_hashes", {}),
            "last_updated": competitor_data.get("last_updated", ""),
            "history_available": bool(competitor_data.get("content_hashes", {}))
        }
