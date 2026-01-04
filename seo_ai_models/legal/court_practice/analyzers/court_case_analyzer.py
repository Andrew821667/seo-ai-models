"""
Анализатор судебных дел для системы анализа судебной практики.
Включает семантический анализ, оценку рисков и предиктивный анализ.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter

from seo_ai_models.legal.court_practice.models.court_case import (
    CourtCase, CaseAnalysis, CaseCategory, CaseStatus
)
from seo_ai_models.common.utils.enhanced_text_processor import EnhancedTextProcessor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CourtCaseAnalyzer:
    """
    Анализатор судебных дел с использованием NLP и статистических методов.
    """

    def __init__(self):
        """Инициализация анализатора."""
        self.text_processor = EnhancedTextProcessor()

        # Ключевые слова для анализа рисков
        self.risk_keywords = {
            'high': [
                'мошенничество', 'fraud', 'хищение', 'theft', 'коррупция', 'corruption',
                'банкротство', 'bankruptcy', 'криминальный', 'criminal', 'арест', 'arrest',
                'конфискация', 'confiscation', 'санкции', 'sanctions'
            ],
            'medium': [
                'неисполнение', 'non-performance', 'нарушение', 'violation',
                'задолженность', 'debt', 'спор', 'dispute', 'претензия', 'claim',
                'иск', 'lawsuit', 'штраф', 'fine', 'пеня', 'penalty'
            ],
            'low': [
                'консультация', 'consultation', 'договор', 'contract',
                'соглашение', 'agreement', 'услуги', 'services'
            ]
        }

        # Статистические данные по категориям дел (примерные)
        self.category_stats = {
            CaseCategory.CONTRACT: {
                'success_rate': 0.65,  # доля удовлетворенных исков
                'avg_duration': 180,   # средняя продолжительность в днях
                'avg_amount': 1500000  # средняя сумма иска
            },
            CaseCategory.PROPERTY: {
                'success_rate': 0.58,
                'avg_duration': 240,
                'avg_amount': 2500000
            },
            CaseCategory.CORPORATE: {
                'success_rate': 0.45,
                'avg_duration': 300,
                'avg_amount': 5000000
            },
            CaseCategory.TAX: {
                'success_rate': 0.35,
                'avg_duration': 200,
                'avg_amount': 800000
            },
            CaseCategory.LABOR: {
                'success_rate': 0.70,
                'avg_duration': 90,
                'avg_amount': 150000
            },
            CaseCategory.ADMINISTRATIVE: {
                'success_rate': 0.40,
                'avg_duration': 120,
                'avg_amount': 50000
            },
            CaseCategory.OTHER: {
                'success_rate': 0.50,
                'avg_duration': 150,
                'avg_amount': 500000
            }
        }

        logger.info("CourtCaseAnalyzer initialized")

    def analyze_case(self, case: CourtCase) -> CaseAnalysis:
        """
        Комплексный анализ судебного дела.

        Args:
            case: Судебное дело для анализа

        Returns:
            CaseAnalysis: Результаты анализа
        """
        try:
            # Семантический анализ
            semantic_analysis = self._perform_semantic_analysis(case)

            # Оценка рисков
            risk_assessment = self._assess_risks(case)

            # Поиск похожих дел
            similar_cases_analysis = self._find_similar_cases(case)

            # Предиктивный анализ исхода
            predictive_outcome = self._predict_outcome(case)

            # Рекомендации
            recommendations = self._generate_recommendations(case, risk_assessment, predictive_outcome)

            return CaseAnalysis(
                case=case,
                semantic_analysis=semantic_analysis,
                risk_assessment=risk_assessment,
                similar_cases_analysis=similar_cases_analysis,
                predictive_outcome=predictive_outcome,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error analyzing case {case.case_number}: {str(e)}")
            # Возвращаем базовый анализ в случае ошибки
            return CaseAnalysis(
                case=case,
                recommendations=["Ошибка анализа - требуется ручная проверка"]
            )

    def _perform_semantic_analysis(self, case: CourtCase) -> Dict[str, Any]:
        """
        Семантический анализ текста дела.
        """
        try:
            # Объединяем весь текстовый контент
            text_content = f"{case.description} {case.claim_subject}"
            for decision in case.decisions:
                if decision.full_text:
                    text_content += f" {decision.full_text}"

            if not text_content.strip():
                return {
                    'semantic_density': 0.0,
                    'readability_score': 0.0,
                    'keywords': [],
                    'topics': [],
                    'sentiment': 'neutral'
                }

            # Анализ с помощью текстового процессора
            analysis_result = self.text_processor.analyze_text(text_content)

            # Дополнительный анализ
            semantic_density = self._calculate_semantic_density(text_content)
            topics = self._extract_topics(text_content)
            sentiment = self._analyze_sentiment(text_content)

            return {
                'semantic_density': semantic_density,
                'readability_score': analysis_result.get('readability', {}).get('flesch_score', 0.0),
                'keywords': analysis_result.get('keywords', [])[:10],  # топ-10 ключевых слов
                'topics': topics,
                'sentiment': sentiment,
                'word_count': len(text_content.split()),
                'language': analysis_result.get('language', 'ru')
            }

        except Exception as e:
            logger.warning(f"Error in semantic analysis: {str(e)}")
            return {'error': str(e)}

    def _calculate_semantic_density(self, text: str) -> float:
        """
        Расчет семантической плотности текста.
        """
        try:
            words = re.findall(r'\b\w+\b', text.lower())
            if not words:
                return 0.0

            # Уникальные слова / общее количество слов
            unique_words = len(set(words))
            total_words = len(words)

            # Нормализуем по логарифму для более реалистичных значений
            import math
            density = unique_words / total_words
            normalized_density = min(density * math.log(total_words + 1), 1.0)

            return round(normalized_density, 3)

        except Exception:
            return 0.0

    def _extract_topics(self, text: str) -> List[str]:
        """
        Извлечение тем из текста.
        """
        try:
            # Простой анализ на основе ключевых слов
            topics = []

            text_lower = text.lower()

            # Определяем темы на основе ключевых слов
            topic_keywords = {
                'договор': ['договор', 'контракт', 'соглашение', 'поставка'],
                'недвижимость': ['недвижимость', 'земля', 'квартира', 'дом'],
                'финансы': ['деньги', 'сумма', 'платеж', 'расчет'],
                'корпорации': ['акции', 'доля', 'общество', 'компания'],
                'налоги': ['налог', 'фискальный', 'взнос', 'отчетность'],
                'труд': ['работник', 'работодатель', 'увольнение', 'зарплата'],
                'административное': ['штраф', 'правонарушение', 'администрация']
            }

            for topic, keywords in topic_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    topics.append(topic)

            return topics[:5]  # максимум 5 тем

        except Exception:
            return []

    def _analyze_sentiment(self, text: str) -> str:
        """
        Анализ тональности текста.
        """
        try:
            # Простой анализ на основе ключевых слов
            positive_words = ['удовлетворить', 'взыскать', 'признать', 'подтвердить']
            negative_words = ['отказать', 'отклонить', 'прекратить', 'нарушение']

            text_lower = text.lower()

            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            if positive_count > negative_count:
                return 'positive'
            elif negative_count > positive_count:
                return 'negative'
            else:
                return 'neutral'

        except Exception:
            return 'neutral'

    def _assess_risks(self, case: CourtCase) -> Dict[str, Any]:
        """
        Оценка рисков судебного дела.
        """
        try:
            risk_score = 0
            risk_factors = []

            # Анализ на основе категории дела
            category_stats = self.category_stats.get(case.category, {})
            base_risk = 1 - category_stats.get('success_rate', 0.5)  # риск = 1 - вероятность успеха

            # Анализ на основе ключевых слов
            text_content = f"{case.description} {case.claim_subject}".lower()

            for level, keywords in self.risk_keywords.items():
                matches = [kw for kw in keywords if kw in text_content]
                if matches:
                    if level == 'high':
                        risk_score += len(matches) * 0.3
                        risk_factors.extend([f"Высокий риск: {kw}" for kw in matches])
                    elif level == 'medium':
                        risk_score += len(matches) * 0.15
                        risk_factors.extend([f"Средний риск: {kw}" for kw in matches])
                    elif level == 'low':
                        risk_score += len(matches) * 0.05
                        risk_factors.extend([f"Низкий риск: {kw}" for kw in matches])

            # Анализ суммы иска
            if case.claim_amount:
                if case.claim_amount > 10000000:  # > 10 млн руб
                    risk_score += 0.2
                    risk_factors.append("Высокая сумма иска (>10 млн руб)")
                elif case.claim_amount > 1000000:  # > 1 млн руб
                    risk_score += 0.1
                    risk_factors.append("Значительная сумма иска (>1 млн руб)")

            # Анализ статуса дела
            if case.status == CaseStatus.APPEALED:
                risk_score += 0.1
                risk_factors.append("Дело обжаловано")

            # Финальная оценка риска
            total_risk = min(base_risk + risk_score, 1.0)

            risk_level = 'low'
            if total_risk > 0.7:
                risk_level = 'high'
            elif total_risk > 0.4:
                risk_level = 'medium'

            return {
                'risk_level': risk_level,
                'risk_score': round(total_risk, 3),
                'risk_factors': risk_factors[:10],  # максимум 10 факторов
                'confidence': 0.8  # уверенность оценки
            }

        except Exception as e:
            logger.warning(f"Error in risk assessment: {str(e)}")
            return {
                'risk_level': 'medium',
                'risk_score': 0.5,
                'risk_factors': ['Ошибка оценки рисков'],
                'confidence': 0.0
            }

    def _find_similar_cases(self, case: CourtCase) -> List[Dict[str, Any]]:
        """
        Поиск похожих судебных дел.
        """
        # В реальной реализации здесь был бы поиск в базе данных
        # Пока возвращаем примерные данные
        try:
            similar_cases = [
                {
                    'case_number': f"A{case.case_number[1:]}1",
                    'similarity_score': 0.85,
                    'outcome': 'удовлетворен',
                    'amount': case.claim_amount * 0.9 if case.claim_amount else None,
                    'duration_days': 120
                },
                {
                    'case_number': f"A{case.case_number[1:]}2",
                    'similarity_score': 0.78,
                    'outcome': 'отклонен',
                    'amount': None,
                    'duration_days': 90
                }
            ]

            return similar_cases

        except Exception as e:
            logger.warning(f"Error finding similar cases: {str(e)}")
            return []

    def _predict_outcome(self, case: CourtCase) -> Dict[str, Any]:
        """
        Предиктивный анализ исхода дела.
        """
        try:
            # Базовая вероятность успеха из статистики категории
            category_stats = self.category_stats.get(case.category, {})
            base_success_rate = category_stats.get('success_rate', 0.5)

            # Корректировка на основе факторов
            adjustment = 0.0

            # Анализ участников (упрощенная модель)
            if len(case.plaintiffs) > 1:
                adjustment += 0.05  # несколько истцов
            if len(case.defendants) > 1:
                adjustment -= 0.05  # несколько ответчиков

            # Анализ суммы иска
            if case.claim_amount:
                if case.claim_amount < 100000:
                    adjustment += 0.1  # небольшие суммы чаще удовлетворяют
                elif case.claim_amount > 5000000:
                    adjustment -= 0.1  # крупные суммы реже

            # Финальная вероятность
            predicted_success_rate = max(0.0, min(1.0, base_success_rate + adjustment))

            # Предсказанная продолжительность
            avg_duration = category_stats.get('avg_duration', 150)

            # Вероятные исходы
            outcomes = {
                'удовлетворен': predicted_success_rate,
                'отклонен': 1 - predicted_success_rate,
                'частично_удовлетворен': 0.1 if predicted_success_rate > 0.5 else 0.05
            }

            return {
                'predicted_success_rate': round(predicted_success_rate, 3),
                'predicted_duration_days': avg_duration,
                'possible_outcomes': outcomes,
                'confidence': 0.75,
                'factors_considered': [
                    f"Категория дела: {case.category.value}",
                    f"Количество участников: {len(case.plaintiffs)} истцов, {len(case.defendants)} ответчиков",
                    f"Сумма иска: {case.claim_amount or 'не указана'}"
                ]
            }

        except Exception as e:
            logger.warning(f"Error in outcome prediction: {str(e)}")
            return {
                'predicted_success_rate': 0.5,
                'predicted_duration_days': 150,
                'possible_outcomes': {'неизвестен': 1.0},
                'confidence': 0.0,
                'factors_considered': []
            }

    def _generate_recommendations(
        self,
        case: CourtCase,
        risk_assessment: Dict[str, Any],
        predictive_outcome: Dict[str, Any]
    ) -> List[str]:
        """
        Генерация рекомендаций на основе анализа.
        """
        try:
            recommendations = []

            risk_level = risk_assessment.get('risk_level', 'medium')
            success_rate = predictive_outcome.get('predicted_success_rate', 0.5)

            # Рекомендации по рискам
            if risk_level == 'high':
                recommendations.append("Высокий уровень риска - рассмотреть возможность досудебного урегулирования")
                recommendations.append("Рекомендуется провести дополнительную юридическую экспертизу")
            elif risk_level == 'medium':
                recommendations.append("Средний уровень риска - подготовить полную доказательственную базу")
            else:
                recommendations.append("Низкий уровень риска - стандартная подготовка к делу")

            # Рекомендации по прогнозу
            if success_rate > 0.7:
                recommendations.append("Высокие шансы на успех - активно вести дело")
            elif success_rate < 0.3:
                recommendations.append("Низкие шансы на успех - рассмотреть возможность отказа от иска")
            else:
                recommendations.append("Средние шансы на успех - взвесить соотношение затрат и выгод")

            # Рекомендации по категории
            if case.category == CaseCategory.CONTRACT:
                recommendations.append("Для договорных споров критически важно наличие письменных доказательств")
            elif case.category == CaseCategory.CORPORATE:
                recommendations.append("Корпоративные споры требуют тщательного анализа корпоративных документов")

            # Рекомендации по сумме
            if case.claim_amount and case.claim_amount > 1000000:
                recommendations.append("Для исков большой суммы рекомендуется привлечение адвоката")

            return recommendations[:10]  # максимум 10 рекомендаций

        except Exception as e:
            logger.warning(f"Error generating recommendations: {str(e)}")
            return ["Требуется дополнительный анализ дела"]
