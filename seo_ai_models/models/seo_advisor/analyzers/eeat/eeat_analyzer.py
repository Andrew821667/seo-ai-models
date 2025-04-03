"""Анализатор E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness).

Анализирует контент на предмет соответствия критериям E-E-A-T для оценки качества контента
с точки зрения Google.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import re
from collections import Counter

from seo_ai_models.common.utils.text_processing import TextProcessor

class EEATAnalyzer:
    """Анализатор E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)."""
    
    def __init__(self, language: str = 'ru'):
        """Инициализация анализатора."""
        # Инициализируем TextProcessor
        self.text_processor = TextProcessor()
        self.language = language
        
        # Маркеры опыта и экспертизы
        self.expertise_markers = [
            'опыт', 'эксперт', 'специалист', 'профессионал', 'квалификация',
            'сертифицированный', 'компетентный', 'практик', 'исследователь', 'аналитик'
        ]
        
        # Маркеры авторитетности
        self.authority_markers = [
            'исследование', 'статистика', 'данные', 'доказано', 'согласно', 
            'по мнению экспертов', 'научно', 'источник', 'цитата', 'ссылка'
        ]
        
        # Маркеры доверия
        self.trust_markers = [
            'достоверный', 'проверенный', 'надежный', 'точный', 'подтвержденный',
            'официальный', 'гарантированный', 'безопасный', 'проверка фактов', 'прозрачность',
            'методология', 'метод', 'данные показывают', 'доказательство', 'подтверждено',
            'публикация', 'обновлено', 'раскрытие информации', 'отказ от ответственности', 'дисклеймер'
        ]
        
        # Маркеры опыта
        self.experience_markers = [
            'личный опыт', 'я использовал', 'я попробовал', 'я тестировал', 'мы проверили',
            'основываясь на опыте', 'из личного опыта', 'собственный опыт', 'я применял'
        ]
        
        # Категории YMYL (Your Money Your Life)
        self.ymyl_categories = {
            'finance', 'health', 'legal', 'medical', 'insurance', 
            'investments', 'taxes', 'real_estate', 'medicine', 'law'
        }
    
    def analyze(self, content: str, industry: str = 'default', language: str = None, 
               html_content: Optional[str] = None) -> Dict[str, Union[float, Dict, List]]:
        """
        Оценка E-E-A-T сигналов в тексте
        
        Args:
            content: Текстовое содержимое
            industry: Отрасль контента
            language: Язык контента (если None, будет определен автоматически)
            html_content: HTML-версия контента (если доступна)
            
        Returns:
            Словарь с оценками и рекомендациями
        """
        # Определяем язык контента
        if language is None:
            language = self.text_processor.detect_language(content)
        
        # Нормализуем текст для анализа
        text_lower = content.lower()
        
        # Оценка опыта (Experience)
        experience_score = self._evaluate_experience(text_lower)
        
        # Оценка экспертизы и квалификации (Expertise)
        expertise_score = self._evaluate_expertise(text_lower)
        
        # Оценка авторитетности (Authoritativeness)
        authority_score = self._evaluate_authority(text_lower)
        
        # Оценка доверия (Trustworthiness)
        trust_score = self._evaluate_trust(text_lower)
        
        # Оценка структуры (влияет на восприятие E-E-A-T)
        structural_score = self._evaluate_structure(content)
        
        # Расчет общей оценки E-E-A-T
        # YMYL отрасли имеют более строгие требования к E-E-A-T
        is_ymyl = industry.lower() in self.ymyl_categories
        ymyl_modifier = 0.8 if is_ymyl else 1.0
        
        overall_eeat_score = (
            experience_score * 0.15 + 
            expertise_score * 0.25 + 
            authority_score * 0.25 + 
            trust_score * 0.25 + 
            structural_score * 0.1
        ) * ymyl_modifier
        
        # Генерация рекомендаций
        recommendations = self._generate_recommendations(
            experience_score, expertise_score, authority_score, trust_score, 
            structural_score, is_ymyl
        )
        
        # Результат анализа
        result = {
            'experience_score': experience_score,
            'expertise_score': expertise_score,
            'authority_score': authority_score,
            'trust_score': trust_score,
            'structural_score': structural_score,
            'overall_eeat_score': overall_eeat_score,
            'recommendations': recommendations,
            'ymyl_status': 1 if is_ymyl else 0,
            'component_details': {
                'expertise': {'found_markers': self._count_markers(text_lower, self.expertise_markers)},
                'authority': {'found_markers': self._count_markers(text_lower, self.authority_markers)},
                'trust': {'found_markers': self._count_markers(text_lower, self.trust_markers)},
                'experience': {'found_markers': self._count_markers(text_lower, self.experience_markers)}
            }
        }
        
        return result
    
    def _count_markers(self, text: str, markers: List[str]) -> Dict[str, int]:
        """Подсчитывает встречаемость маркеров в тексте."""
        result = {}
        for marker in markers:
            count = text.count(marker)
            if count > 0:
                result[marker] = count
        return result
    
    def _evaluate_experience(self, text: str) -> float:
        """Оценка личного опыта в контенте."""
        # Подсчитываем встречаемость маркеров опыта
        experience_count = sum(text.count(marker) for marker in self.experience_markers)
        
        # Нормализуем оценку (макс. 1.0)
        score = min(experience_count / 7, 0.9)  # Max 0.9 за количество маркеров
        
        # Проверка наличия личных местоимений (я, мы, мой, наш)
        personal_pronouns = ['я ', ' я ', 'мы ', ' мы ', 'мой ', 'наш ', ' мой ', ' наш ']
        pronouns_count = sum(text.count(pronoun) for pronoun in personal_pronouns)
        
        # Бонус за наличие личных местоимений
        if pronouns_count > 0:
            score += min(pronouns_count / 10, 0.1)  # Max 0.1 за местоимения
        
        return min(max(score, 0), 1.0)  # Ограничиваем от 0 до 1
    
    def _evaluate_expertise(self, text: str) -> float:
        """Оценка экспертности контента."""
        # Подсчитываем встречаемость маркеров экспертизы
        expertise_count = sum(text.count(marker) for marker in self.expertise_markers)
        
        # Нормализуем оценку (макс. 0.8)
        score = min(expertise_count / 10, 0.8)
        
        # Проверка наличия учёной степени
        has_academic_degree = bool(re.search(r'(доктор|кандидат|профессор|PhD|MD)', text))
        
        # Проверка наличия профессионального статуса
        has_professional_status = bool(re.search(r'(сертифицированный|лицензированный|аккредитованный)', text))
        
        # Бонусы за особые индикаторы
        if has_academic_degree:
            score += 0.1
        if has_professional_status:
            score += 0.1
        
        return min(max(score, 0), 1.0)  # Ограничиваем от 0 до 1
    
    def _evaluate_authority(self, text: str) -> float:
        """Оценка авторитетности контента."""
        # Подсчитываем встречаемость маркеров авторитетности
        authority_count = sum(text.count(marker) for marker in self.authority_markers)
        
        # Нормализуем оценку (макс. 0.7)
        score = min(authority_count / 12, 0.7)
        
        # Подсчет ссылок на внешние источники
        url_count = len(re.findall(r'https?://\S+', text))
        citation_count = len(re.findall(r'\[\d+\]', text))  # Ссылки вида [1], [2]
        
        # Бонусы за цитирование
        citation_bonus = min((url_count + citation_count) / 5, 0.2)
        
        # Проверка наличия упоминаний авторитетных источников
        authoritative_sources = [
            'исследование', 'университет', 'журнал', 'публикация', 'институт'
        ]
        auth_sources_count = sum(text.count(source) for source in authoritative_sources)
        sources_bonus = min(auth_sources_count / 5, 0.1)
        
        score += citation_bonus + sources_bonus
        
        return min(max(score, 0), 1.0)  # Ограничиваем от 0 до 1
    
    def _evaluate_trust(self, text: str) -> float:
        """Оценка доверия к контенту."""
        # Подсчитываем встречаемость маркеров доверия
        trust_count = sum(text.count(marker) for marker in self.trust_markers)
        
        # Нормализуем оценку (макс. 0.6)
        score = min(trust_count / 15, 0.6)
        
        # Проверка наличия дат
        date_count = len(re.findall(r'\d{4}\s?(?:год|г\.)', text))
        date_count += len(re.findall(r'\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}', text))
        
        # Проверка наличия дисклеймеров
        has_disclaimer = 'отказ от ответственности' in text.lower() or 'disclaimer' in text.lower()
        
        # Проверка наличия раздела с источниками
        has_sources_section = bool(re.search(r'источники|литература|references|sources', text))
        
        # Бонусы за дополнительные признаки
        if date_count > 0:
            score += min(date_count / 3, 0.1)
        if has_disclaimer:
            score += 0.1
        if has_sources_section:
            score += 0.2
        
        return min(max(score, 0), 1.0)  # Ограничиваем от 0 до 1
    
    def _evaluate_structure(self, content: str) -> float:
        """Оценка структуры контента."""
        # Анализируем структуру с помощью TextProcessor
        structure = self.text_processor.analyze_text_structure(content)
        
        # Проверяем наличие важных структурных элементов
        has_headers = structure['headers_count'] > 0
        has_paragraphs = structure['paragraphs_count'] > 0
        has_lists = structure['lists_count'] > 0
        has_intro_conclusion = structure['has_introduction'] and structure['has_conclusion']
        
        # Базовая оценка
        score = 0.3  # Минимальная оценка
        
        # Бонусы за структурные элементы
        if has_headers:
            score += 0.2
        if has_paragraphs and structure['paragraphs_count'] > 3:
            score += 0.2
        if has_lists:
            score += 0.1
        if has_intro_conclusion:
            score += 0.2
        
        return min(max(score, 0), 1.0)  # Ограничиваем от 0 до 1
    
    def _generate_recommendations(self, experience_score: float, expertise_score: float, 
                                authority_score: float, trust_score: float,
                                structural_score: float, is_ymyl: bool) -> List[str]:
        """Генерация рекомендаций на основе оценок E-E-A-T."""
        recommendations = []
        
        # Установка порога в зависимости от YMYL-статуса
        threshold = 0.5 if is_ymyl else 0.4
        
        # Рекомендации по опыту
        if experience_score < threshold:
            recommendations.append("Добавьте информацию о личном опыте автора с предметом")
        
        # Рекомендации по экспертизе
        if expertise_score < threshold:
            recommendations.append("Добавьте информацию о квалификации или образовании автора")
        
        # Рекомендации по авторитетности
        if authority_score < threshold:
            recommendations.append("Добавьте ссылки на авторитетные источники или исследования")
        
        # Рекомендации по доверию
        if trust_score < threshold:
            recommendations.append("Добавьте даты публикаций и обновлений контента")
            recommendations.append("Включите раздел с источниками информации")
        
        # Рекомендации по структуре
        if structural_score < threshold:
            recommendations.append("Улучшите структуру контента, добавив заголовки и подзаголовки")
        
        # Если YMYL-отрасль, добавляем специальные рекомендации
        if is_ymyl and (experience_score + expertise_score + authority_score + trust_score) / 4 < 0.6:
            recommendations.append(
                "Для YMYL-контента критически важно усилить сигналы экспертизы и авторитетности"
            )
        
        return recommendations
