"""Анализатор E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)."""

from typing import Dict, List, Optional, Union, Any
import re
from collections import Counter

class EEATAnalyzer:
    """Анализатор E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)"""
    
    def __init__(self):
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
        
        # Маркеры доверия - РАСШИРЕНО с учетом дат, методологий и раскрытия информации
        self.trust_markers = [
            'достоверный', 'проверенный', 'надежный', 'точный', 'подтвержденный',
            'официальный', 'гарантированный', 'безопасный', 'проверка фактов', 'прозрачность',
            'методология', 'метод', 'данные показывают', 'доказательство', 'подтверждено',
            'публикация', 'обновлено', 'раскрытие информации', 'отказ от ответственности', 'дисклеймер'
        ]
        
        # Шаблоны для поиска важных маркеров доверия
        self.special_trust_patterns = [
            r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}',  # Даты
            r'\d{4} год',  # Годы
            r'\d{1,2} [а-я]+ \d{4}',  # Даты в текстовом формате
            r'\d+%',  # Проценты
            r'источник',
            r'автор'
        ]
    
    def analyze(self, content: str, industry: str = 'default') -> Dict[str, Union[float, Dict, List]]:
        """
        Оценка E-E-A-T сигналов в тексте
        
        Args:
            content: Текстовое содержимое
            industry: Отрасль контента
            
        Returns:
            Словарь с оценками и рекомендациями
        """
        text_lower = content.lower()
        
        # Оценка экспертизы и опыта
        expertise_score = self._evaluate_expertise(text_lower)
        
        # Оценка авторитетности
        authority_score = self._evaluate_authority(text_lower)
        
        # Оценка доверия
        trust_score = self._evaluate_trust(text_lower)
        
        # Оценка структуры
        structural_score = self._evaluate_structure(content)
        
        # Расчет общей оценки E-E-A-T
        # YMYL отрасли имеют более строгие требования к E-E-A-T
        is_ymyl = industry in ['finance', 'health', 'legal', 'medical']
        ymyl_modifier = 0.8 if is_ymyl else 1.0
        
        overall_eeat_score = (
            expertise_score * 0.3 + 
            authority_score * 0.3 + 
            trust_score * 0.25 + 
            structural_score * 0.15
        ) * ymyl_modifier
        
        # Генерация рекомендаций
        recommendations = self._generate_recommendations(
            expertise_score, authority_score, trust_score, structural_score, is_ymyl
        )
        
        # Информация о найденных маркерах для диагностики
        expertise_markers_found = [marker for marker in self.expertise_markers if marker in text_lower]
        authority_markers_found = [marker for marker in self.authority_markers if marker in text_lower]
        trust_markers_found = [marker for marker in self.trust_markers if marker in text_lower]
        
        return {
            'expertise_score': expertise_score,
            'authority_score': authority_score,
            'trust_score': trust_score,
            'structural_score': structural_score,
            'overall_eeat_score': overall_eeat_score,
            'recommendations': recommendations,
            'ymyl_status': 1 if is_ymyl else 0,
            'component_details': {
                'expertise': {'found_markers': expertise_markers_found},
                'authority': {'found_markers': authority_markers_found},
                'trust': {'found_markers': trust_markers_found}
            }
        }
    
    def _evaluate_expertise(self, text: str) -> float:
        """Оценка экспертности контента"""
        # Базовая реализация подсчета маркеров
        expertise_count = sum(text.count(marker) for marker in self.expertise_markers)
        score = min(expertise_count / 10, 0.8)
        return score
    
    def _evaluate_authority(self, text: str) -> float:
        """Оценка авторитетности контента"""
        # Базовая реализация подсчета маркеров
        authority_count = sum(text.count(marker) for marker in self.authority_markers)
        score = min(authority_count / 12, 0.7)
        return score
    
    def _evaluate_trust(self, text: str) -> float:
        """Оценка доверия к контенту"""
        # Базовая реализация подсчета маркеров
        trust_count = sum(text.count(marker) for marker in self.trust_markers)
        
        # Поиск дополнительных маркеров доверия
        special_markers_count = 0
        for pattern in self.special_trust_patterns:
            matches = re.findall(pattern, text)
            special_markers_count += len(matches)
        
        score = min((trust_count / 15) + (special_markers_count / 10), 1.0)
        return score
    
    def _evaluate_structure(self, content: str) -> float:
        """Оценка структуры контента"""
        # Упрощенная оценка структуры
        has_headings = '#' in content
        has_lists = '-' in content or '*' in content
        has_paragraphs = '\n\n' in content

        
        score = 0.3
        if has_headings:
            score += 0.3
        if has_lists:
            score += 0.2
        if has_paragraphs:
            score += 0.2
            
        return min(score, 1.0)
    
    def _generate_recommendations(
        self, 
        expertise_score: float, 
        authority_score: float, 
        trust_score: float,
        structural_score: float,
        is_ymyl: bool
    ) -> List[str]:
        """Генерация рекомендаций на основе оценок E-E-A-T"""
        recommendations = []
        
        # Порог для рекомендаций зависит от YMYL-статуса
        threshold = 0.4 if is_ymyl else 0.3
        
        # Экспертиза
        if expertise_score < threshold:
            recommendations.append("Добавьте информацию о профессиональном опыте автора")
            recommendations.append("Включите подтверждение компетенции (образование, сертификаты)")
        
        # Авторитетность
        if authority_score < threshold:
            recommendations.append("Добавьте ссылки на авторитетные источники")
            recommendations.append("Включите статистические данные из проверенных исследований")
        
        # Доверие
        if trust_score < threshold:
            recommendations.append("Добавьте даты публикации и обновления материала")
            recommendations.append("Включите методологию или источники данных")
        
        # Структура
        if structural_score < threshold:
            recommendations.append("Улучшите структуру контента, добавив подзаголовки")
            recommendations.append("Используйте маркированные списки для лучшей читаемости")
        
        # Дополнительные YMYL-рекомендации
        if is_ymyl and (expertise_score + authority_score + trust_score) / 3 < 0.5:
            recommendations.append("Для YMYL-контента критически важно усилить все E-E-A-T сигналы")
        
        return recommendations
