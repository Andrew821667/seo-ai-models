from typing import Dict, List, Union, Optional, Any
import re

class AdvancedEEATAnalyzer:
    """Расширенный анализатор E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)"""
    
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
        
        # Маркеры доверия
        self.trust_markers = [
            'достоверный', 'проверенный', 'надежный', 'точный', 'подтвержденный',
            'официальный', 'гарантированный', 'безопасный', 'проверка фактов', 'прозрачность',
            'методология', 'метод', 'данные показывают', 'доказательство', 'подтверждено',
            'публикация', 'обновлено', 'раскрытие информации', 'отказ от ответственности', 'дисклеймер'
        ]
        
        # Шаблоны регулярных выражений
        self.special_trust_patterns = [
            r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}',  # Даты в различных форматах
            r'\d{4} год',  # Годы
            r'\d{1,2} [а-я]+ \d{4}',  # Даты в текстовом формате
            r'\d+%',  # Проценты
            r'\d+\s*(?:млн|млрд|тыс)',  # Числовые значения с единицами измерения
        ]

    def analyze(self, text: str, industry: str = 'default') -> Dict[str, Union[float, List[str]]]:
        """Анализ E-E-A-T для различных отраслей"""
        text_lower = text.lower()
        
        # Расчет компонентов
        expertise_score = self._evaluate_markers(text_lower, self.expertise_markers)
        authority_score = self._evaluate_markers(text_lower, self.authority_markers)
        trust_score = self._evaluate_trust(text, text_lower)
        structural_score = self._evaluate_structure(text)
        
        # Расчет общего скора
        overall_score = (
            expertise_score * 0.3 + 
            authority_score * 0.3 + 
            trust_score * 0.25 + 
            structural_score * 0.15
        )
        
        # Генерация рекомендаций
        recommendations = self._generate_recommendations(
            expertise_score, authority_score, trust_score, structural_score
        )
        
        return {
            'expertise_score': expertise_score,
            'authority_score': authority_score,
            'trust_score': trust_score,
            'structural_score': structural_score,
            'overall_eeat_score': overall_score,
            'recommendations': recommendations
        }
    
    def _evaluate_markers(self, text: str, markers: List[str]) -> float:
        total_markers = len(markers)
        found_markers = sum(1 for marker in markers if marker in text)
        
        marker_score = min(found_markers / (total_markers / 2), 1.0)
        
        return marker_score
    
    def _evaluate_trust(self, text: str, text_lower: str) -> float:
        """Расширенная оценка доверия"""
        base_trust_score = self._evaluate_markers(text_lower, self.trust_markers)
        
        special_markers_count = sum(
            len(re.findall(pattern, text)) 
            for pattern in self.special_trust_patterns
        )
        
        special_score = min(special_markers_count / 10, 1.0)
        
        return min(base_trust_score + special_score, 1.0)
    
    def _evaluate_structure(self, text: str) -> float:
        """Оценка структуры текста"""
        quotes = text.count('"') + text.count("'") + text.count("«")
        lists = text.count('-') + text.count('*') + text.count('•')
        headers = text.count('#') + text.count('##')
        
        quotes_score = min(quotes / 6, 1.0)
        lists_score = min(lists / 5, 1.0)
        headers_score = min(headers / 3, 1.0)
        
        return (quotes_score + lists_score + headers_score) / 3
    
    def _generate_recommendations(
        self, 
        expertise_score: float, 
        authority_score: float, 
        trust_score: float,
        structural_score: float
    ) -> List[str]:
        recommendations = []
        
        if expertise_score < 0.3:
            recommendations.append("Добавьте информацию о профессиональном опыте автора")
        
        if authority_score < 0.3:
            recommendations.append("Включите ссылки на авторитетные источники")
        
        if trust_score < 0.3:
            recommendations.append("Добавьте даты и источники информации")
        
        if structural_score < 0.3:
            recommendations.append("Улучшите структуру текста: добавьте подзаголовки и списки")
        
        return recommendations
