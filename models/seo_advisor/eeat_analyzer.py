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
        
        # Маркеры доверия
        self.trust_markers = [
            'достоверный', 'проверенный', 'надежный', 'точный', 'подтвержденный',
            'официальный', 'гарантированный', 'безопасный', 'проверка фактов', 'прозрачность'
        ]
    
    def analyze(self, text: str) -> Dict[str, Union[float, Dict, List]]:
        """
        Оценка E-E-A-T сигналов в тексте
        
        Args:
            text: Анализируемый текст
            
        Returns:
            Словарь с оценками и рекомендациями
        """
        text_lower = text.lower()
        
        # Оценка опыта и экспертизы
        expertise_score = self._evaluate_markers(text_lower, self.expertise_markers)
        
        # Оценка авторитетности
        authority_score = self._evaluate_markers(text_lower, self.authority_markers)
        
        # Оценка доверия
        trust_score = self._evaluate_markers(text_lower, self.trust_markers)
        
        # Оценка структурных элементов, укрепляющих E-E-A-T
        structural_score = self._evaluate_structure(text)
        
        # Средневзвешенная оценка
        overall_eeat_score = (
            expertise_score * 0.3 + 
            authority_score * 0.3 + 
            trust_score * 0.2 + 
            structural_score * 0.2
        )
        
        # Формирование рекомендаций
        recommendations = self._generate_recommendations(
            expertise_score, authority_score, trust_score, structural_score
        )
        
        return {
            'expertise_score': expertise_score,
            'authority_score': authority_score,
            'trust_score': trust_score,
            'structural_score': structural_score,
            'overall_eeat_score': overall_eeat_score,
            'recommendations': recommendations
        }
    
    def _evaluate_markers(self, text: str, markers: List[str]) -> float:
        """Оценка наличия маркеров в тексте"""
        total_markers = len(markers)
        found_markers = sum(1 for marker in markers if marker in text)
        
        # Оценка на основе количества найденных маркеров
        marker_score = min(found_markers / (total_markers / 2), 1.0)
        
        # Оцениваем распределение маркеров
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        distribution_score = 0.0
        
        if paragraphs:
            markers_per_paragraph = []
            for paragraph in paragraphs:
                markers_found = sum(1 for marker in markers if marker in paragraph)
                markers_per_paragraph.append(markers_found)
            
            # Оценка распределения маркеров по параграфам
            if sum(markers_per_paragraph) > 0:
                distribution_score = min(1.0, len([m for m in markers_per_paragraph if m > 0]) / len(paragraphs))
        
        # Комбинированная оценка
        return 0.7 * marker_score + 0.3 * distribution_score
    
    def _evaluate_structure(self, text: str) -> float:
        """Оценка структурных элементов, укрепляющих E-E-A-T"""
        score = 0.0
        
        # Проверка наличия цитат - упрощенная версия
        quotes = text.count('"') + text.count("'") + text.count("«")
        quotes_score = min(quotes / 6, 1.0)  # Делим на 6, т.к. каждая цитата имеет открывающую и закрывающую кавычки
        
        # Проверка наличия ссылок на источники - упрощенная версия
        sources = text.lower().count('источник') + text.lower().count('http')
        sources_score = min(sources / 2, 1.0)
        
        # Проверка наличия структурированных данных (списки) - упрощенная версия
        lists = text.count('-') + text.count('*') + text.count('•')
        lists_score = min(lists / 5, 1.0)
        
        # Проверка наличия подзаголовков - упрощенная версия
        headers = text.count('#') + text.count('##')
        headers_score = min(headers / 3, 1.0)
        
        # Взвешенная оценка структуры
        score = (
            quotes_score * 0.25 +
            sources_score * 0.3 +
            lists_score * 0.2 +
            headers_score * 0.25
        )
        
        return score
    
    def _generate_recommendations(
        self, 
        expertise_score: float, 
        authority_score: float, 
        trust_score: float,
        structural_score: float
    ) -> List[str]:
        """Генерация рекомендаций на основе оценок E-E-A-T"""
        recommendations = []
        
        # Рекомендации по экспертизе
        if expertise_score < 0.3:
            recommendations.extend([
                "Добавьте информацию о профессиональном опыте автора в данной области",
                "Включите подтверждение компетенции автора (образование, сертификаты, опыт работы)",
                "Добавьте экспертное мнение по ключевым вопросам темы"
            ])
        elif expertise_score < 0.6:
            recommendations.append("Усильте демонстрацию экспертности, добавив примеры из практики")
        
        # Рекомендации по авторитетности
        if authority_score < 0.3:
            recommendations.extend([
                "Добавьте ссылки на авторитетные источники по теме",
                "Включите статистические данные из проверенных исследований",
                "Цитируйте признанных экспертов отрасли"
            ])
        elif authority_score < 0.6:
            recommendations.append("Усильте авторитетность контента, добавив больше ссылок на внешние источники")
        
        # Рекомендации по доверию
        if trust_score < 0.3:
            recommendations.extend([
                "Включите точную и проверяемую информацию",
                "Добавьте прозрачность в методологию и источники данных",
                "Укажите даты публикации использованных источников"
            ])
        elif trust_score < 0.6:
            recommendations.append("Повысьте прозрачность и достоверность представленных фактов")
        
        # Рекомендации по структуре
        if structural_score < 0.3:
            recommendations.extend([
                "Добавьте структурные элементы для подтверждения E-E-A-T: цитаты, ссылки, списки",
                "Улучшите структуру контента, добавив подзаголовки и разделы",
                "Включите списки и таблицы для структурирования информации"
            ])
        elif structural_score < 0.6:
            recommendations.append("Улучшите структуру текста для более наглядной демонстрации экспертности")
        
        return recommendations
