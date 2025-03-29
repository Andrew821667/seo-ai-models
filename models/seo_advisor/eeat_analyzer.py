
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
        
        # Добавляем шаблоны регулярных выражений для поиска особых маркеров доверия
        self.special_trust_patterns = [
            r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}',  # Даты в различных форматах
            r'\d{4} год',  # Годы
            r'\d{1,2} [а-я]+ \d{4}',  # Даты в текстовом формате
            r'\d+%',  # Проценты
            r'\d+\s*(?:млн|млрд|тыс)',  # Числовые значения с единицами измерения
            r'(?:согласно|по данным|по информации)[^.!?]*источник[^.!?]*',  # Ссылки на источники
            r'автор[^.!?]*опыт[^.!?]*',  # Информация об опыте автора
            r'раскрытие информации[^.!?]*',  # Секции с раскрытием информации
            r'методология[^.!?]*',  # Секции с методологией
            r'об авторе[^.!?]*',  # Информация об авторе
            r'ссылк[а-я] на исследовани[а-я]',  # Ссылки на исследования
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
        
        # Оценка доверия - УЛУЧШЕНО с использованием расширенного анализа
        trust_score = self._evaluate_trust(text, text_lower)
        
        # Оценка структурных элементов, укрепляющих E-E-A-T
        structural_score = self._evaluate_structure(text)
        
        # Средневзвешенная оценка - УЛУЧШЕНО с акцентом на доверие для YMYL
        overall_eeat_score = (
            expertise_score * 0.3 + 
            authority_score * 0.3 + 
            trust_score * 0.25 +  # Увеличен вес доверия
            structural_score * 0.15
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
                markers_found = sum(1 for marker in markers if marker in paragraph.lower())
                markers_per_paragraph.append(markers_found)
            
            # Оценка распределения маркеров по параграфам
            if sum(markers_per_paragraph) > 0:
                distribution_score = min(1.0, len([m for m in markers_per_paragraph if m > 0]) / len(paragraphs))
        
        # Комбинированная оценка
        return 0.7 * marker_score + 0.3 * distribution_score
    
    def _evaluate_trust(self, text: str, text_lower: str) -> float:
        """УЛУЧШЕННАЯ оценка доверия с учетом дат, методологий и других специальных сигналов"""
        # Базовая оценка на основе маркеров
        base_trust_score = self._evaluate_markers(text_lower, self.trust_markers)
        
        # Поиск специальных маркеров доверия с помощью регулярных выражений
        special_markers_count = 0
        for pattern in self.special_trust_patterns:
            matches = re.findall(pattern, text)
            special_markers_count += len(matches)
        
        # Оценка наличия специальных маркеров доверия
        special_score = min(special_markers_count / 10, 1.0)
        
        # Проверка наличия важных секций (методология, об авторе, раскрытие информации)
        important_sections_score = 0.0
        if "методология" in text_lower or "источники" in text_lower:
            important_sections_score += 0.4
        
        if "об авторе" in text_lower or "автор:" in text_lower:
            important_sections_score += 0.3
        
        if "раскрытие информации" in text_lower or "дисклеймер" in text_lower:
            important_sections_score += 0.3
        
        # Поиск дат в тексте
        date_patterns = [r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}', r'\d{4} год', r'\d{1,2} [а-я]+ \d{4}']
        dates_found = False
        for pattern in date_patterns:
            if re.search(pattern, text):
                dates_found = True
                break
        
        date_score = 0.2 if dates_found else 0.0
        
        # Комбинированная оценка доверия
        trust_score = (
            base_trust_score * 0.4 +
            special_score * 0.3 +
            important_sections_score * 0.2 +
            date_score * 0.1
        )
        
        return trust_score
    
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
        
        # Рекомендации по доверию - УЛУЧШЕНО с конкретными рекомендациями
        if trust_score < 0.3:
            recommendations.extend([
                "Добавьте даты публикации и обновления материала",
                "Включите методологию или источники используемых данных",
                "Добавьте раздел с раскрытием информации о возможных конфликтах интересов",
                "Включите точные цифры и статистику с указанием источников"
            ])
        elif trust_score < 0.6:
            recommendations.extend([
                "Расширьте информацию о методологии сбора данных",
                "Добавьте больше точных дат и временных рамок",
                "Усильте прозрачность с помощью раскрытия дополнительной информации"
            ])
        
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
