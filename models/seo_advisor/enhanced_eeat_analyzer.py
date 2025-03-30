
import logging
import joblib
import re
from typing import Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

class EnhancedEEATAnalyzer:
    """Улучшенный анализатор E-E-A-T с поддержкой машинного обучения"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация анализатора
        
        Args:
            model_path: Путь к модели машинного обучения
        """
        self.model = None
        self.ml_model_used = False
        
        # Маркеры опыта и экспертизы
        self.expertise_markers = [
            'эксперт', 'специалист', 'профессионал', 'опыт', 'лет опыта', 
            'квалификация', 'сертифицированный', 'компетентный', 'практик', 
            'исследователь', 'аналитик', 'степень', 'образование'
        ]
        
        # Маркеры авторитетности
        self.authority_markers = [
            'исследование', 'статистика', 'данные', 'доказано', 'согласно', 
            'по мнению экспертов', 'научно', 'источник', 'цитата', 'ссылка',
            'исследователи', 'университет', 'институт', 'лаборатория', 'академия'
        ]
        
        # Маркеры доверия
        self.trust_markers = [
            'достоверный', 'проверенный', 'надежный', 'точный', 'подтвержденный',
            'официальный', 'гарантированный', 'безопасный', 'проверка фактов', 
            'прозрачность', 'методология', 'метод', 'доказательство', 'подтверждено',
            'публикация', 'обновлено', 'раскрытие информации', 'отказ от ответственности'
        ]
        
        # Загрузка модели машинного обучения
        if model_path:
            try:
                self.model = joblib.load(model_path)
                self.ml_model_used = True
                logger.info(f"Модель машинного обучения загружена из {model_path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}")
    
    def analyze(self, content: str, industry: str = 'default') -> Dict[str, Union[float, List[str]]]:
        """
        Анализ содержимого и оценка E-E-A-T метрик
        
        Args:
            content: Текстовое содержимое
            industry: Отрасль контента
            
        Returns:
            Словарь с оценками и рекомендациями
        """
        # Анализ основных компонентов E-E-A-T
        expertise_score = self._evaluate_expertise(content)
        authority_score = self._evaluate_authority(content)
        trust_score = self._evaluate_trust(content)
        structural_score = self._evaluate_structure(content)
        semantic_coherence_score = self._evaluate_semantic_coherence(content)
        
        # Подсчет цитирований и внешних ссылок
        citation_score = self._count_citations(content)
        external_links_score = self._count_external_links(content)
        
        # Определение YMYL-статуса
        ymyl_status = 1 if industry in ['finance', 'health', 'legal', 'medical'] else 0
        
        # Использование модели машинного обучения для расчета общего скора, если доступна
        overall_eeat_score = 0.0
        
        if self.model and self.ml_model_used:
            try:
                # Подготовка данных для модели
                features = np.array([
                    expertise_score,
                    authority_score,
                    trust_score,
                    structural_score,
                    semantic_coherence_score,
                    citation_score,
                    external_links_score,
                    ymyl_status
                ]).reshape(1, -1)
                
                # Предсказание
                overall_eeat_score = float(self.model.predict(features)[0])
                logger.info(f"Оценка E-E-A-T с использованием модели МО: {overall_eeat_score:.4f}")
            except Exception as e:
                logger.error(f"Ошибка при использовании модели МО: {e}")
                self.ml_model_used = False
        
        # Если модель недоступна, используем формулу
        if not self.model or not self.ml_model_used:
            # Весовые коэффициенты для расчета общего скора
            weights = {
                'expertise': 0.25,
                'authority': 0.30,
                'trust': 0.25,
                'structure': 0.10,
                'semantic': 0.10
            }
            
            # YMYL индустрии требуют более высоких E-E-A-T показателей
            ymyl_modifier = 0.8 if ymyl_status else 1.0
            
            # Расчет общего скора
            overall_eeat_score = (
                expertise_score * weights['expertise'] +
                authority_score * weights['authority'] +
                trust_score * weights['trust'] +
                structural_score * weights['structure'] +
                semantic_coherence_score * weights['semantic']
            ) * ymyl_modifier
        
        # Генерация рекомендаций на основе оценок
        recommendations = self._generate_recommendations(
            expertise_score, authority_score, trust_score, 
            structural_score, semantic_coherence_score,
            overall_eeat_score, ymyl_status
        )
        
        return {
            'expertise_score': expertise_score,
            'authority_score': authority_score,
            'trust_score': trust_score,
            'structural_score': structural_score,
            'semantic_coherence_score': semantic_coherence_score,
            'citation_score': citation_score,
            'external_links_score': external_links_score,
            'overall_eeat_score': overall_eeat_score,
            'recommendations': recommendations,
            'ml_model_used': self.ml_model_used
        }
    
    def _evaluate_expertise(self, content: str) -> float:
        """Оценка экспертности контента"""
        content_lower = content.lower()
        
        # Подсчет упоминаний маркеров экспертности
        expertise_count = sum(content_lower.count(marker) for marker in self.expertise_markers)
        
        # Поиск информации об авторе
        author_info_present = bool(re.search(r'автор|об авторе|специалист|эксперт', content_lower))
        
        # Поиск упоминаний квалификации
        qualification_present = bool(re.search(r'стаж|опыт работы|степень|диплом|сертификат|образование', content_lower))
        
        # Базовая оценка на основе количества маркеров
        base_score = min(expertise_count / 10, 0.8)
        
        # Бонусы за наличие информации
        author_bonus = 0.2 if author_info_present else 0
        qualification_bonus = 0.3 if qualification_present else 0
        
        return min(base_score + author_bonus + qualification_bonus, 1.0)
    
    def _evaluate_authority(self, content: str) -> float:
        """Оценка авторитетности контента"""
        content_lower = content.lower()
        
        # Подсчет упоминаний маркеров авторитетности
        authority_count = sum(content_lower.count(marker) for marker in self.authority_markers)
        
        # Поиск ссылок на исследования и статистику
        research_refs_present = bool(re.search(r'исследовани[а-я]|анализ|статистик[а-я]|данны[а-я]', content_lower))
        
        # Поиск цитат/мнений экспертов
        expert_quotes_present = bool(re.search(r'эксперты считают|по мнению|специалисты|цитат[а-я]', content_lower))
        
        # Поиск упоминаний авторитетных организаций
        authority_orgs_present = bool(re.search(r'университет|институт|центр|организаци[а-я]|компани[а-я]', content_lower))
        
        # Базовая оценка на основе количества маркеров
        base_score = min(authority_count / 12, 0.7)
        
        # Бонусы за наличие конкретных элементов
        research_bonus = 0.3 if research_refs_present else 0
        quotes_bonus = 0.2 if expert_quotes_present else 0
        orgs_bonus = 0.2 if authority_orgs_present else 0
        
        return min(base_score + research_bonus + quotes_bonus + orgs_bonus, 1.0)
    
    def _evaluate_trust(self, content: str) -> float:
        """Оценка доверия к контенту"""
        content_lower = content.lower()
        
        # Подсчет упоминаний маркеров доверия
        trust_count = sum(content_lower.count(marker) for marker in self.trust_markers)
        
        # Поиск дат и временных ориентиров
        dates_present = bool(re.search(r'\d{4}|20\d\d|январ[а-я]|феврал[а-я]|март[а-я]|апрел[а-я]|ма[а-я]|июн[а-я]|июл[а-я]|август[а-я]|сентябр[а-я]|октябр[а-я]|ноябр[а-я]|декабр[а-я]', content_lower))
        
        # Поиск конкретных цифр и статистики
        specific_data_present = bool(re.search(r'\d+%|\d+ процент|\d+ человек|\d+ случаев', content_lower))
        
        # Поиск источников и ссылок
        sources_present = bool(re.search(r'источник|ссылк[а-я]|литератур[а-я]|библиограф|согласно', content_lower))
        
        # Проверка наличия структурированных данных
        structured_data_present = len(re.findall(r'\d\.\s|•\s|\*\s|-\s', content)) > 3
        
        # Базовая оценка на основе количества маркеров
        base_score = min(trust_count / 15, 0.6)
        
        # Бонусы за наличие конкретных элементов
        dates_bonus = 0.1 if dates_present else 0
        data_bonus = 0.2 if specific_data_present else 0
        sources_bonus = 0.3 if sources_present else 0
        structure_bonus = 0.1 if structured_data_present else 0
        
        return min(base_score + dates_bonus + data_bonus + sources_bonus + structure_bonus, 1.0)
    
    def _evaluate_structure(self, content: str) -> float:
        """Оценка структуры контента"""
        # Поиск заголовков разных уровней
        h1_count = len(re.findall(r'^#\s.*$|^#{1}[^#].*$', content, re.MULTILINE))
        h2_count = len(re.findall(r'^##\s.*$|^#{2}[^#].*$', content, re.MULTILINE))
        h3_count = len(re.findall(r'^###\s.*$|^#{3}[^#].*$', content, re.MULTILINE))
        
        # Поиск списков
        list_items = len(re.findall(r'^\s*[-*•]\s+.*$|^\s*\d+\.\s+.*$', content, re.MULTILINE))
        
        # Поиск разделителей (горизонтальные линии, переносы строк)
        separators = len(re.findall(r'^\s*[-*=]{3,}\s*$', content, re.MULTILINE))
        
        # Подсчет абзацев
        paragraphs = len(re.findall(r'(?:\n\n|\r\n\r\n)', content)) + 1
        
        # Оценка структуры на основе наличия элементов
        headers_score = min((h1_count + h2_count * 0.8 + h3_count * 0.6) / 5, 1.0)
        lists_score = min(list_items / 10, 1.0)
        paragraphs_score = min(paragraphs / 8, 1.0)
        
        # Оценка иерархии заголовков
        hierarchy_score = 0.0
        if h1_count >= 1 and h2_count >= 2:
            hierarchy_score = 0.5
            if h3_count >= 2:
                hierarchy_score = 1.0
        elif h1_count == 0 and h2_count >= 3:
            hierarchy_score = 0.4
        
        # Комбинированная оценка структуры
        return 0.4 * headers_score + 0.3 * lists_score + 0.1 * paragraphs_score + 0.2 * hierarchy_score
    
    def _evaluate_semantic_coherence(self, content: str) -> float:
        """Оценка семантической связности контента"""
        # В идеале здесь должен быть более сложный алгоритм анализа связности,
        # но для примера используем упрощенный подход
        
        content_lower = content.lower()
        paragraphs = re.split(r'\n\n|\r\n\r\n', content_lower)
        
        # Проверка связок между параграфами
        coherence_words = ['однако', 'но', 'поэтому', 'следовательно', 'таким образом', 
                          'кроме того', 'более того', 'например', 'с другой стороны']
        
        coherence_count = 0
        for paragraph in paragraphs:
            for word in coherence_words:
                if word in paragraph:
                    coherence_count += 1
                    break
        
        # Проверка повторения ключевых терминов
        # Извлекаем самые частые слова (исключая стоп-слова)
        words = re.findall(r'\b[а-яА-Яa-zA-Z][а-яА-Яa-zA-Z-]{3,}\b', content_lower)
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'не', 'это', 'что', 'как', 'он', 'она', 'они', 'мы', 'вы'}
        
        word_counts = {}
        for word in words:
            if word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Топ-5 слов по частоте
        if word_counts:
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Проверка, повторяются ли эти слова в разных параграфах
            term_coherence = 0
            for word, _ in top_words:
                paragraph_coverage = sum(1 for p in paragraphs if word in p)
                if paragraph_coverage >= len(paragraphs) / 3:
                    term_coherence += 1
            
            term_coherence_score = min(term_coherence / 5, 1.0)
        else:
            term_coherence_score = 0.0
        
        # Оценка длины параграфов и их равномерности
        avg_paragraph_length = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0
        
        paragraph_score = 1.0
        if avg_paragraph_length < 20:
            paragraph_score = 0.3
        elif avg_paragraph_length > 300:
            paragraph_score = 0.4
        
        # Комбинированная оценка семантической связности
        coherence_score = 0.3 * min(coherence_count / len(paragraphs), 1.0) if paragraphs else 0
        
        return 0.4 * term_coherence_score + 0.3 * coherence_score + 0.3 * paragraph_score
    
    def _count_citations(self, content: str) -> float:
        """Подсчет цитат и ссылок на источники"""
        content_lower = content.lower()
        
        # Поиск явных цитат
        quotes = len(re.findall(r'"[^"]+"|\"[^\"]+\"', content))
        
        # Поиск ссылок на источники
        sources = len(re.findall(r'источник[а-я]?|ссылк[а-я]|согласно|по данным', content_lower))
        
        # Поиск академических ссылок в формате [1], [2] и т.д.
        academic_refs = len(re.findall(r'\[\d+\]', content))
        
        # Поиск фраз типа "исследование показало"
        research_phrases = len(re.findall(r'исследован[а-я]{1,4} показал[а-я]|анализ выявил|данные подтверждают', content_lower))
        
        # Расчет нормализованного скора
        citation_score = 0.1 * quotes + 0.3 * sources + 0.4 * academic_refs + 0.2 * research_phrases
        
        return min(citation_score / 5, 1.0)
    
    def _count_external_links(self, content: str) -> float:
        """Оценка наличия внешних ссылок"""
        # Поиск URL-адресов
        urls = len(re.findall(r'https?://[^\s]+|www\.[^\s]+', content.lower()))
        
        # Поиск упоминаний доменов
        domains = len(re.findall(r'\.[a-z]{2,}/', content.lower()))
        
        # Поиск фраз типа "ссылка на источник"
        link_phrases = len(re.findall(r'ссылк[а-я] на|источник[а-я]?:|библиография:|литература:', content.lower()))
        
        return min((urls + domains * 0.5 + link_phrases * 0.3) / 5, 1.0)
    
    def _generate_recommendations(
        self,
        expertise_score: float,
        authority_score: float,
        trust_score: float,
        structural_score: float,
        semantic_score: float,
        overall_score: float,
        is_ymyl: int
    ) -> List[str]:
        """Генерация рекомендаций на основе оценок"""
        recommendations = []
        
        # Порог для рекомендаций
        threshold = 0.4 if is_ymyl else 0.3
        
        # Рекомендации по экспертности
        if expertise_score < threshold:
            recommendations.extend([
                "Добавьте явную информацию о профессиональном опыте автора и его квалификации",
                "Включите подтверждение компетенции автора (образование, сертификаты, опыт работы)",
                "Добавьте экспертное мнение по ключевым вопросам темы"
            ])
        elif expertise_score < threshold * 2:
            recommendations.append(
                "Усильте демонстрацию экспертности, добавив примеры из практики"
            )
        
        # Рекомендации по авторитетности
        if authority_score < threshold:
            recommendations.extend([
                "Добавьте ссылки на авторитетные источники по теме",
                "Включите статистические данные из проверенных исследований",
                "Цитируйте признанных экспертов отрасли"
            ])
        elif authority_score < threshold * 2:
            recommendations.append(
                "Улучшите авторитетность контента, добавив больше ссылок на внешние источники"
            )
        
        # Рекомендации по доверию
        if trust_score < threshold:
            recommendations.extend([
                "Добавьте даты публикации и обновления материала",
                "Включите точные цифры и статистику с указанием источников",
                "Добавьте раздел с источниками или список литературы"
            ])
        elif trust_score < threshold * 2:
            recommendations.append(
                "Улучшите прозрачность, указывая источники для ключевых утверждений"
            )
        
        # Рекомендации по структуре
        if structural_score < threshold:
            recommendations.extend([
                "Улучшите структуру контента, добавив подзаголовки (H2, H3)",
                "Используйте маркированные списки для перечислений",
                "Разбейте текст на более короткие абзацы для лучшей читаемости"
            ])
        
        # Рекомендации по семантике
        if semantic_score < threshold:
            recommendations.extend([
                "Повысьте связность текста, используя переходные слова между разделами",
                "Обеспечьте последовательность изложения, выстраивая логическую структуру",
                "Убедитесь, что ключевые термины повторяются по всему тексту"
            ])
        
        # YMYL-специфичные рекомендации
        if is_ymyl and overall_score < 0.6:
            recommendations.extend([
                "Для YMYL-контента крайне важно усилить E-E-A-T сигналы",
                "Добавьте дисклеймеры и предупреждения, где это необходимо",
                "Включите ссылки на научные исследования и официальные рекомендации"
            ])
        
        return recommendations
