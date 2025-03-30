
from typing import Dict, List
from seo_ai_models.common.utils.text_processing import TextProcessor

class ContentAnalyzer:
    def __init__(self):
        self.text_processor = TextProcessor()

    def analyze_text(self, content: str) -> Dict[str, float]:
        """Анализ текста и расчет метрик"""
        metrics = {}
        
        # Базовые метрики
        words = self.text_processor.tokenize(content)
        sentences = self.text_processor.split_sentences(content)
        
        metrics['word_count'] = len(words)
        metrics['sentence_count'] = len(sentences)
        metrics['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        
        # Оценка читабельности
        metrics['readability'] = self._calculate_readability(content)
        
        # Анализ заголовков
        headers = self.text_processor.extract_headers(content)
        metrics['header_score'] = self._calculate_header_score(headers)
        
        # Другие метрики
        metrics['meta_score'] = 0.7  # Заглушка
        metrics['multimedia_score'] = 0.5  # Заглушка
        metrics['linking_score'] = 0.6  # Заглушка
        
        # Семантический анализ
        metrics['semantic_depth'] = 0.75  # Заглушка
        metrics['topic_relevance'] = 0.8  # Заглушка
        metrics['engagement_potential'] = 0.7  # Заглушка
        
        return metrics

    def extract_keywords(self, content: str, target_keywords: List[str]) -> Dict[str, float]:
        """Анализ ключевых слов"""
        normalized_content = self.text_processor.normalize(content)
        word_count = len(self.text_processor.tokenize(normalized_content))
        
        keyword_stats = {
            'density': 0.0,
            'distribution': {},
            'frequency': {}
        }
        
        for keyword in target_keywords:
            count = normalized_content.count(keyword.lower())
            if word_count > 0:
                density = count / word_count
                keyword_stats['frequency'][keyword] = count
                keyword_stats['distribution'][keyword] = density
        
        # Общая плотность ключевых слов
        total_matches = sum(keyword_stats['frequency'].values())
        keyword_stats['density'] = total_matches / word_count if word_count > 0 else 0
        
        return keyword_stats

    def _calculate_readability(self, text: str) -> float:
        """Расчет читабельности текста"""
        words = len(self.text_processor.tokenize(text))
        sentences = len(self.text_processor.split_sentences(text))
        if not sentences or not words:
            return 0
        
        avg_sentence_length = words / sentences
        readability_score = 206.835 - 1.015 * avg_sentence_length
        
        # Нормализация оценки к диапазону 0-100
        return max(min(readability_score, 100), 0)

    def _calculate_header_score(self, headers: List[Dict[str, str]]) -> float:
        """Расчет оценки заголовков"""
        if not headers:
            return 0.0
            
        # Базовая оценка за наличие заголовков
        base_score = min(1.0, len(headers) / 10)  # Максимум за 10 заголовков
        
        # Преобразуем список заголовков в словарь по уровням
        header_levels = {}
        for header in headers:
            level = header['level']
            if level not in header_levels:
                header_levels[level] = []
            header_levels[level].append(header['text'])
        
        # Проверяем структуру заголовков
        hierarchy_score = 1.0
        levels = sorted(header_levels.keys())
        
        # Штраф за пропущенные уровни
        if levels and levels[0] != 1:
            hierarchy_score *= 0.8
        
        # Штраф за большие пропуски между уровнями
        for i in range(len(levels) - 1):
            if levels[i + 1] - levels[i] > 1:
                hierarchy_score *= 0.9
                
        # Учитываем распределение заголовков
        distribution_score = 1.0
        if len(headers) > 1:
            avg_headers_per_level = len(headers) / len(levels)
            for level_headers in header_levels.values():
                if len(level_headers) > avg_headers_per_level * 2:
                    distribution_score *= 0.9
                    
        return (base_score + hierarchy_score + distribution_score) / 3
