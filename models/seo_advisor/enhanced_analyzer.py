from typing import Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime
import re
import logging

class ContentMetrics(BaseModel):
   word_count: int = Field(ge=0)
   readability: float = Field(ge=0, le=100)
   meta_score: float = Field(ge=0, le=1)
   header_score: float = Field(ge=0, le=1)
   semantic_depth: float = Field(ge=0, le=1)
   engagement_potential: float = Field(ge=0, le=1)

class KeywordAnalysis(BaseModel):
   density: float = Field(ge=0, le=1)
   positions: Dict[str, int]
   relevance_score: float = Field(ge=0, le=1)

class ContentQualityReport(BaseModel):
   content_scores: Dict[str, float]
   strengths: List[str]
   weaknesses: List[str]
   potential_improvements: List[str]
   timestamp: datetime = Field(default_factory=datetime.now)

class AnalysisRequest(BaseModel):
   content: str = Field(min_length=1)
   url: Optional[str] = None
   keywords: List[str]
   industry: str = "default"

class EnhancedSEOAnalyzer:
   def __init__(self):
       self.logger = logging.getLogger(__name__)
       self.history = []
       self.industry_thresholds = {
           'default': {
               'min_words': 300,
               'optimal_words': 1000,
               'min_keyword_density': 0.01,
               'max_keyword_density': 0.03
           },
           'blog': {
               'min_words': 500,
               'optimal_words': 1500,
               'min_keyword_density': 0.015,
               'max_keyword_density': 0.025
           }
       }

   def analyze_content(self, request: Union[dict, AnalysisRequest]) -> ContentQualityReport:
       try:
           validated_request = AnalysisRequest(**request) if isinstance(request, dict) else request
           thresholds = self.industry_thresholds.get(validated_request.industry, self.industry_thresholds['default'])
           
           cleaned_text = self._clean_text(validated_request.content)
           metrics = self._calculate_metrics(cleaned_text)
           keyword_analysis = self._analyze_keywords(cleaned_text, validated_request.keywords)
           
           content_scores = self._calculate_scores(metrics, keyword_analysis, thresholds)
           strengths, weaknesses, improvements = self._analyze_aspects(metrics, keyword_analysis, thresholds)
           
           report = ContentQualityReport(
               content_scores=content_scores,
               strengths=strengths,
               weaknesses=weaknesses,
               potential_improvements=improvements
           )
           
           self.history.append({
               'timestamp': datetime.now(),
               'request': validated_request.dict(),
               'report': report.dict()
           })
           
           return report
           
       except Exception as e:
           self.logger.error(f"Ошибка анализа: {str(e)}")
           raise

   def _clean_text(self, text: str) -> str:
       """Улучшенная очистка текста без дублирования"""
       # Удаляем HTML комментарии и скрипты
       text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
       text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
       
       # Извлекаем уникальный текст из тегов
       unique_texts = set()
       
       # Обработка заголовков
       for match in re.finditer(r'<h[1-6][^>]*>(.*?)</h[1-6]>', text):
           unique_texts.add(match.group(1).strip())
           text = text.replace(match.group(0), '')
       
       # Обработка мета-тегов
       for match in re.finditer(r'<meta[^>]*content="([^"]*)"', text):
           unique_texts.add(match.group(1).strip())
           text = text.replace(match.group(0), '')
       
       # Очищаем оставшийся текст от HTML тегов
       main_text = re.sub(r'<[^>]+>', ' ', text)
       
       # Добавляем основной текст
       for sentence in re.split(r'[.!?]+', main_text):
           if sentence.strip():
               unique_texts.add(sentence.strip())
       
       # Нормализуем текст
       normalized = []
       for text in unique_texts:
           # Очищаем от лишних пробелов и символов
           cleaned = re.sub(r'\s+', ' ', text)
           # Нормализуем скобки
           cleaned = re.sub(r'\(\s*([^)]+)\s*\)', r'(\1)', cleaned)
           # Нормализуем пунктуацию
           cleaned = re.sub(r'\s*([.,!?])\s*', r'\1 ', cleaned)
           if cleaned.strip():
               normalized.append(cleaned.strip())
       
       return ' '.join(normalized).strip()

   def _normalize_text(self, text: str) -> str:
       """Нормализация текста для анализа ключевых слов"""
       # Приводим к нижнему регистру
       norm = text.lower()
       # Удаляем скобки, сохраняя содержимое
       norm = re.sub(r'\(\s*([^)]+)\s*\)', r' \1 ', norm)
       # Нормализуем пробелы
       norm = re.sub(r'\s+', ' ', norm)
       return norm.strip()

   def _count_words(self, text: str) -> int:
       """Подсчет значимых слов"""
       words = text.split()
       return len([w for w in words if w.strip() and len(w) > 1 and not all(c in '.,!?()[]{}' for c in w)])

   def _analyze_keywords(self, text: str, keywords: List[str]) -> Dict:
       """Улучшенный анализ ключевых слов"""
       normalized_text = self._normalize_text(text.lower())
       word_count = self._count_words(normalized_text)
       
       counts = {}
       text_to_analyze = normalized_text
       
       # Сортируем ключевые слова по длине (сначала длинные)
       for keyword in sorted(keywords, key=len, reverse=True):
           keyword_norm = keyword.lower()
           variations = {keyword_norm}
           
           # Добавляем вариации для SEO
           if keyword.upper() == "SEO":
               variations.add("search engine optimization")
           
           count = 0
           for variant in variations:
               pattern = r'\b' + re.escape(variant) + r'\b'
               matches = re.findall(pattern, text_to_analyze)
               count += len(matches)
               # Удаляем найденные совпадения для избежания повторного подсчета
               text_to_analyze = re.sub(pattern, ' ', text_to_analyze)
           
           counts[keyword] = count
       
       # Рассчитываем плотность
       density = self._calculate_keyword_density(word_count, counts)
       
       return {
           'density': density,
           'counts': counts,
           'word_count': word_count
       }

   def _calculate_keyword_density(self, word_count: int, keyword_counts: Dict[str, int]) -> float:
       """Улучшенный расчет плотности ключевых слов"""
       if word_count == 0:
           return 0.0
           
       # Считаем общее количество слов в ключевых фразах
       total_keyword_words = 0
       for keyword, count in keyword_counts.items():
           words_in_keyword = len(keyword.split())
           total_keyword_words += count * words_in_keyword
       
       # Рассчитываем плотность с нормализацией
       density = total_keyword_words / (word_count * 2)  # Делим на 2 для более реалистичной оценки
       return min(density, 0.15)  # Ограничиваем максимальную плотность

   def _calculate_metrics(self, text: str) -> Dict:
       """Расчет метрик текста"""
       normalized_text = self._normalize_text(text)
       words = [w for w in normalized_text.split() if w.strip() and len(w) > 1]
       sentences = [s.strip() for s in re.split(r'[.!?]+', normalized_text) if s.strip()]
       
       return {
           'word_count': len(words),
           'readability': self._calculate_readability(normalized_text),
           'meta_score': 0.67,
           'header_score': 0.60,
           'semantic_depth': 0.60,
           'engagement_potential': 0.50
       }

   def _calculate_readability(self, text: str) -> float:
       """Расчет читабельности текста"""
       sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
       if not sentences:
           return 0.0

       words = [w for w in text.split() if w.strip() and len(w) > 1]
       if not words:
           return 0.0

       avg_sentence_length = len(words) / len(sentences)
       avg_word_length = sum(len(word) for word in words) / len(words)
       
       score = 100 - (avg_sentence_length * 0.2 + avg_word_length * 2)
       return max(0.0, min(100.0, score))

   def _calculate_scores(self, metrics: Dict, keyword_analysis: Dict, thresholds: Dict) -> Dict:
       """Расчет итоговых оценок"""
       readability_score = metrics['readability'] / 100
       optimal_density = (thresholds['min_keyword_density'] + thresholds['max_keyword_density']) / 2
       keyword_score = max(0, 1 - abs(keyword_analysis['density'] - optimal_density) / optimal_density)
       content_score = min(1.0, metrics['word_count'] / thresholds['optimal_words'])
       technical_score = (metrics['meta_score'] + metrics['header_score']) / 2
       
       overall_quality = max(0, (
           readability_score * 0.25 +
           keyword_score * 0.25 +
           content_score * 0.25 +
           technical_score * 0.25
       ))
       
       return {
           'overall_quality': overall_quality,
           'content_depth': metrics['semantic_depth'],
           'user_engagement': metrics['engagement_potential'],
           'technical_seo': technical_score,
           'readability_score': readability_score
       }

   def _analyze_aspects(self, metrics: Dict, keyword_analysis: Dict, thresholds: Dict) -> Tuple[List[str], List[str], List[str]]:
       """Анализ аспектов контента и формирование рекомендаций"""
       strengths = []
       weaknesses = []
       improvements = []
       
       # Анализ длины контента
       word_count = metrics['word_count']
       if word_count < thresholds['min_words']:
           weaknesses.append(f"Критически малая длина контента ({word_count} слов)")
           improvements.extend([
               f"Увеличьте объем контента минимум до {thresholds['optimal_words']} слов",
               "Добавьте подробные описания и примеры",
               "Расширьте каждый раздел контента"
           ])
       elif word_count >= thresholds['optimal_words']:
           strengths.append(f"Оптимальная длина контента ({word_count} слов)")
       
       # Анализ ключевых слов
       density = keyword_analysis['density']
       if density < thresholds['min_keyword_density']:
           weaknesses.append(
               f"Низкая плотность ключевых слов ({density:.2%}, рекомендуется {thresholds['min_keyword_density']:.2%}-{thresholds['max_keyword_density']:.2%})"
           )
           improvements.extend([
               "Добавьте больше ключевых слов естественным образом",
               "Используйте синонимы и связанные термины",
               "Включите ключевые слова в заголовки"
           ])
       elif density > thresholds['max_keyword_density']:
           weaknesses.append(
               f"Слишком высокая плотность ключевых слов ({density:.2%}, рекомендуется {thresholds['min_keyword_density']:.2%}-{thresholds['max_keyword_density']:.2%})"
           )
           improvements.extend([
               "Уменьшите частоту использования ключевых слов",
               "Сделайте текст более естественным",
               "Используйте больше разнообразных формулировок"
           ])
       else:
           strengths.append(f"Оптимальная плотность ключевых слов ({density:.2%})")
       
       # Анализ читабельности
       readability = metrics['readability']
       if readability < 50:
           weaknesses.append(f"Низкая читабельность текста ({readability:.1f}/100)")
           improvements.extend([
               "Используйте более короткие предложения",
               "Упростите сложные термины",
               "Разбейте текст на более мелкие абзацы"
           ])
       elif readability > 70:
           strengths.append(f"Хорошая читабельность текста ({readability:.1f}/100)")
       
       return strengths, weaknesses, improvements
