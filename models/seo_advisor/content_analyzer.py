
from typing import Dict, List
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import torch
import re

@dataclass
class ContentMetrics:
    keyword_density: float
    readability_score: float
    header_structure: Dict
    meta_tags_score: float

class ContentAnalyzer:
    def __init__(self, model_name: str = "DeepPavlov/rubert-base-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def analyze_text(self, content: str, target_keywords: List[str]) -> ContentMetrics:
        # Подсчет плотности ключевых слов
        keyword_density = self._calculate_keyword_density(content, target_keywords)
        
        # Оценка читаемости
        readability = self._calculate_readability(content)
        
        # Анализ заголовков
        headers = self._analyze_headers(content)
        
        # Проверка мета-тегов
        meta_score = self._check_meta_tags(content)
        
        return ContentMetrics(
            keyword_density=keyword_density,
            readability_score=readability,
            header_structure=headers,
            meta_tags_score=meta_score
        )
    
    def _calculate_keyword_density(self, text: str, keywords: List[str]) -> float:
        # Приводим текст к нижнему регистру
        text = text.lower()
        
        # Считаем общее количество слов
        total_words = len(text.split())
        
        # Считаем вхождения каждого ключевого слова
        keyword_count = 0
        for keyword in keywords:
            keyword_count += len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text))
            
        # Вычисляем плотность
        return keyword_count / total_words if total_words > 0 else 0
    
    def _calculate_readability(self, text: str) -> float:
        # Простая формула читаемости
        sentences = text.split('.')
        words = text.split()
        
        if not sentences or not words:
            return 0
            
        avg_sentence_length = len(words) / len(sentences)
        return max(0, min(1, 1 - (avg_sentence_length - 15) / 10))
        
    def _analyze_headers(self, content: str) -> Dict:
        headers = {
            'h1': len(re.findall(r'<h1.*?>(.*?)</h1>', content)),
            'h2': len(re.findall(r'<h2.*?>(.*?)</h2>', content)),
            'h3': len(re.findall(r'<h3.*?>(.*?)</h3>', content))
        }
        return headers
        
    def _check_meta_tags(self, content: str) -> float:
        score = 0
        if re.search(r'<title.*?>(.*?)</title>', content):
            score += 0.5
        if re.search(r'<meta\s+name="description".*?>', content):
            score += 0.5
        return score
