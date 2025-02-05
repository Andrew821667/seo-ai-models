import numpy as np
from typing import Dict, List, Union, Optional
import logging
from bs4 import BeautifulSoup
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class SEOFeaturesExtractor:
    """Класс для извлечения SEO-характеристик из текста"""
    
    def __init__(
        self,
        max_features: int = 100,
        min_keyword_length: int = 3,
        language: str = 'english'
    ):
        self.max_features = max_features
        self.min_keyword_length = min_keyword_length
        self.language = language
        
        # Инициализация TF-IDF
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.95
        )
        
        # Загрузка стоп-слов
        self.stop_words = set(stopwords.words(language))
        
    def extract_html_features(self, html: str) -> Dict[str, Union[float, str, List[str]]]:
        """Извлечение характеристик из HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            features = {
                # Meta-теги
                'has_title': int(bool(soup.title)),
                'title_length': len(soup.title.text) if soup.title else 0,
                'has_meta_description': int(bool(soup.find('meta', {'name': 'description'}))),
                'has_meta_keywords': int(bool(soup.find('meta', {'name': 'keywords'}))),
                
                # Заголовки
                'h1_count': len(soup.find_all('h1')),
                'h2_count': len(soup.find_all('h2')),
                'h3_count': len(soup.find_all('h3')),
                
                # Ссылки
                'internal_links': len([a for a in soup.find_all('a', href=True) 
                                    if not a['href'].startswith(('http', 'https', '//', 'www'))]),
                'external_links': len([a for a in soup.find_all('a', href=True) 
                                    if a['href'].startswith(('http', 'https', '//', 'www'))]),
                
                # Изображения
                'img_count': len(soup.find_all('img')),
                'img_alt_ratio': self._calculate_img_alt_ratio(soup),
                
                # Текстовые характеристики
                'text_length': len(soup.get_text())
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting HTML features: {e}")
            return {}
            
    def extract_text_features(self, text: str) -> Dict[str, Union[float, List[str]]]:
        """Извлечение характеристик из текста"""
        try:
            # Токенизация
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalnum() and len(w) >= self.min_keyword_length
                    and w not in self.stop_words]
            
            # Базовые метрики
            features = {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
                
                # Лексическое разнообразие
                'vocabulary_richness': len(set(words)) / len(words) if words else 0,
                
                # Частоты слов
                'word_frequencies': self._calculate_word_frequencies(words),
                
                # Ключевые слова
                'keywords': self._extract_keywords(text)
            }
            
            # Добавляем TF-IDF характеристики
            tfidf_features = self._calculate_tfidf_features([text])
            features.update(tfidf_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return {}
    
    def _calculate_img_alt_ratio(self, soup: BeautifulSoup) -> float:
        """Расчет доли изображений с alt-текстом"""
        images = soup.find_all('img')
        if not images:
            return 0.0
        images_with_alt = len([img for img in images if img.get('alt')])
        return images_with_alt / len(images)
    
    def _calculate_word_frequencies(self, words: List[str]) -> Dict[str, float]:
        """Расчет частот слов"""
        word_counts = Counter(words)
        total_words = len(words)
        return {word: count/total_words for word, count in word_counts.most_common(self.max_features)}
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Извлечение ключевых слов"""
        # Используем TF-IDF для определения ключевых слов
        tfidf_matrix = self.tfidf.fit_transform([text])
        feature_names = self.tfidf.get_feature_names_out()
        
        # Получаем важность слов
        importance = np.squeeze(tfidf_matrix.toarray())
        top_indices = importance.argsort()[-top_k:][::-1]
        
        return [feature_names[i] for i in top_indices]
    
    def _calculate_tfidf_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Расчет TF-IDF характеристик"""
        tfidf_matrix = self.tfidf.fit_transform(texts)
        return {
            'tfidf_features': tfidf_matrix.toarray(),
            'feature_names': self.tfidf.get_feature_names_out().tolist()
        }
    
    def extract_all_features(
        self,
        text: str,
        html: Optional[str] = None
    ) -> Dict[str, Union[float, List[str], Dict[str, float]]]:
        """Извлечение всех характеристик"""
        features = {}
        
        # Извлекаем текстовые характеристики
        text_features = self.extract_text_features(text)
        features.update(text_features)
        
        # Если предоставлен HTML, извлекаем HTML характеристики
        if html:
            html_features = self.extract_html_features(html)
            features.update(html_features)
        
        return features
    
    def batch_process(
        self,
        texts: List[str],
        html_texts: Optional[List[str]] = None
    ) -> List[Dict[str, Union[float, List[str], Dict[str, float]]]]:
        """Пакетная обработка текстов"""
        features_list = []
        
        for i, text in enumerate(texts):
            html = html_texts[i] if html_texts else None
            features = self.extract_all_features(text, html)
            features_list.append(features)
        
        return features_list
