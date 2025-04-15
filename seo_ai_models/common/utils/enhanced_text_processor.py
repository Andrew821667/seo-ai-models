
"""
Улучшенный TextProcessor для более качественной обработки HTML-контента.
Расширяет возможности стандартного TextProcessor добавляя специфическую
логику для работы с JavaScript-генерируемым контентом.
"""

from typing import Dict, List, Tuple, Set, Optional, Union
import re
import os
import string
from collections import Counter
from bs4 import BeautifulSoup

from seo_ai_models.common.utils.text_processing import TextProcessor

class EnhancedTextProcessor(TextProcessor):
    """
    Улучшенная версия TextProcessor с дополнительными возможностями
    для обработки HTML-контента и JavaScript-генерируемого текста.
    """
    
    def __init__(self, language=None):
        """Инициализация процессора текста."""
        super().__init__()
        self.language = language
    
    def process_html_content(self, html_content: str) -> Dict[str, any]:
        """
        Глубокий анализ HTML-контента с извлечением различных метрик.
        
        Args:
            html_content: HTML-контент для анализа
            
        Returns:
            Dict[str, any]: Извлеченные метрики и данные
        """
        if not html_content:
            
        # Извлекаем элементы из HTML
        import re
        elements = []
        
        # Простое извлечение заголовков
        headers = re.findall(r'<h[1-6][^>]*>(.*?)</h[1-6]>', html_content, re.DOTALL | re.IGNORECASE)
        for header in headers:
            elements.append({
                'type': 'header',
                'content': header.strip()
            })
        
        # Извлечение параграфов
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html_content, re.DOTALL | re.IGNORECASE)
        for paragraph in paragraphs:
            elements.append({
                'type': 'paragraph',
                'content': paragraph.strip()
            })
        
        # Извлечение ссылок
        links = re.findall(r'<a[^>]*href=['"]([^'"]*)['"][^>]*>(.*?)</a>', html_content, re.DOTALL | re.IGNORECASE)
        for url, text in links:
            elements.append({
                'type': 'link',
                'url': url,
                'content': text.strip()
            })
        
        # Извлечение изображений
        images = re.findall(r'<img[^>]*src=['"]([^'"]*)['"][^>]*>', html_content, re.IGNORECASE)
        for src in images:
            elements.append({
                'type': 'image',
                'src': src
            })
        
        result = {
            'text': processed_text,
            'paragraphs': paragraphs,
            'word_count': len(words),
            'sentence_count': len(sentences),
            'elements': elements  # Добавляем извлеченные элементы
        }
        
        return result
    
        
        # Используем BeautifulSoup для парсинга HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Удаляем скрипты и стили
        for script in soup(['script', 'style', 'noscript', 'iframe']):
            script.decompose()
        
        # Извлекаем основной текстовый контент
        paragraphs = []
        for p in soup.find_all(['p', 'div', 'span', 'article', 'section']):
            if p.name == 'div' and (p.find('p') or p.find('div')):
                # Пропускаем div-ы, содержащие другие блочные элементы
                continue
                
            text = p.get_text(strip=True)
            if text and len(text) > 20:  # Игнорируем короткие фрагменты
                paragraphs.append(text)
        
        # Извлекаем весь текст
        all_text = ' '.join(paragraphs)
        
        # Анализ заголовков
        headings = []
        for i in range(1, 7):
            for h in soup.find_all(f'h{i}'):
                text = h.get_text(strip=True)
                if text:
                    headings.append({
                        'level': i,
                        'text': text
                    })
        
        # Анализ ссылок
        links = []
        for a in soup.find_all('a', href=True):
            text = a.get_text(strip=True)
            href = a['href']
            if text and href:
                links.append({
                    'text': text,
                    'href': href
                })
        
        # Анализ изображений
        images = []
        for img in soup.find_all('img', src=True):
            alt = img.get('alt', '')
            src = img['src']
            if src:
                images.append({
                    'src': src,
                    'alt': alt
                })
        
        # Анализ списков
        lists = []
        for list_tag in soup.find_all(['ul', 'ol']):
            items = []
            for li in list_tag.find_all('li'):
                text = li.get_text(strip=True)
                if text:
                    items.append(text)
            
            if items:
                lists.append({
                    'type': list_tag.name,
                    'items': items
                })
        
        # Анализ таблиц
        tables = []
        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                cells = []
                for cell in tr.find_all(['td', 'th']):
                    text = cell.get_text(strip=True)
                    cells.append(text)
                
                if cells:
                    rows.append(cells)
            
            if rows:
                tables.append({
                    'rows': rows
                })
        
        # Расчет основных метрик
        words = self.tokenize(all_text)
        sentences = self.split_sentences(all_text)
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return {
            'text': all_text,
            'paragraphs': paragraphs,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'headings': headings,
            'links': links,
            'images': images,
            'lists': lists,
            'tables': tables
        }
    
    def calculate_enhanced_readability(self, text: str, language=None) -> Dict[str, float]:
        """
        Расчет улучшенной метрики читабельности с учетом специфики языка.
        
        Args:
            text: Текст для анализа
            language: Язык текста (если None, определяется автоматически)
            
        Returns:
            Dict[str, float]: Метрики читабельности
        """
        if not text or len(text) < 20:
            return {'flesch_reading_ease': 0, 'readability_score': 0}
            
        # Определяем язык, если не задан
        if language is None:
            language = self.detect_language(text)
            
        # Подсчет базовых компонентов
        words = self.tokenize(text)
        sentences = self.split_sentences(text)
        
        if not sentences or not words:
            return {'flesch_reading_ease': 0, 'readability_score': 0}
        
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Средняя длина предложения в словах
        avg_sentence_length = word_count / sentence_count
        
        # Подсчет слогов (упрощенно)
        syllable_count = 0
        for word in words:
            # Простая эвристика для подсчета слогов
            if language == 'ru':
                # Для русского языка считаем гласные
                vowels = 'аеёиоуыэюя'
                syllable_count += sum(1 for char in word.lower() if char in vowels)
            else:  # для английского и других языков
                # Для английского языка
                vowels = 'aeiouy'
                word = word.lower()
                if word.endswith('e'):
                    word = word[:-1]
                count = 0
                was_vowel = False
                for char in word:
                    is_vowel = char in vowels
                    if is_vowel and not was_vowel:
                        count += 1
                    was_vowel = is_vowel
                
                if count == 0:
                    count = 1
                syllable_count += count
        
        # Средняя длина слова в слогах
        avg_syllables_per_word = syllable_count / word_count if word_count > 0 else 0
        
        # Формула Flesch Reading Ease, адаптированная с учетом языка
        if language == 'ru':
            # Адаптированная формула для русского языка
            readability_score = 206.835 - (1.3 * avg_sentence_length) - (60.1 * avg_syllables_per_word)
        else:
            # Оригинальная формула для английского
            readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Нормализация оценки к диапазону 0-100
        normalized_score = max(min(readability_score, 100), 0)
        
        # Если средняя длина предложения очень большая, корректируем оценку
        if avg_sentence_length > 30:
            normalized_score *= 0.7
        elif avg_sentence_length > 25:
            normalized_score *= 0.8
        
        return {
            'flesch_reading_ease': normalized_score,
            'readability_score': normalized_score / 100,  # В диапазоне 0-1 для совместимости
            'avg_sentence_length': avg_sentence_length,
            'avg_syllables_per_word': avg_syllables_per_word
        }
    
    def extract_main_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """
        Извлекает основные темы из текста, используя группировку ключевых слов.
        
        Args:
            text: Текст для анализа
            max_topics: Максимальное количество тем
            
        Returns:
            List[str]: Список основных тем
        """
        if not text or len(text) < 100:
            return []
        
        # Извлекаем ключевые слова
        keywords = self.extract_keywords(text, max_keywords=20)
        
        # Извлекаем n-граммы для контекста
        bigrams = self.extract_ngrams(text, n=2, max_ngrams=15)
        
        # Простая группировка ключевых слов по контексту
        topics = []
        used_keywords = set()
        
        # Сначала добавляем темы из биграмм
        for bigram, _ in bigrams:
            words = bigram.split()
            if any(keyword in words for keyword, _ in keywords):
                topics.append(bigram)
                for word in words:
                    used_keywords.add(word)
                
                if len(topics) >= max_topics:
                    break
        
        # Дополняем отдельными ключевыми словами
        for keyword, _ in keywords:
            if keyword not in used_keywords and len(topics) < max_topics:
                topics.append(keyword)
                used_keywords.add(keyword)
        
        return topics[:max_topics]
    
    def calculate_keyword_prominence(self, text: str, keywords: List[str]) -> Dict[str, float]:
        """
        Рассчитывает "заметность" (prominence) ключевых слов в тексте.
        
        Учитывает:
        - Наличие в заголовках
        - Наличие в начале и конце текста
        - Частота использования
        
        Args:
            text: Текст для анализа
            keywords: Список ключевых слов
            
        Returns:
            Dict[str, float]: Оценка prominence для каждого ключевого слова
        """
        if not text or not keywords:
            return {keyword: 0.0 for keyword in keywords}
        
        # Нормализуем текст и ключевые слова
        normalized_text = self.normalize(text)
        normalized_keywords = [self.normalize(keyword) for keyword in keywords]
        
        # Разбиваем текст на параграфы
        paragraphs = text.split('\n\n')
        
        # Извлекаем заголовки из текста
        headings = []
        for line in text.split('\n'):
            if re.match(r'^#{1,6}\s+', line):
                headings.append(self.normalize(line))
        
        # Первый и последний параграфы
        first_paragraph = self.normalize(paragraphs[0]) if paragraphs else ""
        last_paragraph = self.normalize(paragraphs[-1]) if len(paragraphs) > 1 else ""
        
        # Подсчитываем "заметность" для каждого ключевого слова
        prominence_scores = {}
        
        for i, keyword in enumerate(normalized_keywords):
            score = 0.0
            
            # Базовый балл за наличие в тексте
            if keyword in normalized_text:
                score += 0.3
                
                # Бонус за частоту
                word_count = len(normalized_text.split())
                if word_count > 0:
                    frequency = normalized_text.count(keyword) / word_count
                    score += min(frequency * 100, 0.2)  # Максимум 0.2 за частоту
            
            # Бонус за наличие в заголовках
            for heading in headings:
                if keyword in heading:
                    score += 0.2
                    break
            
            # Бонус за наличие в первом абзаце
            if keyword in first_paragraph:
                score += 0.15
            
            # Бонус за наличие в последнем абзаце
            if keyword in last_paragraph:
                score += 0.15
            
            prominence_scores[keywords[i]] = min(score, 1.0)  # Максимум 1.0
        
        return prominence_scores
