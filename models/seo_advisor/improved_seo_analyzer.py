
from typing import List, Dict, NamedTuple
from dataclasses import dataclass
import re

@dataclass
class TextAnalysis:
    """Структура для хранения результатов анализа"""
    headers: List[Dict[str, str]]
    lists: List[List[str]]
    images: List[Dict[str, str]]
    word_count: int
    char_count: int
    avg_word_length: float
    avg_sentence_length: float
    sentences: List[str]
    keywords_analysis: Dict[str, Dict]
    keyword_recommendations: List[str]
    links: List[str]
    readability_score: float
    content_structure_score: float

class HeaderAnalyzer:
    def analyze_headers(self, content: str) -> List[Dict[str, str]]:
        """Улучшенный анализ заголовков"""
        headers = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Проверяем следующую строку
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
            
            # Заголовок с подчеркиванием
            if next_line and set(next_line) in [{'-'}, {'='}]:
                level = 1 if '=' in next_line else 2
                headers.append({
                    'text': line,
                    'level': level
                })
                i += 2
                continue
            
            # Markdown заголовки
            if line.startswith('#'):
                level = len(re.match(r'^#+', line).group())
                text = line.lstrip('#').strip()
                if level <= 6 and text:
                    headers.append({
                        'text': text,
                        'level': level
                    })
            
            i += 1
        
        return headers

class ListAnalyzer:
    def analyze_lists(self, content: str) -> List[List[str]]:
        """Улучшенный анализ списков"""
        lists = []
        current_list = []
        
        list_patterns = {
            'unordered': re.compile(r'^\s*[-*•]\s+(.+)$'),
            'ordered': re.compile(r'^\s*\d+\.\s+(.+)$'),
            'separator': re.compile(r'^[-=*]+$')
        }
        
        for line in content.split('\n'):
            line = line.strip()
            
            if not line or list_patterns['separator'].match(line):
                if current_list:
                    lists.append(current_list)
                    current_list = []
                continue
            
            for pattern in [list_patterns['unordered'], list_patterns['ordered']]:
                match = pattern.match(line)
                if match:
                    item_text = match.group(1).strip()
                    if item_text:
                        current_list.append(item_text)
                    break
        
        if current_list:
            lists.append(current_list)
        
        return lists

class ImageAnalyzer:
    def analyze_images(self, content: str) -> List[Dict[str, str]]:
        """Улучшенный анализ изображений"""
        images = []
        
        image_patterns = {
            'markdown_custom': re.compile(r'\[Изображение:\s*([^\]]+)\]'),
            'markdown': re.compile(r'!\[([^\]]*)\]\(([^)]+)\)'),
            'html': re.compile(r'<img[^>]+(?:src=["\'](.*?)["\'][^>]*alt=["\'](.*?)["\']|alt=["\'](.*?)["\'][^>]*src=["\'](.*?)["\']+)[^>]*>')
        }
        
        for match in image_patterns['markdown_custom'].finditer(content):
            alt_text = match.group(1).strip()
            if alt_text:
                images.append({
                    'alt': alt_text,
                    'src': '',
                    'type': 'custom'
                })
        
        return images

class KeywordAnalyzer:
    def __init__(self):
        self.word_forms = {
            'samsung': ['samsung', 'самсунг', 'самсунга', 'самсунгу'],
            'galaxy': ['galaxy', 'галакси'],
            'камера': ['камера', 'камеры', 'камерой', 'камер', 'фотокамера'],
            'батарея': ['батарея', 'батареи', 'батарей', 'батарею', 'аккумулятор'],
            'характеристики': ['характеристики', 'характеристик', 'спецификации', 'параметры'],
            'производительность': ['производительность', 'производительности', 'скорость', 'быстродействие']
        }
        
        self.context_window = 5  # Размер окна для поиска связанных слов
    
    def analyze_keywords(self, content: str, target_keywords: List[str]) -> Dict[str, Dict]:
        """Улучшенный анализ ключевых слов с учетом контекста"""
        content_lower = content.lower()
        words = content_lower.split()
        result = {}
        
        for keyword in target_keywords:
            keyword_parts = keyword.lower().split()
            stats = {
                'exact_matches': content_lower.count(keyword.lower()),
                'partial_matches': 0,
                'contextual_matches': 0,
                'variations': [],
                'density': 0.0,
                'positions': [],
                'context_examples': []
            }
            
            # Поиск вариаций и частичных совпадений
            for base_word, forms in self.word_forms.items():
                if base_word in keyword_parts:
                    for form in forms:
                        positions = self._find_all_positions(content_lower, form)
                        count = len(positions)
                        if count > 0:
                            stats['variations'].append({
                                'form': form,
                                'count': count,
                                'positions': positions
                            })
                            stats['partial_matches'] += count
                            stats['positions'].extend(positions)
            
            # Анализ контекстной близости слов
            if len(keyword_parts) > 1:
                stats['contextual_matches'] = self._analyze_context(
                    words, 
                    keyword_parts,
                    self.word_forms
                )
                
                # Добавляем примеры контекста
                context_examples = self._find_context_examples(
                    content_lower,
                    keyword_parts,
                    self.word_forms
                )
                stats['context_examples'] = context_examples[:3]  # Топ-3 примера
            
            # Расчет улучшенной плотности
            words_total = len(words)
            if words_total > 0:
                # Учитываем контекстные совпадения с меньшим весом
                weighted_matches = (
                    stats['exact_matches'] * 1.0 +
                    stats['contextual_matches'] * 0.8 +
                    stats['partial_matches'] * 0.6
                )
                stats['density'] = weighted_matches / words_total
            
            result[keyword] = stats
        
        return result
    
    def _find_all_positions(self, text: str, word: str) -> List[int]:
        """Поиск всех позиций слова в тексте"""
        positions = []
        start = 0
        while True:
            pos = text.find(word, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions
    
    def _analyze_context(self, words: List[str], keyword_parts: List[str], word_forms: Dict) -> int:
        """Анализ контекстной близости слов"""
        matches = 0
        window = self.context_window
        
        for i in range(len(words)):
            found_parts = set()
            for j in range(max(0, i - window), min(len(words), i + window + 1)):
                for part in keyword_parts:
                    if part not in found_parts:
                        # Проверяем точное совпадение
                        if words[j] == part:
                            found_parts.add(part)
                        else:
                            # Проверяем вариации
                            for forms in word_forms.values():
                                if words[j] in forms and part in word_forms:
                                    found_parts.add(part)
            
            if len(found_parts) == len(keyword_parts):
                matches += 1
        
        return matches
    
    def _find_context_examples(self, text: str, keyword_parts: List[str], word_forms: Dict) -> List[str]:
        """Поиск примеров использования ключевых слов в контексте"""
        examples = []
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        for sentence in sentences:
            parts_found = set()
            for part in keyword_parts:
                if part in sentence:
                    parts_found.add(part)
                else:
                    # Проверяем вариации
                    for forms in word_forms.values():
                        if any(form in sentence for form in forms) and part in word_forms:
                            parts_found.add(part)
            
            if len(parts_found) == len(keyword_parts):
                examples.append(sentence)
        
        return examples
    
    def get_keyword_recommendations(self, analysis: Dict[str, Dict]) -> List[str]:
        """Улучшенные рекомендации по ключевым словам"""
        recommendations = []
        
        for keyword, stats in analysis.items():
            density = stats['density']
            parts = keyword.split()
            
            if stats['exact_matches'] == 0:
                if stats['contextual_matches'] > 0:
                    recommendations.append(
                        f"Используйте точную фразу '{keyword}' вместо раздельных слов"
                    )
                else:
                    recommendations.append(
                        f"Добавьте ключевую фразу '{keyword}' в текст"
                    )
            
            if density < 0.01:
                if len(parts) > 1:
                    recommendations.append(
                        f"Увеличьте частоту совместного использования слов '{' и '.join(parts)}'"
                    )
                else:
                    recommendations.append(
                        f"Увеличьте частоту использования слова '{keyword}'"
                    )
            
            if density > 0.04:
                recommendations.append(
                    f"Уменьшите частоту использования '{keyword}', попробуйте использовать синонимы"
                )
        
        return recommendations

class EnhancedTextAnalyzer:
    def __init__(self):
        self.header_analyzer = HeaderAnalyzer()
        self.list_analyzer = ListAnalyzer()
        self.image_analyzer = ImageAnalyzer()
        self.keyword_analyzer = KeywordAnalyzer()
    
    def analyze_text(self, content: str, target_keywords: List[str]) -> TextAnalysis:
        # Анализ структурных элементов
        headers = self.header_analyzer.analyze_headers(content)
        lists = self.list_analyzer.analyze_lists(content)
        images = self.image_analyzer.analyze_images(content)
        
        # Анализ ключевых слов
        keywords_analysis = self.keyword_analyzer.analyze_keywords(content, target_keywords)
        keyword_recommendations = self.keyword_analyzer.get_keyword_recommendations(keywords_analysis)
        
        # Анализ текста
        words = [w for w in re.findall(r'\b[\w\'-]+\b', content.lower()) if not w.isnumeric()]
        word_count = len(words)
        char_count = sum(len(word) for word in words)
        
        # Анализ предложений
        sentences = [s.strip() for s in re.split(r'[.!?]+(?=\s|[A-ZА-Я]|$)', content) if s.strip()]
        avg_sentence_length = word_count / len(sentences) if sentences else 0
        
        # Поиск ссылок
        links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        
        # Расчет дополнительных метрик
        readability_score = self._calculate_readability_score(sentences, words)
        content_structure_score = self._evaluate_content_structure(headers, lists, images, links, word_count)
        
        return TextAnalysis(
            headers=headers,
            lists=lists,
            images=images,
            word_count=word_count,
            char_count=char_count,
            avg_word_length=char_count / word_count if word_count > 0 else 0,
            avg_sentence_length=avg_sentence_length,
            sentences=sentences,
            keywords_analysis=keywords_analysis,
            keyword_recommendations=keyword_recommendations,
            links=links,
            readability_score=readability_score,
            content_structure_score=content_structure_score
        )
    
    def _calculate_readability_score(self, sentences: List[str], words: List[str]) -> float:
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        complex_words = sum(1 for w in words if len(re.findall(r'[аеёиоуыэюяАЕЁИОУЫЭЮЯ]', w)) > 4)
        complex_words_ratio = complex_words / len(words) if words else 0
        
        readability = 0.78 * (206.835 - 1.3 * avg_sentence_length - 60.1 * complex_words_ratio) / 100
        return max(0.0, min(1.0, readability))
    
    def _evaluate_content_structure(self, headers, lists, images, links, word_count) -> float:
        scores = []
        
        if word_count > 0:
            headers_score = min(1.0, len(headers) / (word_count / 300))
            scores.append(headers_score * 0.3)
        
        if lists:
            lists_score = min(1.0, len(lists) / 5)
            scores.append(lists_score * 0.2)
        
        if word_count > 0:
            images_score = min(1.0, len(images) / (word_count / 500))
            scores.append(images_score * 0.2)
            
            links_score = min(1.0, len(links) / (word_count / 400))
            scores.append(links_score * 0.1)
        
        return sum(scores) if scores else 0.0
