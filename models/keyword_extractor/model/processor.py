from pymystem3 import Mystem
import razdel
from typing import Dict, List, Set, Tuple
import re

class KeywordProcessor:
    def __init__(self, min_weight: float = 15.0):
        self.min_weight = min_weight
        self.mystem = Mystem()
        self.stop_words = self._load_stop_words()
        
    def _load_stop_words(self) -> Set[str]:
        return {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как',
            'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к',
            'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне',
            'быть', 'для', 'от', 'из', 'о', 'один', 'этот', 'при',
            'значительно', 'помогать', 'улучшать', 'анализировать'
        }

    def _get_word_info(self, word: str) -> Tuple[str, str, str]:
        """Получение леммы и части речи"""
        analysis = self.mystem.analyze(word)
        if analysis and analysis[0].get('analysis'):
            info = analysis[0]['analysis'][0]
            gram_parts = info['gr'].split(',')
            return info['lex'], gram_parts[0], info['gr']
        return word, '', ''

    def _analyze_bigram(self, word1: str, word2: str) -> bool:
        """Анализ биграммы на прилагательное + существительное"""
        lemma1, pos1, gram1 = self._get_word_info(word1)
        lemma2, pos2, gram2 = self._get_word_info(word2)
        
        if not (pos1 and pos2):
            return False, ('', '')
            
        if pos1.startswith('A') and pos2.startswith('S'):
            return True, (lemma1, lemma2)
            
        return False, ('', '')

    def extract_keywords(self, text: str, min_weight: float = None) -> Dict[str, float]:
        if min_weight is None:
            min_weight = self.min_weight
            
        # Предварительная обработка текста
        text = text.lower()
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Разбиваем на токены, исключая пробелы
        tokens = [token.text for token in razdel.tokenize(text) if not token.text.isspace()]
        
        keywords = {}
        processed_words = set()
        
        # Обработка биграмм
        for i in range(len(tokens) - 1):
            word1, word2 = tokens[i], tokens[i + 1]
            is_valid_pair, (lemma1, lemma2) = self._analyze_bigram(word1, word2)
            
            if is_valid_pair and not any(lemma in self.stop_words for lemma in [lemma1, lemma2]):
                phrase = f"{lemma1} {lemma2}"
                keywords[phrase] = 35.0
                processed_words.update([lemma1, lemma2])
        
        # Обработка отдельных существительных
        for token in tokens:
            lemma, pos, _ = self._get_word_info(token)
            if (pos.startswith('S') and 
                lemma not in processed_words and 
                lemma not in self.stop_words and 
                len(lemma) > 2):
                keywords[lemma] = 25.0
        
        return dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True))

    def debug_analyze(self, text: str) -> None:
        """Отладочная функция"""
        words = text.split()
        if len(words) == 2:
            is_valid, (lemma1, lemma2) = self._analyze_bigram(words[0], words[1])
            print(f"Анализ пары: {words[0]} {words[1]}")
            print(f"Первое слово: {self._get_word_info(words[0])}")
            print(f"Второе слово: {self._get_word_info(words[1])}")
            print(f"Является парой прил.+сущ.: {is_valid}")
            if is_valid:
                print(f"Леммы: {lemma1} {lemma2}")
