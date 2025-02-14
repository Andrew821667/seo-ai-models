from pymystem3 import Mystem
import razdel
from typing import Dict, List, Set, Tuple, Optional
import re

class KeywordProcessor:
    def __init__(self, min_weight: float = 15.0):
        self.min_weight = min_weight
        self.mystem = Mystem()
        self.stop_words = self._load_stop_words()
        
        # Новые весовые коэффициенты
        self.weights = {
            'adj_noun': 40.0,  # прилагательное + существительное
            'noun_noun': 35.0,  # существительное + существительное
            'single_noun': 30.0,  # одиночное существительное
            
            # Позиционные множители
            'position_multipliers': {
                'title': 2.0,      # заголовок
                'first_para': 1.5,  # первый абзац
                'last_para': 1.2    # последний абзац
            }
        }

    def _load_stop_words(self) -> Set[str]:
        return {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все',
            'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по',
            'только', 'ее', 'мне', 'быть', 'для', 'от', 'из', 'о', 'один', 'этот', 'при',
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

    def _analyze_word_pair(self, word1: str, word2: str) -> Tuple[bool, str, Tuple[str, str]]:
        """Расширенный анализ пары слов"""
        lemma1, pos1, gram1 = self._get_word_info(word1)
        lemma2, pos2, gram2 = self._get_word_info(word2)
        
        if not (pos1 and pos2):
            return False, '', ('', '')
            
        # Проверка на прилагательное + существительное
        if pos1.startswith('A') and pos2.startswith('S'):
            return True, 'adj_noun', (lemma1, lemma2)
            
        # Проверка на существительное + существительное
        if pos1.startswith('S') and pos2.startswith('S'):
            return True, 'noun_noun', (lemma1, lemma2)
            
        return False, '', ('', '')

    def _get_position_multiplier(self, text: str, position: int) -> float:
        """Определение позиционного множителя"""
        paragraphs = [p for p in text.split('\n\n') if p.strip()]  # Игнорируем пустые параграфы
        if not paragraphs:
            return 1.0
            
        # Получаем все слова до текущей позиции
        words_before_position = ' '.join(paragraphs).split()[:position]
        
        # Слова первого параграфа
        first_para_words = paragraphs[0].split()
        
        # Проверяем, находится ли слово в заголовке (первая строка)
        if position < len(first_para_words[0].split('\n')[0].split()):
            return self.weights['position_multipliers']['title']
            
        # В первом параграфе
        if position < len(first_para_words):
            return self.weights['position_multipliers']['first_para']
            
        # В последнем параграфе
        last_para_start = sum(len(p.split()) for p in paragraphs[:-1])
        if position >= last_para_start:
            return self.weights['position_multipliers']['last_para']
            
        return 1.0

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
        
        # Обработка пар слов
        for i in range(len(tokens) - 1):
            word1, word2 = tokens[i], tokens[i + 1]
            is_valid, pair_type, (lemma1, lemma2) = self._analyze_word_pair(word1, word2)
            
            if is_valid and not any(lemma in self.stop_words for lemma in [lemma1, lemma2]):
                phrase = f"{lemma1} {lemma2}"
                base_weight = self.weights[pair_type]
                position_mult = self._get_position_multiplier(text, i)
                keywords[phrase] = base_weight * position_mult
                processed_words.update([lemma1, lemma2])

        # Обработка отдельных существительных
        for i, token in enumerate(tokens):
            lemma, pos, _ = self._get_word_info(token)
            if (pos.startswith('S') and 
                lemma not in processed_words and 
                lemma not in self.stop_words and 
                len(lemma) > 2):
                position_mult = self._get_position_multiplier(text, i)
                keywords[lemma] = self.weights['single_noun'] * position_mult

        return dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True))

    def debug_analyze(self, text: str) -> None:
        """Отладочная функция"""
        words = text.split()
        if len(words) == 2:
            is_valid, pair_type, (lemma1, lemma2) = self._analyze_word_pair(words[0], words[1])
            print(f"Анализ пары: {words[0]} {words[1]}")
            print(f"Первое слово: {self._get_word_info(words[0])}")
            print(f"Второе слово: {self._get_word_info(words[1])}")
            print(f"Тип пары: {pair_type}")
            if is_valid:
                print(f"Леммы: {lemma1} {lemma2}")
