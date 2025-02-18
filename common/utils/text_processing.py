
import re
from typing import List, Dict
import razdel

class TextProcessor:
    def __init__(self):
        self.stop_words = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как',
            'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к',
            'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне'
        }

    def tokenize(self, text: str) -> List[str]:
        """Разбивает текст на токены"""
        return [token.text for token in razdel.tokenize(text)]

    def split_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения"""
        return [sentence.text for sentence in razdel.sentenize(text)]

    def normalize(self, text: str) -> str:
        """Нормализует текст"""
        text = text.lower()
        text = re.sub(r'[^\w\s-]', ' ', text)
        return ' '.join(text.split())

    def get_word_count(self, text: str) -> int:
        """Подсчитывает количество слов"""
        return len(self.tokenize(text))

    def get_sentence_count(self, text: str) -> int:
        """Подсчитывает количество предложений"""
        return len(self.split_sentences(text))

    def get_average_sentence_length(self, text: str) -> float:
        """Вычисляет среднюю длину предложения"""
        sentences = self.split_sentences(text)
        if not sentences:
            return 0
        return sum(len(self.tokenize(sent)) for sent in sentences) / len(sentences)

    def extract_headers(self, text: str) -> List[Dict[str, str]]:
        """Извлекает заголовки из текста"""
        headers = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Проверяем, является ли строка заголовком
            if line and (line.startswith('#') or 
                        re.match(r'^[0-9]+\.', line) or 
                        line.isupper() or
                        len(line) <= 100 and line.endswith(':') or
                        'FAQ' in line):
                headers.append({
                    'text': line,
                    'level': 1 if line.startswith('#') else 2
                })
        return headers
