"""Утилиты для обработки текста."""

from typing import Dict, List


class TextProcessor:
    """Класс для обработки и анализа текста."""
    
    def tokenize(self, text: str) -> List[str]:
        """Разбивает текст на токены (слова)."""
        if not text:
            return []
        # Простая реализация для заглушки
        return text.split()
    
    def normalize(self, text: str) -> str:
        """Нормализует текст (приводит к нижнему регистру)."""
        if not text:
            return ""
        return text.lower()
    
    def split_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения."""
        if not text:
            return []
        # Простая реализация для заглушки
        import re
        return re.split(r'[.!?]+', text)
    
    def extract_headers(self, text: str) -> List[Dict[str, str]]:
        """Извлекает заголовки из текста (для Markdown)."""
        if not text:
            return []
        
        headers = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                # Подсчитываем количество символов #
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break
                
                header_text = line[level:].strip()
                if header_text:
                    headers.append({
                        'level': level,
                        'text': header_text
                    })
        
        return headers
