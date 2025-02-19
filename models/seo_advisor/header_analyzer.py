
from typing import List, Dict
import re

class HeaderAnalyzer:
    def analyze_headers(self, content: str) -> List[Dict[str, str]]:
        """Улучшенный анализ заголовков"""
        headers = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Проверяем следующую строку на наличие подчеркивания
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
            
            # Основной заголовок (с подчеркиванием =)
            if next_line and set(next_line) == {'='}:
                headers.append({
                    'text': line,
                    'level': 1
                })
                continue
                
            # Подзаголовок (с подчеркиванием -)
            if next_line and set(next_line) == {'-'}:
                headers.append({
                    'text': line,
                    'level': 2
                })
                continue
                
            # Markdown заголовки
            if line.startswith('#'):
                level = len(re.match(r'^#+', line).group())
                text = line.lstrip('#').strip()
                if level <= 6 and text:  # Проверяем валидность уровня и наличие текста
                    headers.append({
                        'text': text,
                        'level': level
                    })
                continue
            
            # Заголовки с двоеточием в конце (если это не элемент списка)
            if line.endswith(':') and not line.startswith(('-', '*', '•', '1.')):
                if len(line) <= 100:  # Проверяем длину, чтобы исключить длинные предложения
                    headers.append({
                        'text': line.rstrip(':'),
                        'level': 2
                    })
