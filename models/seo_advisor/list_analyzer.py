
from typing import List, Dict
import re

class ListAnalyzer:
    def analyze_lists(self, content: str) -> List[List[str]]:
        """Улучшенный анализ списков"""
        lists = []
        current_list = []
        
        lines = content.split('\n')
        in_list = False
        
        list_markers = {
            'unordered': ['-', '*', '•'],
            'ordered': re.compile(r'^\d+\.')
        }
        
        for line in lines:
            line = line.strip()
            
            # Пропускаем пустые строки и разделители
            if not line or set(line) in [{'-'}, {'='}, {'*'}]:
                if current_list:
                    lists.append(current_list)
                    current_list = []
                in_list = False
                continue
            
            # Проверяем маркеры списка
            is_list_item = False
            
            # Проверка неупорядоченных списков
            for marker in list_markers['unordered']:
                if line.startswith(marker + ' '):
                    item_text = line[len(marker)+1:].strip()
                    if item_text:  # Проверяем, что есть текст после маркера
                        current_list.append(item_text)
                        in_list = True
                        is_list_item = True
                        break
            
            # Проверка упорядоченных списков
            if not is_list_item and list_markers['ordered'].match(line):
                item_text = re.sub(r'^\d+\.\s*', '', line).strip()
                if item_text:  # Проверяем, что есть текст после номера
                    current_list.append(item_text)
                    in_list = True
                    is_list_item = True
            
            # Если строка не является элементом списка
            if not is_list_item and in_list:
                if current_list:
                    lists.append(current_list)
                    current_list = []
                in_list = False
        
        # Добавляем последний список, если он есть
        if current_list:
            lists.append(current_list)
        
        return lists
