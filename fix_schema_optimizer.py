import re

# Путь к файлу
file_path = "seo_ai_models/parsers/unified/extractors/schema_optimizer.py"

# Чтение файла
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Найдем и заменим заглушку
pattern = r"try:\s+parsed_date = dateutil\.parser\.parse\(value\)\s+result\[prop\] = parsed_date\.strftime\('%Y-%m-%d'\)\s+except Exception:\s+# Оставляем как есть, если не удалось преобразовать\s+pass"

# Новая реализация
replacement = """try:
                parsed_date = dateutil.parser.parse(value)
                result[prop] = parsed_date.strftime('%Y-%m-%d')
            except Exception as e:
                # Оставляем как есть, если не удалось преобразовать
                self.logger.warning(f"Не удалось преобразовать дату '{value}' для свойства {prop}: {str(e)}")
                
                # Попытка извлечь дату с использованием регулярных выражений
                # Форматы: YYYY-MM-DD, DD.MM.YYYY, MM/DD/YYYY, и т.д.
                date_patterns = [
                    r'(\\d{4})[/-](\\d{1,2})[/-](\\d{1,2})',  # YYYY-MM-DD или YYYY/MM/DD
                    r'(\\d{1,2})[/.](\\d{1,2})[/.](\\d{4})',  # DD.MM.YYYY или DD/MM/YYYY
                    r'(\\d{4})(\\d{2})(\\d{2})'               # YYYYMMDD
                ]
                
                for pattern in date_patterns:
                    match = re.search(pattern, value)
                    if match:
                        groups = match.groups()
                        if len(groups) == 3:
                            if len(groups[0]) == 4:  # Первая группа - год (YYYY-MM-DD)
                                year, month, day = groups
                            elif len(groups[2]) == 4:  # Последняя группа - год (DD.MM.YYYY)
                                day, month, year = groups
                            else:
                                continue
                            
                            try:
                                # Проверяем, что значения в допустимом диапазоне
                                year_int = int(year)
                                month_int = int(month)
                                day_int = int(day)
                                
                                if 1 <= month_int <= 12 and 1 <= day_int <= 31 and 1900 <= year_int <= 2100:
                                    formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                                    result[prop] = formatted_date
                                    self.logger.info(f"Дата преобразована альтернативным способом: {value} -> {formatted_date}")
                                    return
                            except ValueError:
                                continue
                
                # Если не удалось разобрать дату, оставляем оригинальное значение
                self.logger.info(f"Оставляем оригинальное значение даты: {value}")
                result[prop] = value"""

# Выполним замену
content = re.sub(pattern, replacement, content)

# Если в файле нет импорта re, добавим его
if "import re" not in content:
    import_section = content.find("import")
    import_section_end = content.find("\n\n", import_section)
    content = content[:import_section_end] + "\nimport re" + content[import_section_end:]

# Записываем обратно в файл
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Файл {file_path} успешно обновлен")
