# Путь к файлу
file_path = "seo_ai_models/parsers/unified/extractors/schema_optimizer.py"

# Чтение файла
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Найдем строку с заглушкой
line_number = 1146
lines = content.split('\n')

if len(lines) >= line_number and "pass" in lines[line_number-1]:
    # Заменим заглушку на нормальный код обработки ошибки
    lines[line_number-1] = """                # Оставляем как есть, если не удалось преобразовать
                self.logger.warning(f"Не удалось преобразовать дату '{value}' для свойства {prop}")
                
                # Попытка извлечь дату с использованием регулярных выражений
                try:
                    # YYYY-MM-DD или YYYY/MM/DD
                    if re.search(r'\\d{4}[/-]\\d{1,2}[/-]\\d{1,2}', value):
                        parts = re.split(r'[/-]', value)
                        if len(parts) == 3 and len(parts[0]) == 4:  # год идет первым
                            year, month, day = parts
                            formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            result[prop] = formatted_date
                            return
                    
                    # DD.MM.YYYY или MM/DD/YYYY
                    elif re.search(r'\\d{1,2}[/.]\\d{1,2}[/.]\\d{4}', value):
                        parts = re.split(r'[/.]', value)
                        if len(parts) == 3 and len(parts[2]) == 4:  # год идет последним
                            if '.' in value:  # вероятно DD.MM.YYYY
                                day, month, year = parts
                            else:  # вероятно MM/DD/YYYY
                                month, day, year = parts
                            formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            result[prop] = formatted_date
                            return
                
                    # Если не удалось разобрать дату, оставляем оригинальное значение
                    result[prop] = value
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке даты: {str(e)}")
                    result[prop] = value"""

    # Убедимся, что импорт re присутствует
    import_lines = '\n'.join(lines[:50])
    if "import re" not in import_lines:
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                lines.insert(i+1, "import re")
                break

    # Записываем изменения обратно в файл
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Заглушка в строке {line_number} файла {file_path} успешно заменена")
else:
    print(f"Заглушка в строке {line_number} не найдена или уже исправлена")
