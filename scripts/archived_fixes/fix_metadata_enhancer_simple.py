# Путь к файлу
file_path = "seo_ai_models/parsers/unified/extractors/metadata_enhancer.py"

# Чтение файла
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Найдем строку с заглушкой
line_number = 348
lines = content.split('\n')

if len(lines) >= line_number and "pass" in lines[line_number-1]:
    # Заменим заглушку на нормальный код обработки ошибки
    lines[line_number-1] = """                self.logger.warning(f"Не удалось извлечь год из даты: {published_date}")
                # Проверка на другие форматы даты
                try:
                    # Проверка на формат DD.MM.YYYY или MM/DD/YYYY
                    date_match = re.search(r'\\b\\d{1,2}[/.]\\d{1,2}[/.]\\d{4}\\b', published_date)
                    if date_match:
                        date_str = date_match.group(0)
                        year = date_str.split('/')[-1].split('.')[-1]
                        gost += f"— {year}. "
                    else:
                        # Если ничего не найдено, добавляем информацию о доступе
                        accessed_date = datetime.now().strftime("%d.%m.%Y")
                        gost += f"— (дата обращения: {accessed_date}). "
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке даты: {str(e)}")
                    # Если все не удалось, указываем только дату обращения
                    accessed_date = datetime.now().strftime("%d.%m.%Y")
                    gost += f"— (дата обращения: {accessed_date}). """

    # Добавляем импорт datetime, если его нет
    import_lines = '\n'.join(lines[:50])
    if "from datetime import datetime" not in import_lines:
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                lines.insert(i+1, "from datetime import datetime")
                break

    # Записываем изменения обратно в файл
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Заглушка в строке {line_number} файла {file_path} успешно заменена")
else:
    print(f"Заглушка в строке {line_number} не найдена или уже исправлена")
