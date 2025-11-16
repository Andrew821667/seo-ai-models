import re

# Путь к файлу
file_path = "seo_ai_models/parsers/unified/extractors/metadata_enhancer.py"

# Чтение файла
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Найдем и заменим заглушку
pattern = r"if published_date:\s+try:\s+year = re\.search\(r'\\d{4}', published_date\)\.group\(0\)\s+gost \+= f\"— {year}\. \"\s+except:\s+pass"

# Новая реализация
replacement = """if published_date:
            try:
                year = re.search(r'\\d{4}', published_date).group(0)
                gost += f"— {year}. "
            except:
                # Если не удалось найти 4 цифры, пробуем другие форматы
                self.logger.warning(f"Не удалось извлечь год из даты: {published_date}")
                
                # Проверка на наличие 2 цифр, которые могут быть годом (например, '22 для 2022)
                year_match = re.search(r'\\b\\d{2}\\b', published_date)
                if year_match:
                    year_short = year_match.group(0)
                    current_century = datetime.now().year // 100
                    year = f"{current_century}{year_short}"
                    self.logger.info(f"Извлечен короткий год: {year_short}, преобразован в: {year}")
                    gost += f"— {year}. "
                else:
                    # Если ничего не найдено, добавляем информацию о доступе
                    accessed_date = datetime.now().strftime("%d.%m.%Y")
                    gost += f"— (дата обращения: {accessed_date}). """

# Выполним замену
content = re.sub(pattern, replacement, content)

# Если в файле нет импорта datetime, добавим его
if "from datetime import datetime" not in content:
    import_section = content.find("import")
    import_section_end = content.find("\n\n", import_section)
    content = content[:import_section_end] + "\nfrom datetime import datetime" + content[import_section_end:]

# Записываем обратно в файл
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Файл {file_path} успешно обновлен")
