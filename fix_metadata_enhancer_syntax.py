# Путь к файлу
file_path = "seo_ai_models/parsers/unified/extractors/metadata_enhancer.py"

# Чтение файла
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Находим и исправляем нетерминированный строковый литерал на строке 365
error_line = 'gost += f"— (дата обращения: {accessed_date}). '

if error_line in content:
    # Исправляем - добавляем закрывающую кавычку
    fixed_line = 'gost += f"— (дата обращения: {accessed_date}). "'
    content = content.replace(error_line, fixed_line)
    
    # Записываем исправленный контент
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Синтаксическая ошибка в {file_path} исправлена")
else:
    print(f"Строка с ошибкой не найдена в {file_path}")
