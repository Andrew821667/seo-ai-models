import re

# Путь к файлу
file_path = "seo_ai_models/parsers/unified/extractors/metadata_enhancer.py"

# Чтение файла построчно
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Ищем проблемную строку и исправляем ее
for i, line in enumerate(lines):
    if "дата обращения" in line and not line.strip().endswith('"'):
        print(f"Найдена проблемная строка {i+1}: {line.strip()}")
        lines[i] = line.rstrip() + '"\n'  # Добавляем закрывающую кавычку
        print(f"Исправлено на: {lines[i].strip()}")

# Записываем исправленный файл
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Файл {file_path} успешно обновлен")
