import re

# Путь к файлу
file_path = "seo_ai_models/parsers/unified/extractors/schema_optimizer.py"

# Чтение файла построчно
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Ищем проблемную строку и исправляем ее
fixed = False
for i, line in enumerate(lines):
    if "jsonld_pattern" in line and "re.compile" in line:
        print(f"Найдена проблемная строка {i+1}: {line.strip()}")
        # Исправляем регулярное выражение
        lines[i] = '        jsonld_pattern = re.compile(r\'<script[^>]*type=["|\']application/ld\\+json["|\'][^>]*>(.*?)</script>\', re.DOTALL)\n'
        print(f"Исправлено на: {lines[i].strip()}")
        fixed = True

if not fixed:
    print("Проблемная строка не найдена")

# Записываем исправленный файл
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Файл {file_path} успешно обновлен")
