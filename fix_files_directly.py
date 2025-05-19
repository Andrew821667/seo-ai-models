# Полностью перезаписываем проблемные части файлов

import os

# Исправляем metadata_enhancer.py
metadata_path = "seo_ai_models/parsers/unified/extractors/metadata_enhancer.py"
schema_path = "seo_ai_models/parsers/unified/extractors/schema_optimizer.py"

# Создаем резервные копии
os.system(f"cp {metadata_path} {metadata_path}.bak")
os.system(f"cp {schema_path} {schema_path}.bak")

print("Созданы резервные копии файлов")

# Читаем содержимое файлов
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata_content = f.readlines()

with open(schema_path, 'r', encoding='utf-8') as f:
    schema_content = f.readlines()

# Исправляем проблему в metadata_enhancer.py
for i, line in enumerate(metadata_content):
    if "дата обращения" in line and not line.strip().endswith('"'):
        # Определяем уровень отступа
        indent = len(line) - len(line.lstrip())
        # Создаем корректную строку с правильной кавычкой в конце
        metadata_content[i] = ' ' * indent + 'gost += f"— (дата обращения: {accessed_date}). "\n'
        print(f"Исправлена строка {i+1} в {metadata_path}")

# Исправляем проблему в schema_optimizer.py
for i, line in enumerate(schema_content):
    if "jsonld_pattern" in line:
        # Определяем уровень отступа
        indent = len(line) - len(line.lstrip())
        # Создаем корректную строку с исправленным регулярным выражением
        schema_content[i] = ' ' * indent + 'jsonld_pattern = re.compile(r\'<script[^>]*type=["|\']application/ld\\+json["|\'][^>]*>(.*?)</script>\', re.DOTALL)\n'
        print(f"Исправлена строка {i+1} в {schema_path}")

# Записываем исправленные файлы
with open(metadata_path, 'w', encoding='utf-8') as f:
    f.writelines(metadata_content)

with open(schema_path, 'w', encoding='utf-8') as f:
    f.writelines(schema_content)

print("Файлы успешно исправлены")
