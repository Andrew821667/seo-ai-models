# Создаем временные файлы для проверки
import os

# Путь к исходному файлу
file_path = "seo_ai_models/parsers/unified/extractors/schema_optimizer.py"

# Проверяем существование файла
if not os.path.exists(file_path):
    print(f"Файл {file_path} не существует")
    exit(1)

# Создаем временную копию файла для анализа
temp_file_path = "schema_optimizer_temp.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

with open(temp_file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Создана временная копия файла {file_path} в {temp_file_path}")

# Поиск строки с jsonld_pattern
pattern = "jsonld_pattern"
lines = content.split('\n')
found = False

for i, line in enumerate(lines):
    if pattern in line:
        print(f"Строка {i+1}: {line}")
        found = True

if not found:
    print(f"Строка с '{pattern}' не найдена")
