# Создаем временные файлы для проверки
import os

# Путь к исходному файлу
file_path = "seo_ai_models/parsers/unified/extractors/metadata_enhancer.py"

# Проверяем существование файла
if not os.path.exists(file_path):
    print(f"Файл {file_path} не существует")
    exit(1)

# Создаем временную копию файла для анализа
temp_file_path = "metadata_enhancer_temp.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

with open(temp_file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Создана временная копия файла {file_path} в {temp_file_path}")

# Просмотр проблемного метода
method_name = "_create_citation_gost_style"
method_start = content.find(f"def {method_name}")
if method_start != -1:
    # Ищем начало и конец метода
    method_end = content.find("\n    def ", method_start + 1)
    if method_end == -1:
        method_end = len(content)
    
    method_content = content[method_start:method_end]
    
    # Записываем метод в отдельный файл для анализа
    method_file_path = f"{method_name}_temp.py"
    with open(method_file_path, 'w', encoding='utf-8') as f:
        f.write(method_content)
    
    print(f"Метод {method_name} сохранен в файл {method_file_path}")
else:
    print(f"Метод {method_name} не найден в файле")

# Поиск строки с "дата обращения"
pattern = "дата обращения"
lines = content.split('\n')
found = False

for i, line in enumerate(lines):
    if pattern in line:
        print(f"Строка {i+1}: {line}")
        found = True

if not found:
    print(f"Строка с '{pattern}' не найдена")
