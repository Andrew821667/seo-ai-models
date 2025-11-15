import os

# Пути к файлам
metadata_path = "seo_ai_models/parsers/unified/extractors/metadata_enhancer.py"
schema_path = "seo_ai_models/parsers/unified/extractors/schema_optimizer.py"

# Полностью переписываем проблемный метод в metadata_enhancer.py
def fix_metadata_enhancer():
    # Читаем файл построчно
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Ищем и исправляем проблемные строки
    for i, line in enumerate(lines):
        # Заменяем проблемные строки с датой обращения
        if "дата обращения" in line and not line.strip().endswith('"'):
            indent = len(line) - len(line.lstrip())
            lines[i] = ' ' * indent + 'gost += f"— (дата обращения: {accessed_date}). "\n'
            print(f"Исправлена строка {i+1} в {metadata_path}")
    
    # Записываем исправленный файл
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Файл {metadata_path} успешно исправлен")

# Исправляем проблему с регулярным выражением в schema_optimizer.py
def fix_schema_optimizer():
    # Читаем файл построчно
    with open(schema_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Ищем и исправляем проблемную строку с jsonld_pattern
    for i, line in enumerate(lines):
        if "jsonld_pattern" in line and "re.compile" in line:
            # Создаем корректную строку
            indent = len(line) - len(line.lstrip())
            lines[i] = ' ' * indent + 'jsonld_pattern = re.compile(r"""<script[^>]*type=["|\']application/ld\\+json["|\'][^>]*>(.*?)</script>""", re.DOTALL)\n'
            print(f"Исправлена строка {i+1} в {schema_path}")
    
    # Записываем исправленный файл
    with open(schema_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Файл {schema_path} успешно исправлен")

# Запускаем исправления
fix_metadata_enhancer()
fix_schema_optimizer()

print("Все исправления успешно применены")
