# Путь к файлу
file_path = "seo_ai_models/parsers/unified/extractors/schema_optimizer.py"

# Чтение файла
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Находим строку с ошибкой в регулярном выражении
error_pattern = "jsonld_pattern = re.compile(r'<script[^>]*type=[\"']application/ld\\+json[\"'][^>]*>(.*?)</script>', re.DOTALL)"

if error_pattern in content:
    # Исправляем - корректируем регулярное выражение
    fixed_pattern = 'jsonld_pattern = re.compile(r\'<script[^>]*type=["|\']application/ld\\+json["|\'][^>]*>(.*?)</script>\', re.DOTALL)'
    content = content.replace(error_pattern, fixed_pattern)
    
    # Записываем исправленный контент
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Синтаксическая ошибка в {file_path} исправлена")
else:
    print(f"Строка с ошибкой не найдена в {file_path}")
