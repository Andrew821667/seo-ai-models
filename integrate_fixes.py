import os
import re
import shutil
from pathlib import Path

def backup_file(file_path):
    """Создает резервную копию файла"""
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Создана резервная копия файла: {backup_path}")

def integrate_enhanced_spa_crawler_fixes():
    """Интегрирует исправления в enhanced_spa_crawler_llm.py"""
    file_path = "seo_ai_models/parsers/unified/crawlers/enhanced_spa_crawler_llm.py"
    backup_file(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Заменяем заглушку метода _init_crawler
    pattern = r"def _init_crawler\(self\):\s+\"\"\"[^\"]*\"\"\"\s+pass\s+#\s+Инициализация происходит в методе crawl_url"
    with open('enhanced_spa_crawler_fixes.py', 'r', encoding='utf-8') as f:
        init_crawler_code = f.read()
    
    content = re.sub(pattern, init_crawler_code, content)
    
    # Заменяем заглушку обработки исключения при определении языка
    pattern = r"except:\s+pass\s+# По умолчанию возвращаем язык, указанный при инициализации"
    replacement = """except:
        lang_code = self.improved_language_detection_exception_handling(text_sample)
        return lang_code
        
        # По умолчанию возвращаем язык, указанный при инициализации"""
    
    content = re.sub(pattern, replacement, content)
    
    # Добавляем новый метод для обработки исключения
    with open('enhanced_spa_crawler_fixes.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    improved_language_detection_code = "".join(lines[lines.index("def improved_language_detection_exception_handling(self, text_sample):\n"):])
    
    # Добавляем новый метод перед концом класса
    last_method_end = content.rfind("    def ")
    last_method_end = content.find("\n\n", last_method_end)
    
    content = content[:last_method_end] + "\n\n" + improved_language_detection_code + content[last_method_end:]
    
    # Добавляем импорт для re, если его нет
    if "import re" not in content:
        import_section_end = content.find("\n\n", content.find("import"))
        content = content[:import_section_end] + "\nimport re" + content[import_section_end:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Исправления успешно интегрированы в {file_path}")

def integrate_metadata_enhancer_fixes():
    """Интегрирует исправления в metadata_enhancer.py"""
    file_path = "seo_ai_models/parsers/unified/extractors/metadata_enhancer.py"
    backup_file(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Заменяем заглушку обработки исключения при извлечении года
    pattern = r"if published_date:\s+try:\s+year = re\.search\(r'\\d{4}', published_date\)\.group\(0\)\s+gost \+= f\"— {year}\. \"\s+except:\s+pass"
    replacement = """if published_date:
        gost = self.improved_date_extraction(published_date, gost, url)
    else:"""
    
    content = re.sub(pattern, replacement, content)
    
    # Убираем дублирующуюся обработку URL
    pattern = r"if url:\s+gost \+= f\"URL: {url}\""
    replacement = "# URL уже добавлен в improved_date_extraction"
    
    content = re.sub(pattern, replacement, content)
    
    # Добавляем новый метод для обработки даты
    with open('metadata_enhancer_fixes.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    improved_date_extraction_code = "".join(lines[lines.index("def improved_date_extraction(self, published_date, gost, url):\n"):])
    
    # Добавляем новый метод перед концом класса
    last_method_end = content.rfind("    def ")
    last_method_end = content.find("\n\n", last_method_end)
    
    content = content[:last_method_end] + "\n\n" + improved_date_extraction_code + content[last_method_end:]
    
    # Добавляем импорт для datetime, если его нет
    if "from datetime import datetime" not in content:
        import_section_end = content.find("\n\n", content.find("import"))
        content = content[:import_section_end] + "\nfrom datetime import datetime" + content[import_section_end:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Исправления успешно интегрированы в {file_path}")

def integrate_schema_optimizer_fixes():
    """Интегрирует исправления в schema_optimizer.py"""
    file_path = "seo_ai_models/parsers/unified/extractors/schema_optimizer.py"
    backup_file(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Заменяем заглушку обработки исключения при преобразовании даты
    pattern = r"try:\s+parsed_date = dateutil\.parser\.parse\(value\)\s+result\[prop\] = parsed_date\.strftime\('%Y-%m-%d'\)\s+except Exception:\s+# Оставляем как есть, если не удалось преобразовать\s+pass"
    replacement = """try:
            self.improved_date_processing(prop, value, result)
        except Exception as e:
            self.logger.error(f"Непредвиденная ошибка при обработке даты {value}: {str(e)}")
            result[prop] = value"""
    
    content = re.sub(pattern, replacement, content)
    
    # Добавляем новый метод для обработки даты
    with open('schema_optimizer_fixes.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    improved_date_processing_code = "".join(lines[lines.index("def improved_date_processing(self, prop, value, result):\n"):])
    
    # Добавляем новый метод перед концом класса
    last_method_end = content.rfind("    def ")
    last_method_end = content.find("\n\n", last_method_end)
    
    content = content[:last_method_end] + "\n\n" + improved_date_processing_code + content[last_method_end:]
    
    # Добавляем импорт для re, если его нет
    if "import re" not in content:
        import_section_end = content.find("\n\n", content.find("import"))
        content = content[:import_section_end] + "\nimport re" + content[import_section_end:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Исправления успешно интегрированы в {file_path}")

if __name__ == "__main__":
    print("Интеграция исправлений в файлы проекта...")
    
    integrate_enhanced_spa_crawler_fixes()
    integrate_metadata_enhancer_fixes()
    integrate_schema_optimizer_fixes()
    
    print("Все исправления успешно интегрированы!")
