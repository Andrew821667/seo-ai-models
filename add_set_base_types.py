# Добавляем метод _set_base_types() в SchemaOptimizer

schema_path = "seo_ai_models/parsers/unified/extractors/schema_optimizer.py"

# Чтение файла
with open(schema_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Добавляем реализацию _set_base_types
set_base_types_method = """    def _set_base_types(self):
        \"\"\"Устанавливает базовые типы Schema.org при неудачной загрузке схемы.\"\"\"
        base_types = [
            "Thing", "Action", "CreativeWork", "Event", "Organization", 
            "Person", "Place", "Product", "WebPage", "WebSite"
        ]
        self.schema_types = base_types
        
        # Базовые свойства для наиболее распространенных типов
        self.schema_properties = {
            "Article": ["headline", "author", "datePublished", "dateModified", "publisher", "description", "image"],
            "NewsArticle": ["headline", "author", "datePublished", "dateModified", "publisher", "description", "image"],
            "BlogPosting": ["headline", "author", "datePublished", "dateModified", "publisher", "description", "image"],
            "Product": ["name", "description", "brand", "offers", "sku", "image", "review", "aggregateRating"],
            "Organization": ["name", "description", "logo", "url", "address", "telephone", "email"],
            "Person": ["name", "jobTitle", "telephone", "email", "address", "image"],
            "Event": ["name", "startDate", "endDate", "location", "organizer", "description"],
            "WebPage": ["name", "description", "url", "author", "datePublished", "dateModified"],
            "WebSite": ["name", "description", "url", "publisher"]
        }
        
        self.logger.info(f"Установлены базовые типы Schema.org: {', '.join(base_types)}")
"""

# Находим место для вставки метода (после метода load_schema)
position = content.find("    def improved_date_processing")
if position == -1:
    print("Не удалось найти позицию для вставки метода _set_base_types()")
else:
    # Вставляем метод
    new_content = content[:position] + set_base_types_method + content[position:]
    
    # Записываем обновленный файл
    with open(schema_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Метод _set_base_types() успешно добавлен в {schema_path}")
