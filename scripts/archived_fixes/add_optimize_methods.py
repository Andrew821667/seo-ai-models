# Добавляем методы для оптимизации схем

schema_path = "seo_ai_models/parsers/unified/extractors/schema_optimizer.py"

# Чтение файла
with open(schema_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Добавляем реализацию методов для оптимизации схем
optimize_methods = """    def _optimize_common_properties(self, schema):
        \"\"\"Оптимизирует общие свойства схемы.\"\"\"
        # Оптимизируем даты
        date_properties = ['datePublished', 'dateModified', 'dateCreated', 'startDate', 'endDate', 'validFrom', 'validThrough']
        
        for prop in date_properties:
            if prop in schema and schema[prop]:
                try:
                    self.improved_date_processing(prop, schema[prop], schema)
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке даты {prop}: {str(e)}")
        
        # Оптимизируем URL
        url_properties = ['url', 'sameAs', 'contentUrl', 'thumbnailUrl']
        
        for prop in url_properties:
            if prop in schema and schema[prop]:
                # Убедимся, что URL абсолютный
                if isinstance(schema[prop], str) and not schema[prop].startswith(('http://', 'https://')):
                    schema[prop] = f"https://{schema[prop]}"
        
        return schema

    def _optimize_article(self, schema):
        \"\"\"Оптимизирует схему статьи.\"\"\"
        # Убедимся, что заголовок есть
        if 'headline' not in schema or not schema['headline']:
            if 'name' in schema:
                schema['headline'] = schema['name']
        
        # Убедимся, что описание есть
        if 'description' not in schema or not schema['description']:
            if 'articleBody' in schema:
                # Берем первые 200 символов текста как описание
                schema['description'] = schema['articleBody'][:200] + '...'
        
        # Оптимизируем автора
        if 'author' in schema:
            if isinstance(schema['author'], dict):
                if '@type' not in schema['author']:
                    schema['author']['@type'] = 'Person'
                
                if 'name' not in schema['author'] or not schema['author']['name']:
                    schema['author']['name'] = 'Unknown Author'
            elif isinstance(schema['author'], str):
                author_name = schema['author']
                schema['author'] = {
                    '@type': 'Person',
                    'name': author_name
                }
        
        return schema

    def _optimize_product(self, schema):
        \"\"\"Оптимизирует схему продукта.\"\"\"
        # Убедимся, что цена указана правильно
        if 'offers' in schema and isinstance(schema['offers'], dict):
            if 'price' in schema['offers'] and isinstance(schema['offers']['price'], str):
                try:
                    # Попытка преобразовать строку в число
                    schema['offers']['price'] = float(schema['offers']['price'].replace(',', '.'))
                except ValueError:
                    pass
            
            # Убедимся, что валюта указана
            if 'priceCurrency' not in schema['offers']:
                schema['offers']['priceCurrency'] = 'USD'
        
        return schema

    def _optimize_organization(self, schema):
        \"\"\"Оптимизирует схему организации.\"\"\"
        # Оптимизируем логотип
        if 'logo' in schema and isinstance(schema['logo'], str):
            logo_url = schema['logo']
            schema['logo'] = {
                '@type': 'ImageObject',
                'url': logo_url
            }
        
        return schema

    def _optimize_person(self, schema):
        \"\"\"Оптимизирует схему человека.\"\"\"
        # Оптимизируем имя
        if 'name' in schema and isinstance(schema['name'], str):
            # Если имя содержит только инициалы, развернем их
            name = schema['name']
            if re.match(r'^[A-ZА-Я]\.\s*[A-ZА-Я]\.\s*[A-ZА-Яa-zа-я]+$', name):
                schema['name'] = f"{name.split('.')[-1].strip()} {'.'.join(name.split('.')[:2])}."
        
        return schema

    def _optimize_event(self, schema):
        \"\"\"Оптимизирует схему события.\"\"\"
        # Оптимизируем местоположение
        if 'location' in schema and isinstance(schema['location'], str):
            location_name = schema['location']
            schema['location'] = {
                '@type': 'Place',
                'name': location_name
            }
        
        return schema

    def _optimize_webpage(self, schema):
        \"\"\"Оптимизирует схему веб-страницы.\"\"\"
        # Убедимся, что указан язык
        if 'inLanguage' not in schema:
            schema['inLanguage'] = 'en'
        
        return schema
"""

# Находим место для вставки методов (в конце класса)
class_end = content.rfind("        return schemas")
if class_end == -1:
    print("Не удалось найти позицию для вставки методов оптимизации")
else:
    # Находим позицию после последней строки метода
    class_end = content.find("\n", class_end)
    
    # Вставляем методы
    new_content = content[:class_end] + "\n" + optimize_methods + content[class_end:]
    
    # Записываем обновленный файл
    with open(schema_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Методы оптимизации успешно добавлены в {schema_path}")
