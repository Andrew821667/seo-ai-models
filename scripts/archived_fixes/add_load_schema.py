# Добавляем метод load_schema() в SchemaOptimizer

schema_path = "seo_ai_models/parsers/unified/extractors/schema_optimizer.py"

# Чтение файла
with open(schema_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Добавляем реализацию load_schema
load_schema_method = """    def load_schema(self):
        \"\"\"Загружает и обрабатывает схему Schema.org.\"\"\"
        try:
            response = requests.get(self.schema_url, timeout=10)
            if response.status_code == 200:
                schema_data = response.json()
                
                # Извлекаем типы и свойства
                if '@graph' in schema_data:
                    for item in schema_data['@graph']:
                        if item.get('@type') == 'rdfs:Class':
                            self.schema_types.append(item.get('@id'))
                        elif item.get('@type') == 'rdf:Property':
                            property_id = item.get('@id')
                            domain = item.get('schema:domainIncludes', [])
                            
                            if not isinstance(domain, list):
                                domain = [domain]
                            
                            for d in domain:
                                if isinstance(d, dict) and '@id' in d:
                                    domain_id = d['@id']
                                    if domain_id not in self.schema_properties:
                                        self.schema_properties[domain_id] = []
                                    self.schema_properties[domain_id].append(property_id)
                
                self.logger.info(f"Схема Schema.org успешно загружена. Найдено {len(self.schema_types)} типов и {len(self.schema_properties)} доменов свойств.")
            else:
                self.logger.warning(f"Не удалось загрузить схему Schema.org. Код ответа: {response.status_code}")
                self._set_base_types()
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке схемы Schema.org: {str(e)}")
            self._set_base_types()
"""

# Находим место для вставки метода (после метода __init__)
init_end = content.find("    def improved_date_processing")
if init_end == -1:
    print("Не удалось найти позицию для вставки метода load_schema()")
else:
    # Вставляем метод
    new_content = content[:init_end] + load_schema_method + content[init_end:]
    
    # Записываем обновленный файл
    with open(schema_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Метод load_schema() успешно добавлен в {schema_path}")
