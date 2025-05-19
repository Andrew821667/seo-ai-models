# Путь к файлу
file_path = "seo_ai_models/parsers/unified/crawlers/enhanced_spa_crawler_llm.py"

# Чтение файла
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Найдем строку с заглушкой для _init_crawler
init_pattern = r'def _init_crawler\(self\):\s+"""Инициализирует компоненты краулера\."""\s+pass\s+#\s+Инициализация происходит в методе crawl_url'

if init_pattern in content:
    replacement = '''def _init_crawler(self):
        """Инициализирует компоненты краулера."""
        # Инициализируем playwright если еще не инициализирован
        if not hasattr(self, 'playwright') or self.playwright is None:
            self._start_browser()
        
        # Инициализируем параметры
        self.browser_content = None
        self.navigation_history = []
        self.extracted_links = set()
        self.processed_links = set()
        self.resources_info = {}
        self.requests_log = []
        self.dom_changes = []
        
        # Инициализируем компоненты для извлечения данных
        try:
            from seo_ai_models.parsers.unified.extractors.structured_data_extractor import StructuredDataExtractor
            from seo_ai_models.parsers.unified.extractors.metadata_enhancer import MetadataEnhancer
            
            self.structured_data_extractor = StructuredDataExtractor()
            self.metadata_enhancer = MetadataEnhancer()
            
            logger.info(f"Краулер инициализирован с параметрами: headless={self.headless}, "
                        f"wait_for_network_idle={self.wait_for_network_idle}, wait_time={self.wait_time}")
        except ImportError as e:
            logger.warning(f"Не удалось импортировать компоненты для извлечения данных: {str(e)}")'''
    
    content = content.replace(init_pattern, replacement)

    # Найдем другую заглушку при определении языка
    lang_pattern = r'except:\s+pass\s+# По умолчанию возвращаем язык, указанный'
    
    if lang_pattern in content:
        lang_replacement = '''except Exception as e:
        logger.warning(f"Ошибка при определении языка с помощью langdetect: {str(e)}")
        
        # Альтернативный подход - анализ наличия характерных символов
        ru_chars = len(re.findall('[а-яА-Я]', text_sample))
        en_chars = len(re.findall('[a-zA-Z]', text_sample))
        
        if ru_chars > 0 and ru_chars > en_chars * 0.3:  # если русских символов больше 30% от английских
            logger.info("Язык определен альтернативным способом: ru")
            return 'ru'
        elif en_chars > 0:
            logger.info("Язык определен альтернативным способом: en")
            return 'en'
        
        # По умолчанию возвращаем язык, указанный'''
        
        content = content.replace(lang_pattern, lang_replacement)

    # Убедимся, что импорт re присутствует
    if "import re" not in content:
        import_section_end = content.find("\n\n", content.find("import"))
        content = content[:import_section_end] + "\nimport re" + content[import_section_end:]

    # Записываем изменения обратно в файл
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Заглушки в файле {file_path} успешно заменены")
else:
    print(f"Заглушки не найдены или уже исправлены в файле {file_path}")
