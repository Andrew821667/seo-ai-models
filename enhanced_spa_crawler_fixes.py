def _init_crawler(self):
    """Инициализирует компоненты краулера."""
    # Инициализируем playwright если еще не инициализирован
    if self.playwright is None:
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
    self.structured_data_extractor = None
    self.metadata_enhancer = None
    
    # Настройка логгера для сбора данных о работе краулера
    self.logger.info(f"Краулер инициализирован с параметрами: headless={self.headless}, "
                     f"wait_for_network_idle={self.wait_for_network_idle}, wait_time={self.wait_time}")

# Улучшенная обработка исключения при определении языка
def improved_language_detection_exception_handling(self, text_sample):
    try:
        lang_code = langdetect.detect(text_sample)
        return lang_code
    except langdetect.LangDetectException as e:
        self.logger.warning(f"Ошибка LangDetect при определении языка: {str(e)}")
        # Альтернативный способ определения языка - по частоте встречаемости характерных символов
        ru_chars = len(re.findall('[а-яА-Я]', text_sample))
        en_chars = len(re.findall('[a-zA-Z]', text_sample))
        
        if ru_chars > en_chars * 0.3:  # если русских символов больше 30% от английских
            self.logger.info("Язык определен альтернативным способом: ru")
            return 'ru'
        elif en_chars > 0:
            self.logger.info("Язык определен альтернативным способом: en")
            return 'en'
        
        # Если не удалось определить язык, используем язык по умолчанию
        self.logger.info(f"Не удалось определить язык, используется язык по умолчанию: {self.language}")
        return self.language
