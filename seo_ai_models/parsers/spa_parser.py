"""
Заглушка для SPAParser для совместимости с существующим ядром SEO AI Models.
"""

class SPAParser:
    """
    Заглушка для SPAParser.
    Используется только для совместимости с импортами в ядре системы.
    Фактический функционал теперь перенесен в unified_parser.py.
    """
    
    def __init__(self, **kwargs):
        self.initialized = True
        
    def parse(self, url, **kwargs):
        """
        Заглушка для метода parse.
        
        Args:
            url: URL для парсинга
            **kwargs: Дополнительные параметры
            
        Returns:
            dict: Результаты парсинга
        """
        return {
            "url": url,
            "note": "This is a stub for SPAParser. Use UnifiedParser instead."
        }
