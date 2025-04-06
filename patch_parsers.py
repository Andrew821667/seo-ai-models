"""
Скрипт для исправления проблем с парсерами.
"""

from seo_ai_models.parsers.extractors.meta_extractor import MetaExtractor
from seo_ai_models.parsers.extractors.meta_extractor_update import update_meta_extractor

# Обновляем MetaExtractor
update_meta_extractor(MetaExtractor)

print("Модули успешно обновлены!")
