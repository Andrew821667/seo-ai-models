
from pydantic import BaseModel, Field
from typing import Optional, List

class KeywordExtractorConfig(BaseModel):
    """Конфигурация для KeywordExtractor"""
    # Параметры модели
    model_name: str = "cointegrated/rubert-tiny2"
    max_length: int = 512
    language: str = "russian"
    
    # Параметры извлечения ключевых слов
    min_word_length: int = 3
    max_keywords: int = 20
    
    # Параметры TF-IDF
    ngram_range: tuple = (1, 2)
    min_df: int = 1
    max_df: float = 1.0
    max_features: int = 5000
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = True
    
    # Фильтрация по частям речи
    valid_pos_tags: List[str] = Field(
        default=["NOUN", "ADJ", "VERB"],
        description="Допустимые части речи для ключевых слов"
    )

def get_default_config() -> KeywordExtractorConfig:
    """Возвращает конфигурацию по умолчанию"""
    return KeywordExtractorConfig()
