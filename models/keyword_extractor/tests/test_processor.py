import pytest
from ..model.processor import KeywordProcessor

def test_basic_keyword_extraction():
    processor = KeywordProcessor(min_weight=5.0)
    text = "искусственный интеллект обрабатывает естественный язык"
    
    keywords = processor.extract_keywords(text)
    
    assert "искусственный интеллект" in keywords
    assert "естественный язык" in keywords
    assert all(weight >= 5.0 for weight in keywords.values())

def test_weight_filtering():
    processor = KeywordProcessor(min_weight=10.0)
    text = "простой тест обработки текста"
    
    keywords = processor.extract_keywords(text)
    
    assert all(weight >= 10.0 for weight in keywords.values())

def test_stop_words_filtering():
    processor = KeywordProcessor()
    text = "и в на простой текст"
    
    keywords = processor.extract_keywords(text)
    
    assert "и" not in keywords
    assert "в" not in keywords
    assert "на" not in keywords

def test_verb_filtering():
    processor = KeywordProcessor()
    text = "программа обрабатывает данные"
    
    keywords = processor.extract_keywords(text)
    
    assert "обрабатывать" not in keywords
