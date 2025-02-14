import pytest
from models.keyword_extractor.model.processor import KeywordProcessor

def test_keyword_processor_initialization():
    processor = KeywordProcessor()
    assert processor.weights['adj_noun'] == 40.0
    assert processor.weights['noun_noun'] == 35.0
    assert processor.weights['single_noun'] == 30.0
    assert processor.weights['position_multipliers']['title'] == 2.0

def test_basic_keyword_extraction():
    processor = KeywordProcessor()
    text = """Новый метод анализа
    
    Этот инновационный подход позволяет улучшить результаты.
    
    Метод анализа показывает эффективность."""
    
    keywords = processor.extract_keywords(text)
    assert len(keywords) > 0
    # Проверяем, что 'новый метод' имеет больший вес из-за позиции в заголовке
    assert keywords.get('новый метод', 0) > keywords.get('метод анализ', 0)

def test_word_pair_analysis():
    processor = KeywordProcessor()
    
    # Тест прилагательное + существительное
    is_valid, pair_type, (lemma1, lemma2) = processor._analyze_word_pair('новый', 'метод')
    assert is_valid
    assert pair_type == 'adj_noun'
    
    # Тест существительное + существительное
    is_valid, pair_type, (lemma1, lemma2) = processor._analyze_word_pair('метод', 'анализ')
    assert is_valid
    assert pair_type == 'noun_noun'

def test_position_multiplier():
    processor = KeywordProcessor()
    text = """Заголовок статьи
    
    Первый параграф текста.
    
    Средний параграф.
    
    Последний параграф."""
    
    # Проверяем множитель для заголовка
    title_mult = processor._get_position_multiplier(text, 0)
    assert title_mult == processor.weights['position_multipliers']['title']
    
    # Проверяем множитель для первого параграфа
    first_para_mult = processor._get_position_multiplier(text, 3)
    assert first_para_mult == processor.weights['position_multipliers']['first_para']
