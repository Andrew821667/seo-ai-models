import pytest
from bs4 import BeautifulSoup
import numpy as np

from seo_ai_models.models.dim_reducer.features import SEOFeaturesExtractor

@pytest.fixture
def extractor():
    """Фикстура для создания экстрактора"""
    return SEOFeaturesExtractor(max_features=50)

@pytest.fixture
def sample_text():
    """Фикстура с тестовым текстом"""
    return """
    This is a sample text for testing SEO features extraction. 
    It contains multiple sentences and some important keywords.
    Keywords like SEO, features, and extraction should be detected.
    This text is long enough to calculate meaningful statistics.
    """

@pytest.fixture
def sample_html():
    """Фикстура с тестовым HTML"""
    return """
    <html>
        <head>
            <title>Test Page Title</title>
            <meta name="description" content="Test description">
            <meta name="keywords" content="test, keywords">
        </head>
        <body>
            <h1>Main Header</h1>
            <h2>Subheader 1</h2>
            <h2>Subheader 2</h2>
            <img src="test.jpg" alt="Test image">
            <img src="test2.jpg">
            <a href="/internal">Internal Link</a>
            <a href="https://external.com">External Link</a>
            <p>Test paragraph content</p>
        </body>
    </html>
    """

def test_text_features_extraction(extractor, sample_text):
    """Тест извлечения текстовых характеристик"""
    features = extractor.extract_text_features(sample_text)
    
    assert isinstance(features, dict)
    assert 'word_count' in features
    assert 'sentence_count' in features
    assert 'avg_word_length' in features
    assert 'vocabulary_richness' in features
    assert 'keywords' in features
    
    assert features['word_count'] > 0
    assert features['sentence_count'] > 0
    assert 0 < features['vocabulary_richness'] <= 1
    assert isinstance(features['keywords'], list)

def test_html_features_extraction(extractor, sample_html):
    """Тест извлечения HTML характеристик"""
    features = extractor.extract_html_features(sample_html)
    
    assert isinstance(features, dict)
    assert features['has_title'] == 1
    assert features['has_meta_description'] == 1
    assert features['has_meta_keywords'] == 1
    assert features['h1_count'] == 1
    assert features['h2_count'] == 2
    assert features['img_count'] == 2
    assert features['internal_links'] == 1
    assert features['external_links'] == 1
    assert features['img_alt_ratio'] == 0.5  # 1 из 2 картинок имеет alt

def test_keyword_extraction(extractor, sample_text):
    """Тест извлечения ключевых слов"""
    features = extractor.extract_text_features(sample_text)
    keywords = features['keywords']
    
    assert isinstance(keywords, list)
    assert len(keywords) > 0
    assert 'seo' in [k.lower() for k in keywords]
    assert 'features' in [k.lower() for k in keywords]

def test_tfidf_features(extractor, sample_text):
    """Тест TF-IDF характеристик"""
    features = extractor.extract_text_features(sample_text)
    
    assert 'tfidf_features' in features
    assert 'feature_names' in features
    assert isinstance(features['tfidf_features'], np.ndarray)
    assert isinstance(features['feature_names'], list)
    assert len(features['feature_names']) == len(features['tfidf_features'][0])

def test_word_frequencies(extractor, sample_text):
    """Тест расчета частот слов"""
    features = extractor.extract_text_features(sample_text)
    word_freqs = features['word_frequencies']
    
    assert isinstance(word_freqs, dict)
    assert len(word_freqs) > 0
    assert all(isinstance(freq, float) for freq in word_freqs.values())
    assert all(0 <= freq <= 1 for freq in word_freqs.values())
    assert sum(word_freqs.values()) <= 1.0 + 1e-10  # с небольшим допуском для погрешности

def test_batch_processing(extractor):
    """Тест пакетной обработки"""
    texts = [
        "First test text with some keywords.",
        "Second text with different content.",
        "Third text for batch processing test."
    ]
    
    features_list = extractor.batch_process(texts)
    
    assert isinstance(features_list, list)
    assert len(features_list) == 3
    assert all(isinstance(f, dict) for f in features_list)
    assert all('word_count' in f for f in features_list)
    assert all('keywords' in f for f in features_list)

def test_combined_extraction(extractor, sample_text, sample_html):
    """Тест совместного извлечения характеристик из текста и HTML"""
    features = extractor.extract_all_features(sample_text, sample_html)
    
    assert isinstance(features, dict)
    # Проверяем наличие текстовых характеристик
    assert 'word_count' in features
    assert 'keywords' in features
    # Проверяем наличие HTML характеристик
    assert 'has_title' in features
    assert 'h1_count' in features

def test_error_handling(extractor):
    """Тест обработки ошибок"""
    # Тест с пустым текстом
    features = extractor.extract_text_features("")
    assert isinstance(features, dict)
    assert features.get('word_count', 0) == 0
    
    # Тест с невалидным HTML
    features = extractor.extract_html_features("<invalid>html>")
    assert isinstance(features, dict)
    
    # Тест с None
    features = extractor.extract_text_features(None)
    assert isinstance(features, dict)

def test_language_support(sample_text):
    """Тест поддержки разных языков"""
    # Создаем экстрактор с другим языком
    extractor_ru = SEOFeaturesExtractor(language='russian')
    features = extractor_ru.extract_text_features(sample_text)
    
    assert isinstance(features, dict)
    assert 'word_count' in features
    assert 'keywords' in features

def test_custom_settings():
    """Тест пользовательских настроек"""
    extractor = SEOFeaturesExtractor(
        max_features=10,
        min_keyword_length=5
    )
    
    text = "This is a test text with some long keywords like optimization"
    features = extractor.extract_text_features(text)
    
    assert len(features['keywords']) <= 10
    assert all(len(k) >= 5 for k in features['keywords'])
