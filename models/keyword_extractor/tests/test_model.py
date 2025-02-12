# tests/test_model.py

import pytest
import torch
from pathlib import Path
import tempfile
import json

from model.model import KeywordExtractorModel
from model.config.model_config import KeywordModelConfig

@pytest.fixture
def model_config():
    """Фикстура с конфигурацией модели"""
    return KeywordModelConfig(
        model_name="bert-base-uncased",
        max_length=128,
        input_dim=768,
        hidden_dim=256,
        num_layers=2
    )

@pytest.fixture
def model(model_config):
    """Фикстура с инициализированной моделью"""
    return KeywordExtractorModel(model_config)

@pytest.fixture
def sample_texts():
    """Фикстура с тестовыми текстами"""
    return [
        "This is a sample text for keyword extraction.",
        "Another example with some important keywords to find."
    ]

class TestKeywordExtractorModel:
    """Тесты для модели извлечения ключевых слов"""
    
    def test_model_initialization(self, model_config):
        """Тест инициализации модели"""
        model = KeywordExtractorModel(model_config)
        assert isinstance(model, KeywordExtractorModel)
        assert model.config == model_config
        
    def test_model_forward_pass(self, model, sample_texts):
        """Тест прямого прохода модели"""
        # Кодирование текстов
        inputs = model.processor.encode_texts(sample_texts)
        
        # Прямой проход
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
        # Проверка выходов
        assert 'keyword_logits' in outputs
        assert 'trend_scores' in outputs
        assert outputs['keyword_logits'].shape[0] == len(sample_texts)
        
    def test_keyword_extraction(self, model, sample_texts):
        """Тест извлечения ключевых слов"""
        keywords = model.extract_keywords(
            texts=sample_texts,
            threshold=0.5
        )
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Проверка формата результатов
        for kw in keywords:
            assert isinstance(kw, dict)
            assert 'keyword' in kw
            assert 'score' in kw
            assert isinstance(kw['keyword'], str)
            assert isinstance(kw['score'], float)
            
    def test_model_save_load(self, model, tmp_path):
        """Тест сохранения и загрузки модели"""
        # Сохранение модели
        save_path = tmp_path / "test_model"
        model.save_pretrained(save_path)
        
        # Проверка сохраненных файлов
        assert (save_path / "pytorch_model.bin").exists()
        assert (save_path / "config.json").exists()
        
        # Загрузка модели
        loaded_model = KeywordExtractorModel.from_pretrained(save_path)
        assert isinstance(loaded_model, KeywordExtractorModel)
        
        # Проверка равенства весов
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2)
            
    def test_batch_processing(self, model):
        """Тест пакетной обработки"""
        # Создание батча
        texts = [f"Sample text {i}" for i in range(5)]
        
        # Обработка батча
        inputs = model.processor.encode_texts(texts)
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
        # Проверка размеров выходов
        assert outputs['keyword_logits'].shape[0] == len(texts)
        assert outputs['trend_scores'].shape[0] == len(texts)
        
    @pytest.mark.parametrize(
        "threshold",
        [0.3, 0.5, 0.7]
    )
    def test_threshold_effect(self, model, sample_texts, threshold):
        """Тест влияния порога уверенности"""
        keywords = model.extract_keywords(
            texts=sample_texts,
            threshold=threshold
        )
        
        # Проверка, что все оценки выше порога
        for text_keywords in keywords:
            for kw in text_keywords:
                assert kw['score'] >= threshold
                
    def test_error_handling(self, model):
        """Тест обработки ошибок"""
        # Тест с пустым текстом
        with pytest.raises(ValueError):
            model.extract_keywords([""])
            
        # Тест с None
        with pytest.raises(ValueError):
            model.extract_keywords([None])
            
        # Тест с некорректным порогом
        with pytest.raises(ValueError):
            model.extract_keywords(
                ["Sample text"],
                threshold=1.5
            )
            
    def test_attention_mechanism(self, model, sample_texts):
        """Тест механизма внимания"""
        # Получение весов внимания
        inputs = model.processor.encode_texts(sample_texts)
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
        # Проверка весов внимания
        assert 'attention_weights' in outputs
        attention_weights = outputs['attention_weights']
        
        # Проверка нормализации весов
        assert torch.allclose(
            attention_weights.sum(dim=-1),
            torch.ones_like(attention_weights.sum(dim=-1)),
            atol=1e-5
        )
        
    def test_model_config_serialization(self, model_config):
        """Тест сериализации конфигурации"""
        # Сохранение конфигурации
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            model_config.save(f.name)
            
            # Загрузка конфигурации
            loaded_config = KeywordModelConfig.load(f.name)
            
        # Проверка равенства конфигураций
        assert model_config.dict() == loaded_config.dict()
