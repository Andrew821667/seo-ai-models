# model/examples/basic_usage.py

import torch
from pathlib import Path
from model.model import KeywordExtractorModel
from model.config.model_config import KeywordModelConfig

def basic_extraction():
    """Базовый пример извлечения ключевых слов"""
    # Инициализация конфигурации
    config = KeywordModelConfig(
        model_name="xlm-roberta-base",
        max_length=512,
        hidden_dim=256
    )
    
    # Создание модели
    model = KeywordExtractorModel(config)
    
    # Пример текста
    text = """
    Machine learning is a subset of artificial intelligence that involves 
    training computer systems to learn from data without explicit programming. 
    Deep learning, a more specialized form of machine learning, uses neural 
    networks with multiple layers to process complex patterns.
    """
    
    # Извлечение ключевых слов
    keywords = model.extract_keywords(
        texts=[text],
        threshold=0.5
    )
    
    # Вывод результатов
    print("Extracted keywords:")
    for kw in keywords:
        print(f"- {kw['keyword']} (score: {kw['score']:.2f})")
        
def batch_processing():
    """Пример пакетной обработки текстов"""
    config = KeywordModelConfig()
    model = KeywordExtractorModel(config)
    
    # Список текстов
    texts = [
        "Python is a high-level programming language.",
        "Data science combines statistics and programming.",
        "Natural language processing analyzes human language."
    ]
    
    # Пакетная обработка
    results = model.extract_keywords(
        texts=texts,
        threshold=0.4,
        batch_size=2
    )
    
    # Вывод результатов по каждому тексту
    for i, text_results in enumerate(results):
        print(f"\nText {i + 1} keywords:")
        for kw in text_results['keywords']:
            print(f"- {kw['keyword']} (score: {kw['score']:.2f})")
            
def save_and_load():
    """Пример сохранения и загрузки модели"""
    # Создание и сохранение
    config = KeywordModelConfig()
    model = KeywordExtractorModel(config)
    
    save_path = Path("saved_model")
    model.save_pretrained(save_path)
    
    # Загрузка
    loaded_model = KeywordExtractorModel.from_pretrained(save_path)
    
    # Проверка работы загруженной модели
    text = "Testing the loaded model with sample text."
    keywords = loaded_model.extract_keywords([text])
    print("\nKeywords from loaded model:")
    for kw in keywords:
        print(f"- {kw['keyword']} (score: {kw['score']:.2f})")
        
def main():
    print("1. Basic keyword extraction:")
    basic_extraction()
    
    print("\n2. Batch processing:")
    batch_processing()
    
    print("\n3. Save and load model:")
    save_and_load()

if __name__ == "__main__":
    main()
