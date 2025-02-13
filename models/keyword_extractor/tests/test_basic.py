import sys
sys.path.append('/content')

from keyword_extractor.processor import KeywordProcessor

def test_keyword_extraction():
    text = """
    Искусственный интеллект (ИИ) - это способность компьютерных систем выполнять задачи, 
    которые обычно требуют человеческого интеллекта. Эти задачи включают распознавание речи, 
    принятие решений, визуальное восприятие и перевод между языками. ИИ становится все более 
    важным в современных технологиях, находя применение в различных областях от медицины до 
    автомобильной промышленности.
    """

    processor = KeywordProcessor()
    print("Начинаем извлечение ключевых слов...")
    keywords = processor.extract_keywords(text)
    print("\nИтоговые ключевые слова:")
    for word, score in keywords:
        print(f"{word}: {score:.4f}")

if __name__ == "__main__":
    test_keyword_extraction()
