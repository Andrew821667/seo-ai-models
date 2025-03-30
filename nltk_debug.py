import nltk
import sys
from nltk.tokenize import sent_tokenize, word_tokenize

def debug_nltk():
    """Диагностика ресурсов NLTK и токенизации"""
    print("=== Диагностика NLTK ===")
    
    # Проверка путей поиска NLTK
    print("\nПути для поиска ресурсов NLTK:")
    for path in nltk.data.path:
        print(f"- {path}")
    
    # Проверка наличия ключевых ресурсов
    resources = ['punkt', 'stopwords', 'punkt_tab']
    print("\nПроверка ресурсов:")
    for resource in resources:
        try:
            if resource == 'punkt_tab':
                # Специальная проверка для punkt_tab
                try:
                    nltk.data.find('tokenizers/punkt_tab/english')
                    print(f"✅ {resource} (english): найден")
                except LookupError:
                    print(f"❌ {resource} (english): НЕ найден")
            else:
                nltk.data.find(f'tokenizers/{resource}')
                print(f"✅ {resource}: найден")
        except LookupError:
            print(f"❌ {resource}: НЕ найден")
    
    # Попробуем токенизировать текст
    print("\nТестирование токенизации:")
    test_text = "Это тестовое предложение. А это второе предложение!"
    
    try:
        sentences = sent_tokenize(test_text)
        print(f"sent_tokenize успешно: {sentences}")
    except Exception as e:
        print(f"sent_tokenize ошибка: {e}")
    
    try:
        words = word_tokenize(test_text)
        print(f"word_tokenize успешно: {words[:5]}...")
    except Exception as e:
        print(f"word_tokenize ошибка: {e}")
    
    # Проверка зависимостей punkt_tab
    print("\nГлубокая проверка зависимостей:")
    
    # Проверяем, какие модули импортируют punkt_tab
    # Используем инспекцию исходного кода NLTK
    try:
        import inspect
        from nltk.tokenize import punkt
        print("Исходный код пакета punkt:")
        print("----------------")
        punkt_source = inspect.getsource(punkt)
        
        # Ищем упоминания punkt_tab
        import re
        punkt_tab_refs = re.findall(r'punkt_tab', punkt_source)
        print(f"Найдено упоминаний punkt_tab: {len(punkt_tab_refs)}")
        
        # Если нашли упоминания, показываем строки с ними
        if punkt_tab_refs:
            lines = punkt_source.split('\n')
            for i, line in enumerate(lines):
                if 'punkt_tab' in line:
                    context_start = max(0, i-1)
                    context_end = min(len(lines), i+2)
                    print(f"Строки {context_start}-{context_end}:")
                    for j in range(context_start, context_end):
                        print(f"{j+1}: {lines[j]}")
    except Exception as e:
        print(f"Ошибка при анализе кода: {e}")

if __name__ == "__main__":
    debug_nltk()
