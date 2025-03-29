import sys
sys.path.append('/content/seo-ai-models')

# Предварительная загрузка ресурсов NLTK
import nltk
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    print("✅ NLTK ресурсы успешно загружены")
except Exception as e:
    print(f"⚠️ Проблема при загрузке NLTK ресурсов: {str(e)}")

from models.seo_advisor.enhanced_eeat_analyzer import EnhancedEEATAnalyzer, generate_synthetic_training_data
import time
import json

def test_basic_functionality():
    """Тест базовой функциональности анализатора"""
    print("=== Тест базовой функциональности ===")
    
    # Создаем экземпляр анализатора
    analyzer = EnhancedEEATAnalyzer()
    
    # Тестовый текст
    sample_text = """
    # Инвестиционные стратегии для начинающих
    
    По данным исследования, проведенного экспертами финансового рынка в 2023 году, 
    более 65% начинающих инвесторов совершают одни и те же ошибки.
    
    ## Основные принципы
    
    Согласно рекомендациям профессиональных финансовых консультантов, 
    следует придерживаться следующих принципов:
    
    * Диверсификация активов
    * Долгосрочная стратегия
    * Регулярные инвестиции
    
    ## Распределение активов
    
    Как отмечает Джон Смит, сертифицированный финансовый аналитик с опытом более 15 лет:
    «Правильное распределение активов может составлять до 90% успеха вашей инвестиционной стратегии».
    
    Источник: Financial Times, https://ft.com/article12345
    
    > Важно: данная информация не является индивидуальной инвестиционной рекомендацией. 
    > Проконсультируйтесь со своим финансовым советником перед принятием решений.
    
    Последнее обновление: 10.03.2023
    """
    
    # Анализируем текст
    start_time = time.time()
    result = analyzer.analyze(sample_text, industry='finance')
    analysis_time = time.time() - start_time
    
    # Выводим результаты
    print(f"Анализ выполнен за {analysis_time:.2f} сек.")
    print(f"Общая оценка E-E-A-T: {result['overall_eeat_score']:.2f}")
    print(f"Оценка экспертизы: {result['expertise_score']:.2f}")
    print(f"Оценка авторитетности: {result['authority_score']:.2f}")
    print(f"Оценка доверия: {result['trust_score']:.2f}")
    print(f"Оценка структуры: {result['structural_score']:.2f}")
    print(f"Оценка семантической связности: {result['semantic_coherence_score']:.2f}")
    
    # Выводим найденные маркеры
    component_details = result['component_details']
    print("\nНайденные маркеры экспертизы:")
    for marker in component_details['expertise'].get('found_markers', [])[:5]:
        print(f"- {marker}")
    
    # Выводим рекомендации
    print("\nРекомендации:")
    for rec in result['recommendations']:
        print(f"- {rec}")
        
    return result

def test_model_training():
    """Тест обучения модели машинного обучения"""
    print("\n=== Тест обучения модели машинного обучения ===")
    
    # Генерация синтетических данных
    data = generate_synthetic_training_data(count=200)
    print(f"Сгенерировано {len(data)} примеров для обучения")
    
    # Создание анализатора и обучение модели
    analyzer = EnhancedEEATAnalyzer()
    model_path = 'eeat_model.joblib'
    
    # Обучение модели
    start_time = time.time()
    analyzer.train_model(data, output_path=model_path)
    training_time = time.time() - start_time
    
    print(f"Модель обучена за {training_time:.2f} сек. и сохранена в {model_path}")
    
    # Получение важности признаков
    feature_importance = analyzer.get_feature_importance()
    
    print("\nВажность признаков:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"- {feature}: {importance:.4f}")
    
    # Тестирование модели на новом примере
    sample_text = """
    # Как создать здоровый рацион питания
    
    В этой статье, написанной доктором медицинских наук, профессором Ивановым И.И., 
    мы рассмотрим основные принципы составления здорового рациона.
    
    Согласно последним исследованиям, опубликованным в журнале "The Lancet" в 2023 году,
    правильное питание может существенно снизить риск развития сердечно-сосудистых заболеваний.
    
    ## Основные правила
    1. Разнообразие продуктов
    2. Баланс белков, жиров и углеводов
    3. Достаточное количество воды
    
    «Питание должно быть не только сбалансированным, но и персонализированным с учетом 
    индивидуальных особенностей организма», – отмечает профессор Смирнов, руководитель 
    Института питания.
    
    > Внимание: информация носит общий характер и не заменяет консультацию врача.
    
    Источники:
    - WHO Guidelines, 2022
    - Национальные рекомендации по питанию, 2023
    - https://www.nejm.org/nutrition/study12345
    
    Последнее обновление: 15.01.2023
    Медицинский редактор: к.м.н. Петрова А.В.
    """
    
    # Анализ с использованием обученной модели
    print("\nАнализ текста с использованием обученной модели:")
    enhanced_analyzer = EnhancedEEATAnalyzer(model_path=model_path)
    result = enhanced_analyzer.analyze(sample_text, industry='health')
    
    print(f"Общая оценка E-E-A-T: {result['overall_eeat_score']:.2f}")
    print(f"Рекомендации: {len(result['recommendations'])}")
    
    return model_path, result

def main():
    # Запуск тестов
    test_basic_functionality()
    test_model_training()

if __name__ == "__main__":
    main()
