
import sys
sys.path.append('/content/seo-ai-models')

from models.seo_advisor.eeat_analyzer import EEATAnalyzer
from models.seo_advisor.calibrated_rank_predictor import CalibratedRankPredictor

def demonstrate_eeat_impact():
    # Инициализация компонентов
    eeat_analyzer = EEATAnalyzer()
    
    # Тестовый пример с низким E-E-A-T
    low_eeat_text = '''
    Как инвестировать в акции в 2024 году?
    
    Инвестировать в акции могут все. Сейчас это лучшее время для входа на рынок.
    Вот несколько компаний, которые могут вырасти в цене:
    - Компания A
    - Компания B
    - Компания C
    
    Покупайте эти акции и получайте прибыль!
    '''
    
    # Тестовый пример с высоким E-E-A-T
    high_eeat_text = '''
    # Инвестиционная стратегия на 2024 год: анализ рынка акций
    
    По данным Bloomberg от 15.03.2024, после коррекции в январе 2024 года, 
    рынок акций демонстрирует признаки восстановления. Индекс S&P 500 вырос на 3.5% 
    за последний месяц.
    
    ## Анализ перспективных секторов экономики
    
    Исследование Morgan Stanley от февраля 2024 показывает, что следующие секторы 
    могут продемонстрировать опережающий рост:
    
    1. Информационные технологии (прогнозируемый рост: 12-15%)
    2. Здравоохранение (прогнозируемый рост: 8-10%)
    3. Возобновляемые источники энергии (прогнозируемый рост: 10-14%)
    
    ### Фактор риска
    
    Инвесторам следует помнить о рисках: волатильность рынка остается высокой из-за 
    геополитических факторов и инфляционных ожиданий. Рекомендуется диверсифицировать портфель.
    
    **Источники данных:**
    - Bloomberg Financial Analysis, март 2024
    - Morgan Stanley Global Investment Outlook 2024
    - Financial Times Market Report, февраль 2024
    
    *Методология: анализ основан на агрегации данных из 15 инвестиционных отчетов 
    и экспертной оценке финансовых аналитиков с опытом от 10 лет.*
    
    *Обновлено: 20.03.2024*
    
    **Об авторе:** Анна Петрова, CFA, MBA, 15 лет опыта в инвестиционном банкинге,
    бывший аналитик Goldman Sachs, автор книги "Разумный инвестор в эпоху цифровизации".
    
    *Дисклеймер: Данный материал не является инвестиционной рекомендацией. 
    Инвестиции связаны с риском потери капитала. Прежде чем принимать инвестиционные 
    решения, проконсультируйтесь с финансовым консультантом.*
    '''
    
    # Анализ текстов с низким и высоким E-E-A-T
    low_eeat_analysis = eeat_analyzer.analyze(low_eeat_text, industry='finance')
    high_eeat_analysis = eeat_analyzer.analyze(high_eeat_text, industry='finance')
    
    # Вывод результатов анализа
    print("====== СРАВНЕНИЕ ТЕКСТОВ С РАЗНЫМ УРОВНЕМ E-E-A-T ======")
    print("\n=== Текст с низким E-E-A-T ===")
    print(f"Общая оценка E-E-A-T: {low_eeat_analysis['overall_eeat_score']:.2f}")
    print("Компоненты:")
    print(f"- Экспертиза: {low_eeat_analysis['expertise_score']:.2f}")
    print(f"- Авторитетность: {low_eeat_analysis['authority_score']:.2f}")
    print(f"- Доверие: {low_eeat_analysis['trust_score']:.2f}")
    print(f"- Структура: {low_eeat_analysis['structural_score']:.2f}")
    
    print("\n=== Текст с высоким E-E-A-T ===")
    print(f"Общая оценка E-E-A-T: {high_eeat_analysis['overall_eeat_score']:.2f}")
    print("Компоненты:")
    print(f"- Экспертиза: {high_eeat_analysis['expertise_score']:.2f}")
    print(f"- Авторитетность: {high_eeat_analysis['authority_score']:.2f}")
    print(f"- Доверие: {high_eeat_analysis['trust_score']:.2f}")
    print(f"- Структура: {high_eeat_analysis['structural_score']:.2f}")
    
    # Демонстрация влияния E-E-A-T на ранжирование
    print("\n====== ВЛИЯНИЕ E-E-A-T НА РАНЖИРОВАНИЕ ======")
    
    # Подготовка базовых метрик для обоих текстов (одинаковые для изоляции влияния E-E-A-T)
    base_features = {
        'keyword_density': 0.02,
        'content_length': 500,
        'readability_score': 70,
        'meta_tags_score': 0.7,
        'header_structure_score': 0.7,
        'multimedia_score': 0.5,
        'internal_linking_score': 0.5,
        'topic_relevance': 0.7,
        'semantic_depth': 0.7,
        'engagement_potential': 0.6
    }
    
    # Добавляем E-E-A-T метрики для текста с низким E-E-A-T
    low_eeat_features = base_features.copy()
    low_eeat_features.update({
        'expertise_score': low_eeat_analysis['expertise_score'],
        'authority_score': low_eeat_analysis['authority_score'],
        'trust_score': low_eeat_analysis['trust_score'],
        'overall_eeat_score': low_eeat_analysis['overall_eeat_score']
    })
    
    # Добавляем E-E-A-T метрики для текста с высоким E-E-A-T
    high_eeat_features = base_features.copy()
    high_eeat_features.update({
        'expertise_score': high_eeat_analysis['expertise_score'],
        'authority_score': high_eeat_analysis['authority_score'],
        'trust_score': high_eeat_analysis['trust_score'],
        'overall_eeat_score': high_eeat_analysis['overall_eeat_score']
    })
    
    # Инициализация предикторов для разных отраслей
    predictors = {
        'finance': CalibratedRankPredictor(industry='finance'),  # YMYL отрасль
        'default': CalibratedRankPredictor(industry='default')  # не-YMYL отрасль
    }
    
    # Оценка позиций для разных отраслей
    for industry, predictor in predictors.items():
        print(f"\n== Отрасль: {industry} ==")
        
        # Предсказание для текста с низким E-E-A-T
        low_prediction = predictor.predict_position(low_eeat_features)
        print(f"Позиция текста с низким E-E-A-T: {low_prediction['position']:.1f}")
        
        # Предсказание для текста с высоким E-E-A-T
        high_prediction = predictor.predict_position(high_eeat_features)
        print(f"Позиция текста с высоким E-E-A-T: {high_prediction['position']:.1f}")
        
        # Влияние E-E-A-T
        if 'eeat_position_improvement' in high_prediction:
            print(f"Улучшение позиции благодаря E-E-A-T: {high_prediction['eeat_position_improvement']:.1f} пунктов")
        
        # Разница между текстами
        print(f"Разница в позициях: {low_prediction['position'] - high_prediction['position']:.1f} пунктов")
        print(f"Процентное улучшение: {((low_prediction['position'] - high_prediction['position']) / low_prediction['position'] * 100):.1f}%")

if __name__ == "__main__":
    demonstrate_eeat_impact()
