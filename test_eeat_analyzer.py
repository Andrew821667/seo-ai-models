
import sys
sys.path.append('/content/seo-ai-models')

from models.seo_advisor.eeat_analyzer import EEATAnalyzer

def test_eeat_analyzer():
    # Инициализация анализатора
    analyzer = EEATAnalyzer()
    
    # Тестовые примеры текстов для разных отраслей
    test_texts = {
        'general': '''
        Это общий текст о SEO оптимизации. 
        Контент должен быть качественным и полезным для пользователей.
        ''',
        
        'finance': '''
        По данным Центрального Банка на 15.03.2024, средняя ставка по ипотеке составляет 12.5%.
        Наш финансовый аналитик Иван Петров, имеющий 10-летний опыт в банковском секторе, 
        рекомендует обратить внимание на программы с господдержкой.
        
        # Список рекомендуемых банков
        - Альфа-Банк (лицензия ЦБ РФ №1326)
        - ВТБ (лицензия ЦБ РФ №1000)
        
        Источник данных: https://www.cbr.ru/statistics/bank_sector/
        
        Об авторе: Иван Петров - сертифицированный финансовый консультант с дипломом МГУ.
        ''',
        
        'health': '''
        Как снизить давление без лекарств?
        
        Регулярные физические упражнения помогают снизить артериальное давление.
        В исследовании, проведенном в 2023 году, было показано, что 30 минут 
        умеренной активности 5 раз в неделю способны снизить давление на 5-8 мм рт.ст.
        
        Методология: мета-анализ 15 клинических исследований с общим количеством 
        участников более 2500 человек.
        
        Внимание: данная информация носит справочный характер и не заменяет 
        консультацию с врачом. При высоком давлении обратитесь к специалисту.
        '''
    }
    
    # Тестирование для каждой отрасли
    for industry, text in test_texts.items():
        print(f"\n===== Тестирование для отрасли: {industry} =====")
        
        # Используем соответствующую отрасль для YMYL текстов
        analysis_industry = industry
        if industry == 'general':
            analysis_industry = 'default'
            
        # Получаем результаты анализа
        results = analyzer.analyze(text, industry=analysis_industry)
        
        # Выводим основные метрики
        print(f"Оценка экспертизы: {results['expertise_score']:.2f}")
        print(f"Оценка авторитетности: {results['authority_score']:.2f}")
        print(f"Оценка доверия: {results['trust_score']:.2f}")
        print(f"Оценка структуры: {results['structural_score']:.2f}")
        print(f"Общая оценка E-E-A-T: {results['overall_eeat_score']:.2f}")
        
        # Выводим YMYL статус
        print(f"YMYL статус: {'Да' if results['ymyl_status'] else 'Нет'}")
        
        # Выводим рекомендации
        print("\nРекомендации:")
        for rec in results['recommendations']:
            print(f"- {rec}")
        
        # Выводим найденные маркеры (только для первых двух категорий)
        print("\nНайденные маркеры экспертизы:")
        for marker in results['component_details']['expertise']['found_markers'][:5]:
            print(f"- {marker}")
            
        print("\nНайденные маркеры авторитетности:")
        for marker in results['component_details']['authority']['found_markers'][:5]:
            print(f"- {marker}")

if __name__ == "__main__":
    test_eeat_analyzer()
