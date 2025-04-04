import sys
import os
from pprint import pprint

# Добавляем корневую директорию в путь поиска модулей
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from seo_ai_models.models.seo_advisor.analyzers.eeat.eeat_analyzer import EEATAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.eeat.enhanced_eeat_analyzer import EnhancedEEATAnalyzer

def test_enhanced_eeat_analyzer():
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ENHANCEDEEATANALYZER С ML-МОДЕЛЬЮ")
    print("=" * 80)
    
    # Создаем экземпляры анализаторов
    base_analyzer = EEATAnalyzer()
    enhanced_analyzer = EnhancedEEATAnalyzer()  # Без указания пути - должен найти модель автоматически
    
    # Проверяем статус загрузки модели
    print(f"\n>> Статус ML-модели:")
    print(f"Модель загружена: {enhanced_analyzer.ml_model_used}")
    print(f"Тип модели: {type(enhanced_analyzer.model).__name__ if enhanced_analyzer.model else 'None'}")
    
    # Тестовые контенты с разными уровнями E-E-A-T
    high_eeat_content = """
    # Исследование влияния физической активности на когнитивные функции
    
    ## Введение
    
    В данной статье представлены результаты исследования, проведенного в 2024 году 
    Международной ассоциацией неврологии, по изучению влияния регулярной 
    физической активности на когнитивные функции у людей разных возрастных групп.
    
    ## Методология исследования
    
    В исследовании приняли участие 1500 человек в возрасте от 25 до 75 лет. 
    Участники были разделены на три группы в зависимости от уровня физической
    активности:
    
    * Группа 1: низкая активность (менее 2 часов в неделю)
    * Группа 2: средняя активность (2-5 часов в неделю)
    * Группа 3: высокая активность (более 5 часов в неделю)
    
    Когнитивные функции оценивались с помощью стандартизированных тестов: 
    Montreal Cognitive Assessment (MoCA) и Trail Making Test (TMT).
    
    ## Результаты и обсуждение
    
    Согласно полученным данным, участники из группы с высокой физической 
    активностью продемонстрировали результаты в среднем на 28% лучше, чем
    участники с низкой активностью. Статистический анализ показал значимую
    корреляцию (p < 0.001) между количеством часов физической активности
    и показателями когнитивных функций.
    
    Профессор нейробиологии Джеймс Картер отмечает: "Наши исследования
    подтверждают, что регулярная физическая активность стимулирует
    нейрогенез и улучшает пластичность мозга, что напрямую влияет
    на когнитивные способности".
    
    ## Выводы
    
    На основании проведенного исследования можно сделать следующие выводы:
    
    1. Регулярная физическая активность положительно влияет на когнитивные функции
    2. Оптимальный уровень активности составляет 3-5 часов в неделю
    3. Эффект наблюдается во всех возрастных группах, но наиболее выражен у людей старше 50 лет
    
    ## Источники
    
    1. Carter, J., et al. (2024). Physical Activity and Cognitive Function. Journal of Neuroscience, 45(3), 112-128.
    2. International Neurological Association. (2024). Guidelines for Brain Health Maintenance.
    3. Smith, A., & Johnson, B. (2023). Exercise and Brain Plasticity. Cognitive Science Review, 12(2), 45-60.
    
    Авторы: Доктор медицинских наук Александр Петров, Профессор нейробиологии Мария Смирнова
    
    Дата публикации: 15 февраля 2025 г.
    Последнее обновление: 20 марта 2025 г.
    
    Отказ от ответственности: Данная информация предназначена только для образовательных целей 
    и не заменяет консультацию с медицинским специалистом.
    """
    
    medium_eeat_content = """
    # Физическая активность и когнитивные функции
    
    ## Введение
    
    Физическая активность может положительно влиять на работу мозга и когнитивные способности.
    В этой статье мы рассмотрим связь между регулярными упражнениями и улучшением памяти,
    внимания и других когнитивных функций.
    
    ## Как физическая активность влияет на мозг
    
    Регулярные физические упражнения улучшают кровообращение, в том числе в головном мозге.
    Это способствует лучшему питанию нейронов и может стимулировать образование новых
    нервных клеток.
    
    ## Рекомендации по физической активности
    
    * Старайтесь заниматься спортом не менее 3 раз в неделю
    * Комбинируйте кардио и силовые тренировки
    * Начинайте с небольших нагрузок, постепенно увеличивая их
    
    ## Заключение
    
    Физическая активность — важный компонент поддержания здоровья мозга. Регулярные упражнения
    могут помочь улучшить когнитивные функции и снизить риск когнитивных нарушений в будущем.
    
    Источники:
    1. Журнал "Здоровье и спорт", 2023
    2. Рекомендации Всемирной организации здравоохранения
    
    Автор: Иван Петров, фитнес-тренер
    """
    
    low_eeat_content = """
    Как улучшить работу мозга с помощью спорта
    
    Занятия спортом полезны для мозга. Если вы будете регулярно заниматься,
    ваши когнитивные способности улучшатся.
    
    Лучшие виды спорта для мозга:
    - Бег
    - Плавание
    - Йога
    
    Занимайтесь не менее 30 минут в день, и вы заметите улучшение памяти и внимания.
    
    Не забывайте также о правильном питании и режиме дня.
    """
    
    # Тестируем анализаторы на контентах разного качества
    industries = ["health", "blog"]
    contents = {
        "Высокий E-E-A-T": high_eeat_content,
        "Средний E-E-A-T": medium_eeat_content,
        "Низкий E-E-A-T": low_eeat_content
    }
    
    for industry in industries:
        print(f"\n>> Анализ для отрасли '{industry}':")
        
        for content_type, content in contents.items():
            print(f"\n### {content_type} контент:")
            
            # Базовый анализ
            base_result = base_analyzer.analyze(content, industry=industry)
            base_score = base_result['overall_eeat_score']
            
            # Улучшенный анализ
            enhanced_result = enhanced_analyzer.analyze(content, industry=industry)
            enhanced_score = enhanced_result['overall_eeat_score']
            
            # Сравнение результатов
            print(f"Базовый анализатор - E-E-A-T оценка: {base_score:.2f}")
            print(f"Улучшенный анализатор - E-E-A-T оценка: {enhanced_score:.2f}")
            
            if enhanced_result.get('ml_model_used', False):
                print(f"Разница оценок (ML - базовая): {enhanced_result.get('ml_score_difference', 0):.2f}")
            
            # Вывод рекомендаций от ML-модели
            if enhanced_result.get('ml_model_used', False):
                ml_recs = [rec for rec in enhanced_result['recommendations'] if 'ML-анализ' in rec]
                if ml_recs:
                    print("\nРекомендации от ML-модели:")
                    for rec in ml_recs:
                        print(f"- {rec}")
    
    print("\n>> Детальный анализ высококачественного контента (отрасль health):")
    detailed = enhanced_analyzer.analyze(high_eeat_content, industry="health")
    
    # Выводим компонентные оценки
    print("\nКомпонентные оценки:")
    for component in ['experience_score', 'expertise_score', 'authority_score', 'trust_score', 'structural_score']:
        print(f"{component}: {detailed[component]:.2f}")
    
    # Выводим найденные маркеры
    print("\nНайденные маркеры авторитетности:")
    authority_markers = detailed['component_details']['authority']['found_markers']
    for marker, count in list(authority_markers.items())[:5]:
        print(f"- '{marker}': {count} раз")

if __name__ == "__main__":
    test_enhanced_eeat_analyzer()
