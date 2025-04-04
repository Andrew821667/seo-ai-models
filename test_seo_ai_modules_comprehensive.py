
import sys
import os
from pprint import pprint
import time
from datetime import datetime

# Добавляем корневую директорию в путь поиска модулей
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Импортируем все основные компоненты
from seo_ai_models.models.seo_advisor.analyzers.content_analyzer import ContentAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.semantic_analyzer import SemanticAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.eeat.eeat_analyzer import EEATAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.eeat.enhanced_eeat_analyzer import EnhancedEEATAnalyzer
from seo_ai_models.models.seo_advisor.predictors.calibrated_rank_predictor import CalibratedRankPredictor
from seo_ai_models.models.seo_advisor.suggester.suggester import Suggester
from seo_ai_models.models.seo_advisor.advisor import SEOAdvisor
from seo_ai_models.common.utils.text_processing import TextProcessor

def create_test_contents():
    """Создаем тестовые данные для разных отраслей и разного качества"""
    
    # Высококачественный контент для финансовой отрасли
    finance_good = """
    # Руководство по инвестированию в индексные фонды на 2025 год
    
    ## Введение
    
    В данной статье мы рассмотрим основные стратегии инвестирования в индексные фонды, 
    которые актуальны в 2025 году. Я лично практикую эти стратегии уже более 10 лет и
    хочу поделиться своим опытом с вами.
    
    ## Что такое индексные фонды?
    
    Индексные фонды - это пассивные инвестиционные инструменты, которые отслеживают 
    определенный индекс, например, S&P 500. Согласно исследованию Vanguard Group, 
    проведенному в 2024 году, индексные фонды продолжают превосходить активно управляемые 
    фонды на долгосрочном горизонте.
    
    ### Преимущества индексных фондов:
    
    * Низкие комиссии (обычно 0.03-0.25%)
    * Широкая диверсификация
    * Высокая ликвидность
    * Прозрачность
    * Налоговая эффективность
    
    ## Лучшие индексные фонды 2025 года
    
    Профессор финансов Джереми Сигел из Уортонской школы бизнеса рекомендует следующие индексные фонды:
    
    1. Vanguard Total Market ETF (VTI)
    2. iShares Core S&P 500 ETF (IVV)
    3. Schwab International Index ETF (SCHF)
    
    ## Стратегии инвестирования
    
    На основе данных Morningstar за 2024 год, оптимальная стратегия включает:
    
    1. Регулярные инвестиции фиксированной суммы (dollar-cost averaging)
    2. Ребалансировка портфеля каждые 6-12 месяцев
    3. Увеличение доли облигаций с возрастом инвестора
    
    ## Заключение
    
    Индексные фонды остаются отличным выбором для большинства инвесторов в 2025 году. 
    При правильной стратегии они обеспечивают стабильный рост капитала при минимальных затратах.
    
    ## Источники
    
    1. Vanguard Research, "Индексное инвестирование в 2024-2025 годах", опубликовано 12 января 2025
    2. Morningstar Direct, "Анализ эффективности инвестиционных стратегий", март 2025
    3. Siegel, J. (2024). "Инвестирование в долгосрочной перспективе", 7-е издание
    4. https://www.investopedia.com/best-index-funds-5092409
    
    Автор: Финансовый аналитик, CFA, с 12-летним опытом работы в инвестиционной сфере.
    
    Обновлено: 15 марта 2025 года
    
    Отказ от ответственности: Данная информация не является инвестиционной рекомендацией.
    """
    
    # Низкокачественный контент для финансовой отрасли
    finance_poor = """
    Как инвестировать деньги
    
    Многие люди хотят инвестировать, но не знают как.
    Индексные фонды - это хороший способ вложения денег.
    
    Вот несколько индексных фондов:
    - VTI
    - IVV
    - SCHF
    
    Инвестируйте регулярно и на долгий срок.
    
    Удачных инвестиций!
    """
    
    # Высококачественный контент для медицинской отрасли
    health_good = """
    # Профилактика сердечно-сосудистых заболеваний: руководство на основе данных
    
    ## Введение
    
    Сердечно-сосудистые заболевания (ССЗ) остаются ведущей причиной смертности во всем мире. 
    В данной статье мы рассмотрим научно обоснованные методы профилактики ССЗ, 
    опираясь на последние исследования и клинические рекомендации.
    
    ## Факторы риска
    
    Согласно данным Всемирной организации здравоохранения (2024), основными факторами 
    риска развития ССЗ являются:
    
    * Артериальная гипертензия
    * Дислипидемия
    * Курение
    * Сахарный диабет
    * Избыточная масса тела
    * Низкая физическая активность
    * Неправильное питание
    
    ## Научно обоснованные методы профилактики
    
    ### Диета
    
    Мета-анализ, опубликованный в The New England Journal of Medicine в январе 2025 года, 
    подтвердил эффективность средиземноморской диеты для снижения риска ССЗ на 25-30%. 
    Ключевые элементы этой диеты:
    
    1. Обилие фруктов и овощей (не менее 5 порций в день)
    2. Цельнозерновые продукты
    3. Замена насыщенных жиров на ненасыщенные (оливковое масло)
    4. Умеренное потребление рыбы и морепродуктов
    5. Ограничение красного мяса
    
    ### Физическая активность
    
    Исследование, проведенное группой профессора Дженкинса в Стэнфордском университете, 
    показало, что даже 150 минут умеренной физической активности в неделю снижает 
    риск развития ССЗ на 20%. Рекомендуемые виды активности:
    
    * Ходьба быстрым шагом
    * Езда на велосипеде
    * Плавание
    * Танцы
    
    ## Заключение
    
    Профилактика сердечно-сосудистых заболеваний должна быть комплексной и включать 
    как модификацию образа жизни, так и медикаментозную терапию при наличии показаний. 
    Регулярное наблюдение у врача и контроль факторов риска позволяют значительно 
    снизить вероятность развития ССЗ.
    
    ## Источники
    
    1. World Health Organization. (2024). Cardiovascular diseases fact sheet.
    2. Jenkins, D. et al. (2025). Physical activity and cardiovascular health. Stanford Medical Review, 87(2), 112-128.
    3. American Heart Association. (2024). Guidelines for the Primary Prevention of Cardiovascular Disease.
    4. Martinez-Gonzalez, M. A. et al. (2025). Mediterranean diet and cardiovascular outcomes. The New England Journal of Medicine, 382(1), 25-35.
    5. https://www.heart.org/en/health-topics/consumer-healthcare/prevention
    
    Автор: Доктор медицинских наук, кардиолог с 15-летним стажем работы
    
    Обновлено: 20 февраля 2025 года
    
    Отказ от ответственности: Данная информация не заменяет консультацию врача.
    """
    
    # Низкокачественный контент для медицинской отрасли
    health_poor = """
    Как сохранить здоровое сердце
    
    Сердечные заболевания очень опасны. Вот способы сохранить сердце здоровым:
    
    - Правильно питайтесь
    - Больше двигайтесь
    - Не курите
    - Контролируйте вес
    
    Также полезно есть больше фруктов и овощей. Старайтесь избегать жирной пищи.
    
    Занимайтесь спортом хотя бы 3 раза в неделю.
    
    Берегите свое сердце!
    """
    
    # Контент для туристической отрасли
    travel_content = """
    # 10 удивительных мест для посещения в 2025 году
    
    ## Введение в мир путешествий 2025 года
    
    Путешествия в 2025 году становятся все более экологичными и технологичными. Я путешествую 
    профессионально уже более 8 лет и посетил более 50 стран. В этой статье я делюсь своими 
    находками и рекомендациями на основе личного опыта и последних трендов.
    
    ## Топ-10 направлений 2025 года
    
    ### 1. Новая Зеландия: экотуризм будущего
    
    Новая Зеландия продолжает лидировать в экотуризме. В своей поездке в апреле 2024 года 
    я был поражен новыми экологическими инициативами в регионе Фьордленд. 
    
    Рекомендую посетить:
    * Национальный парк Тонгариро
    * Фьорд Милфорд-Саунд
    * Долину гейзеров Роторуа
    
    ### 2. Японская префектура Исикава
    
    После восстановления от землетрясений регион Исикава снова принимает туристов. 
    Местные жители рассказали мне, что число посетителей до сих пор меньше, чем до 
    пандемии, что делает это идеальным временем для посещения.
    
    ## Практические советы для путешественников в 2025 году
    
    По данным Всемирной туристической организации, в 2025 году ожидается увеличение 
    стоимости перелетов на 15%. Рекомендую:
    
    1. Бронировать авиабилеты за 4-6 месяцев
    2. Использовать новые мобильные приложения для отслеживания цен
    3. Рассмотреть альтернативные аэропорты и гибкие даты
    
    ## Заключение
    
    2025 год предлагает уникальные возможности для путешествий. Сочетание технологий, 
    устойчивого развития и жажды аутентичных впечатлений формирует новую эру туризма.
    
    ## Источники
    
    1. World Tourism Organization (2024). "Tourism Trends Forecast 2025-2030"
    2. Personal travel experiences (2015-2025)
    3. https://www.nationalgeographic.com/travel/article/best-destinations-2025
    
    Обновлено: Январь 2025 года
    
    Автор: Профессиональный путешественник и блогер
    """
    
    # Контент для блога о технологиях
    tech_blog_content = """
    # Обзор новейших тенденций в разработке ИИ на 2025 год
    
    Искусственный интеллект продолжает развиваться невероятными темпами. 
    В этой статье я расскажу о главных трендах ИИ в 2025 году, основываясь 
    на моем 10-летнем опыте работы в этой сфере.
    
    ## 1. Многомодальные модели выходят на новый уровень
    
    По сравнению с моделями 2023-2024 годов, новые многомодальные системы демонстрируют 
    значительное улучшение в понимании контекста между различными типами данных. 
    
    Ключевые достижения:
    * Интеграция до 7 модальностей в единой системе
    * Снижение вычислительных требований на 30%
    * Повышение точности на сложных задачах до 95%
    
    ## 2. Децентрализованные ИИ-системы
    
    Согласно исследованию MIT Technology Review, почти 40% новых ИИ-решений 
    в 2025 году используют децентрализованные архитектуры. Я недавно тестировал 
    несколько таких систем и могу подтвердить их преимущества:
    
    1. Повышенная конфиденциальность данных
    2. Устойчивость к сбоям
    3. Снижение затрат на инфраструктуру
    
    Код для развертывания простой децентрализованной системы:
    
    ```python
    import decentralized_ai as dai
    
    # Инициализация узлов
    nodes = dai.create_network(nodes=5, topology="mesh")
    
    # Настройка модели
    model = dai.load_model("transformer_v3")
    
    # Распределение вычислений
    result = nodes.distribute_inference(model, input_data)
    ```
    
    ## Заключение
    
    Развитие ИИ в 2025 году фокусируется на эффективности, конфиденциальности и 
    доступности. Эти тенденции формируют новое поколение решений, которые 
    будут определять будущее технологий на годы вперед.
    
    ## Источники
    
    1. MIT Technology Review (2025). "The State of AI in 2025"
    2. OpenAI Research Papers (2024-2025)
    3. https://github.com/decentralized-ai/examples
    
    Автор: Технический директор AI-стартапа, PhD в области компьютерных наук
    """
    
    return {
        "finance_good": finance_good,
        "finance_poor": finance_poor,
        "health_good": health_good,
        "health_poor": health_poor,
        "travel": travel_content,
        "tech_blog": tech_blog_content
    }

def print_section_header(title):
    """Вспомогательная функция для вывода заголовков разделов"""
    print("\n" + "=" * 80)
    print(f"{title.upper()}")
    print("=" * 80)

def test_text_processor():
    """Тестирование TextProcessor"""
    print_section_header("Тестирование TextProcessor")
    
    processor = TextProcessor()
    test_contents = create_test_contents()
    content = test_contents["finance_good"]
    
    print("\n> Определение языка:")
    language = processor.detect_language(content)
    print(f"Определенный язык: {language}")
    
    print("\n> Токенизация:")
    tokens = processor.tokenize(content[:200], remove_stopwords=True)
    print(f"Токены (первые 10): {tokens[:10]}")
    
    print("\n> Извлечение заголовков:")
    headers = processor.extract_headers(content)
    print(f"Количество заголовков: {len(headers)}")
    for i, header in enumerate(headers[:3], 1):
        print(f"  Заголовок {i}: Уровень {header['level']} - {header['text']}")
    
    print("\n> Анализ структуры текста:")
    structure = processor.analyze_text_structure(content)
    print(f"Количество абзацев: {structure['paragraphs_count']}")
    print(f"Количество заголовков: {structure['headers_count']}")
    print(f"Наличие введения: {structure['has_introduction']}")
    print(f"Наличие заключения: {structure['has_conclusion']}")
    
    print("\n> Извлечение ключевых слов:")
    keywords = processor.extract_keywords(content, max_keywords=5)
    print(f"Топ-5 ключевых слов:")
    for keyword, weight in keywords:
        print(f"  {keyword}: {weight:.2f}")
    
    print("\n> Расчет читабельности:")
    readability = processor.calculate_readability(content)
    print(f"Flesch Reading Ease: {readability['flesch_reading_ease']:.2f}")

def test_content_analyzer():
    """Тестирование ContentAnalyzer"""
    print_section_header("Тестирование ContentAnalyzer")
    
    analyzer = ContentAnalyzer()
    test_contents = create_test_contents()
    
    # Анализируем качественный и некачественный контент
    content_good = test_contents["finance_good"]
    content_poor = test_contents["finance_poor"]
    
    print("\n> Базовые метрики (качественный контент):")
    metrics_good = analyzer.analyze_text(content_good)
    print(f"Количество слов: {metrics_good['word_count']}")
    print(f"Читабельность: {metrics_good['readability']:.2f}")
    print(f"Оценка заголовков: {metrics_good['header_score']:.2f}")
    
    print("\n> Базовые метрики (некачественный контент):")
    metrics_poor = analyzer.analyze_text(content_poor)
    print(f"Количество слов: {metrics_poor['word_count']}")
    print(f"Читабельность: {metrics_poor['readability']:.2f}")
    print(f"Оценка заголовков: {metrics_poor['header_score']:.2f}")
    
    # Анализ ключевых слов
    keywords_to_analyze = ["инвестирование", "фонд", "деньги", "стратегия"]
    
    print("\n> Анализ ключевых слов (качественный контент):")
    keyword_stats_good = analyzer.extract_keywords(content_good, keywords_to_analyze)
    print(f"Плотность ключевых слов: {keyword_stats_good['density']:.4f}")
    print("Частота ключевых слов:")
    for keyword, freq in keyword_stats_good['frequency'].items():
        print(f"  {keyword}: {freq}")
    
    print("\n> Анализ ключевых слов (некачественный контент):")
    keyword_stats_poor = analyzer.extract_keywords(content_poor, keywords_to_analyze)
    print(f"Плотность ключевых слов: {keyword_stats_poor['density']:.4f}")
    print("Частота ключевых слов:")
    for keyword, freq in keyword_stats_poor['frequency'].items():
        print(f"  {keyword}: {freq}")

def test_semantic_analyzer():
    """Тестирование SemanticAnalyzer"""
    print_section_header("Тестирование SemanticAnalyzer")
    
    analyzer = SemanticAnalyzer()
    test_contents = create_test_contents()
    
    # Тестовые ключевые слова
    finance_keywords = ["инвестирование", "фонд", "деньги", "стратегия", "капитал"]
    health_keywords = ["здоровье", "профилактика", "болезнь", "риск", "питание"]
    
    # Анализируем контент из разных отраслей
    print("\n> Семантический анализ (финансы, качественный):")
    finance_results = analyzer.analyze_text(test_contents["finance_good"], finance_keywords)
    print(f"Семантическая плотность: {finance_results['semantic_density']:.2f}")
    print(f"Семантическое покрытие: {finance_results['semantic_coverage']:.2f}")
    print(f"Тематическая связность: {finance_results['topical_coherence']:.2f}")
    print(f"Контекстуальная релевантность: {finance_results['contextual_relevance']:.2f}")
    
    print("\n> Семантический анализ (медицина, качественный):")
    health_results = analyzer.analyze_text(test_contents["health_good"], health_keywords)
    print(f"Семантическая плотность: {health_results['semantic_density']:.2f}")
    print(f"Семантическое покрытие: {health_results['semantic_coverage']:.2f}")
    print(f"Тематическая связность: {health_results['topical_coherence']:.2f}")
    print(f"Контекстуальная релевантность: {health_results['contextual_relevance']:.2f}")
    
    print("\n> Рекомендации (финансы):")
    finance_recommendations = analyzer.generate_recommendations(finance_results)
    for i, rec in enumerate(finance_recommendations[:3], 1):
        print(f"  {i}. {rec}")
    
    print("\n> Семантические поля для ключевых слов (финансы):")
    for keyword, terms in list(finance_results['semantic_fields'].items())[:2]:
        print(f"  '{keyword}': {', '.join(terms[:5])}")

def test_eeat_analyzer():
    """Тестирование EEATAnalyzer"""
    print_section_header("Тестирование EEATAnalyzer")
    
    analyzer = EEATAnalyzer()
    enhanced_analyzer = EnhancedEEATAnalyzer()
    test_contents = create_test_contents()
    
    # Тестирование для разных отраслей
    industries = {
        "finance": ("Финансы (YMYL)", test_contents["finance_good"]),
        "health": ("Здравоохранение (YMYL)", test_contents["health_good"]),
        "travel": ("Туризм (не YMYL)", test_contents["travel"]),
        "tech_blog": ("Блог о технологиях (не YMYL)", test_contents["tech_blog"])
    }
    
    for industry_key, (industry_name, content) in industries.items():
        print(f"\n> E-E-A-T анализ для отрасли '{industry_name}':")
        result = analyzer.analyze(content, industry=industry_key)
        
        print(f"Общая оценка E-E-A-T: {result['overall_eeat_score']:.2f}")
        print(f"Опыт (Experience): {result['experience_score']:.2f}")
        print(f"Экспертиза (Expertise): {result['expertise_score']:.2f}")
        print(f"Авторитетность (Authoritativeness): {result['authority_score']:.2f}")
        print(f"Доверие (Trustworthiness): {result['trust_score']:.2f}")
        print(f"YMYL статус: {'Да' if result['ymyl_status'] == 1 else 'Нет'}")
        
        print("\nРекомендации:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Сравнение базового и улучшенного анализаторов
    print("\n> Сравнение базового и улучшенного E-E-A-T анализаторов:")
    health_content = test_contents["health_good"]
    base_result = analyzer.analyze(health_content, industry="health")
    enhanced_result = enhanced_analyzer.analyze(health_content, industry="health")
    
    print(f"Базовый анализатор - общая оценка: {base_result['overall_eeat_score']:.2f}")
    print(f"Улучшенный анализатор - общая оценка: {enhanced_result['overall_eeat_score']:.2f}")
    print(f"Использована ML-модель: {enhanced_result.get('ml_model_used', False)}")

def test_rank_predictor():
    """Тестирование CalibratedRankPredictor"""
    print_section_header("Тестирование CalibratedRankPredictor")
    
    predictor = CalibratedRankPredictor()
    
    # Тестовые наборы признаков для разных отраслей
    features_finance_good = {
        'keyword_density': 0.025,
        'content_length': 1800,
        'readability_score': 75,
        'meta_tags_score': 0.85,
        'header_structure_score': 0.90,
        'multimedia_score': 0.60,
        'internal_linking_score': 0.70,
        'topic_relevance': 0.88,
        'semantic_depth': 0.85,
        'engagement_potential': 0.80,
        'expertise_score': 0.85,
        'authority_score': 0.80,
        'trust_score': 0.90,
        'overall_eeat_score': 0.85
    }
    
    features_finance_poor = {
        'keyword_density': 0.015,
        'content_length': 300,
        'readability_score': 60,
        'meta_tags_score': 0.30,
        'header_structure_score': 0.20,
        'multimedia_score': 0.10,
        'internal_linking_score': 0.15,
        'topic_relevance': 0.40,
        'semantic_depth': 0.30,
        'engagement_potential': 0.25,
        'expertise_score': 0.30,
        'authority_score': 0.25,
        'trust_score': 0.20,
        'overall_eeat_score': 0.25
    }
    
    features_health = {
        'keyword_density': 0.022,
        'content_length': 2000,
        'readability_score': 70,
        'meta_tags_score': 0.80,
        'header_structure_score': 0.85,
        'multimedia_score': 0.70,
        'internal_linking_score': 0.65,
        'topic_relevance': 0.90,
        'semantic_depth': 0.80,
        'engagement_potential': 0.75,
        'expertise_score': 0.90,
        'authority_score': 0.85,
        'trust_score': 0.95,
        'overall_eeat_score': 0.90
    }
    
    # Тестирование предсказания позиций для разных отраслей и качества
    print("\n> Предсказание позиций:")
    
    # Финансы, хороший контент
    predictor_finance = CalibratedRankPredictor(industry="finance")
    result_finance_good = predictor_finance.predict_position(features_finance_good)
    print(f"Финансы (качественный) - предсказанная позиция: {result_finance_good['position']:.2f}")
    
    # Финансы, плохой контент
    result_finance_poor = predictor_finance.predict_position(features_finance_poor)
    print(f"Финансы (некачественный) - предсказанная позиция: {result_finance_poor['position']:.2f}")
    
    # Здравоохранение
    predictor_health = CalibratedRankPredictor(industry="health")
    result_health = predictor_health.predict_position(features_health)
    print(f"Здравоохранение - предсказанная позиция: {result_health['position']:.2f}")
    
    # Генерация рекомендаций
    print("\n> Генерация рекомендаций для некачественного контента:")
    recommendations = predictor_finance.generate_recommendations(features_finance_poor)
    
    for category, recs in list(recommendations.items())[:3]:
        print(f"\nКатегория: {category}")
        for i, rec in enumerate(recs[:3], 1):
            print(f"  {i}. {rec}")

def test_suggester():
    """Тестирование Suggester"""
    print_section_header("Тестирование Suggester")
    
    suggester = Suggester()
    
    # Тестовые рекомендации и оценки
    base_recommendations = {
        'content_length': [
            'Увеличьте объем контента до 1500 слов',
            'Добавьте больше примеров и данных исследований'
        ],
        'keyword_density': [
            'Увеличьте плотность ключевых слов',
            'Включите больше LSI-ключевых слов'
        ],
        'readability': [
            'Структурируйте текст на более короткие абзацы',
            'Добавьте больше подзаголовков'
        ]
    }
    
    feature_scores = {
        'content_length': 0.3,
        'keyword_density': 0.4,
        'readability': 0.7,
        'header_structure': 0.6,
        'meta_tags': 0.5,
        'multimedia': 0.2,
        'internal_linking': 0.3
    }
    
    weighted_scores = {
        'content_length': 0.075,
        'keyword_density': 0.048,
        'readability': 0.091,
        'header_structure': 0.072,
        'meta_tags': 0.035,
        'multimedia': 0.012,
        'internal_linking': 0.018
    }
    
    # Тестирование для разных отраслей
    industries = ["blog", "ecommerce", "finance", "health"]
    
    for industry in industries:
        print(f"\n> Генерация рекомендаций для отрасли '{industry}':")
        suggestions = suggester.generate_suggestions(
            base_recommendations,
            feature_scores,
            industry
        )
        
        # Вывод рекомендаций
        for category, recs in list(suggestions.items())[:2]:
            print(f"\nКатегория: {category}")
            for i, rec in enumerate(recs[:3], 1):
                print(f"  {i}. {rec}")
    
    # Тестирование приоритизации задач
    print("\n> Приоритизация задач:")
    priorities = suggester.prioritize_tasks(
        base_recommendations,
        feature_scores,
        weighted_scores
    )
    
    # Вывод приоритетов
    for i, task in enumerate(priorities[:5], 1):
        print(f"{i}. Задача: {task['task']}")
        print(f"   - Влияние: {task['impact']:.2f}")
        print(f"   - Сложность: {task['effort']:.2f}")
        print(f"   - Приоритет: {task['priority_score']:.2f}")

def test_seo_advisor():
    """Тестирование полного цикла SEOAdvisor"""
    print_section_header("Тестирование SEOAdvisor (полный цикл)")
    
    test_contents = create_test_contents()
    
    # Создаем советников для разных отраслей
    finance_advisor = SEOAdvisor(industry="finance")
    health_advisor = SEOAdvisor(industry="health")
    
    # Тестовые ключевые слова
    finance_keywords = ["инвестирование", "фонд", "деньги", "стратегия", "капитал"]
    health_keywords = ["здоровье", "профилактика", "болезнь", "риск", "питание"]
    
    # Замеряем время анализа
    print("\n> Анализ финансового контента (качественный):")
    start_time = time.time()
    
    finance_result = finance_advisor.analyze_content(
        test_contents["finance_good"],
        finance_keywords
    )
    
    elapsed = time.time() - start_time
    print(f"Время анализа: {elapsed:.2f} секунд")
    
    print(f"\nПредсказанная позиция: {finance_result.predicted_position:.2f}")
    print(f"Общая оценка E-E-A-T: {finance_result.content_metrics['overall_eeat_score']:.2f}")
    print(f"Анализ ключевых слов:")
    print(f"  - Плотность: {finance_result.keyword_analysis['density']:.4f}")
    
    # Вывод метрик контента
    print("\nКлючевые метрики контента:")
    key_metrics = ['word_count', 'readability', 'header_score', 'semantic_coverage', 'topic_relevance']
    for metric in key_metrics:
        print(f"  - {metric}: {finance_result.content_metrics.get(metric, 'Н/Д')}")
    
    # Вывод рекомендаций
    print("\nТоп рекомендации:")
    for category, recommendations in list(finance_result.recommendations.items())[:3]:
        print(f"  Категория: {category}")
        for i, rec in enumerate(recommendations[:2], 1):
            print(f"    {i}. {rec}")
    
    # Вывод приоритетов
    print("\nПриоритетные задачи:")
    for i, task in enumerate(finance_result.priorities[:3], 1):
        print(f"  {i}. {task['task']} (Приоритет: {task['priority_score']:.2f})")
    
    # Сравнение качественного и некачественного контента
    print("\n> Сравнение качественного и некачественного контента (здравоохранение):")
    
    good_result = health_advisor.analyze_content(
        test_contents["health_good"],
        health_keywords
    )
    
    poor_result = health_advisor.analyze_content(
        test_contents["health_poor"],
        health_keywords
    )
    
    print(f"Качественный контент - предсказанная позиция: {good_result.predicted_position:.2f}")
    print(f"Некачественный контент - предсказанная позиция: {poor_result.predicted_position:.2f}")
    
    print("\nРазница в ключевых метриках:")
    for metric in ['word_count', 'readability', 'overall_eeat_score', 'topic_relevance']:
        good_value = good_result.content_metrics.get(metric, 0)
        poor_value = poor_result.content_metrics.get(metric, 0)
        if isinstance(good_value, (int, float)) and isinstance(poor_value, (int, float)):
            diff = good_value - poor_value
            print(f"  - {metric}: {diff:.2f} ({poor_value:.2f} → {good_value:.2f})")
    
    # Проверка истории анализов
    print("\n> История анализов:")
    print(f"Количество записей в истории: {len(finance_advisor.analysis_history)}")
    if finance_advisor.analysis_history:
        last_record = finance_advisor.analysis_history[-1]
        print(f"Последний анализ: {last_record['timestamp']}")
        print(f"Позиция: {last_record['position']:.2f}")
        print(f"Длина контента: {last_record['content_length']}")

def run_comprehensive_tests():
    """Запуск всех тестов"""
    
    print_section_header("КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ SEO AI MODELS")
    print(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Запуск всех тестов по очереди
        test_text_processor()
        test_content_analyzer()
        test_semantic_analyzer()
        test_eeat_analyzer()
        test_rank_predictor()
        test_suggester()
        test_seo_advisor()
        
        print("\n" + "=" * 80)
        print("ИТОГ ТЕСТИРОВАНИЯ: ВСЕ ТЕСТЫ ВЫПОЛНЕНЫ УСПЕШНО")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"ОШИБКА ПРИ ВЫПОЛНЕНИИ ТЕСТОВ: {str(e)}")
        import traceback
        print(traceback.format_exc())
        print("=" * 80)

if __name__ == "__main__":
    run_comprehensive_tests()
