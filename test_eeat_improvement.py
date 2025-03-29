
import sys
sys.path.append('/content/seo-ai-models')

from models.seo_advisor.eeat_analyzer import EEATAnalyzer
from models.seo_advisor.calibrated_rank_predictor import CalibratedRankPredictor

def test_eeat_improvement():
    # Инициализация компонентов
    eeat_analyzer = EEATAnalyzer()
    
    # Исходный текст среднего качества
    original_text = '''
    Методы снижения веса без строгих диет
    
    Исследования показывают, что плавный подход к снижению веса дает более 
    устойчивые результаты, чем строгие диеты.
    
    ## Эффективные стратегии
    
    * Интервальное питание (16/8) помогает снизить общее потребление калорий
    * Умеренная физическая активность 3-4 раза в неделю ускоряет метаболизм
    * Замена простых углеводов сложными повышает чувство насыщения
    
    Последние данные свидетельствуют, что люди, постепенно меняющие пищевые привычки,
    теряют в среднем 5-10% веса за 6 месяцев.
    
    Источник: Журнал "Питание и Метаболизм", 2023.
    '''
    
    # Улучшенный текст на основе рекомендаций E-E-A-T
    improved_text = '''
    # Научно обоснованные методы снижения веса без строгих диет
    
    *Обновлено: 20 марта 2024 года*
    
    По данным исследования, опубликованного в журнале "Питание и Метаболизм" в декабре 2023 года, 
    постепенный подход к снижению веса дает более устойчивые и долгосрочные результаты, 
    чем краткосрочные строгие диеты. Метаанализ 15 клинических исследований показал, 
    что 70% людей, использующих строгие диеты, возвращаются к исходному весу в течение года.
    
    ## Эффективные стратегии с доказанной эффективностью
    
    1. **Интервальное питание (16/8)**: 
       * Клиническое исследование Университета Чикаго (2023) подтвердило, что этот режим помогает 
         снизить общее потребление калорий на 20-25% без чувства голода
       * Уровень инсулина снижается на 30-35%, что способствует использованию жировых запасов
    
    2. **Умеренная физическая активность**:
       * Исследования показывают, что 30-40 минут активности 3-4 раза в неделю ускоряют 
         метаболизм на 10-15%
       * Оптимальное сочетание: 2 дня силовых тренировок и 2 дня кардио нагрузки
    
    3. **Качественные изменения в питании**:
       * Замена простых углеводов (белый хлеб, сахар) сложными (овощи, цельные злаки) 
         увеличивает чувство насыщения на 40%
       * Увеличение потребления белка до 25-30% от общего калоража стабилизирует уровень 
         сахара в крови и снижает тягу к перекусам
    
    ### Результаты и прогнозы
    
    Согласно данным Американской ассоциации диетологов, люди, постепенно меняющие пищевые привычки 
    без строгих ограничений, теряют в среднем 5-10% веса за 6 месяцев и, что более важно, 
    удерживают достигнутый результат в 65% случаев в течение 5 лет.
    
    > "Устойчивые изменения питания и образа жизни - это ключ к долгосрочному контролю веса, 
    > а не краткосрочные резкие ограничения" - доктор Элизабет Кац, доктор медицинских наук, 
    > профессор нутрициологии Стэнфордского университета.
    
    ### Предупреждение
    
    Перед началом любой программы снижения веса проконсультируйтесь с врачом, особенно если 
    у вас есть хронические заболевания, такие как диабет, гипертония или нарушения 
    работы щитовидной железы.
    
    **Методология**: Данная статья основана на анализе рецензируемых научных публикаций 
    из журналов с импакт-фактором >2.0 за период 2020-2024 гг. и рекомендациях 
    ведущих медицинских организаций.
    
    **Об авторе**: Статья подготовлена Анной Петровой, сертифицированным диетологом-нутрициологом 
    с 8-летним опытом клинической практики, членом Национальной ассоциации диетологов, 
    выпускницей медицинского факультета МГУ.
    
    **Источники**:
    - Journal of Nutrition and Metabolism, Dec 2023: "Long-term weight management strategies"
    - American Journal of Clinical Nutrition, Nov 2023: "Intermittent fasting outcomes"
    - Harvard Medical School, Health Publications, Jan 2024: "Sustainable weight loss approaches"
    - Stanford University, Nutrition Science Department, Feb 2024: "Metabolic adaptations to dietary changes"
    
    *Дата публикации: 15 марта 2024 | Дата обновления: 20 марта 2024*
    '''
    
    # Проверяем исходный и улучшенный текст
    print("====== СРАВНЕНИЕ ИСХОДНОГО И УЛУЧШЕННОГО ТЕКСТА ======")
    
    # Отрасль для тестирования
    industry = 'health'  # YMYL тематика
    
    # Анализ E-E-A-T для исходного текста
    original_eeat = eeat_analyzer.analyze(original_text, industry=industry)
    
    print("\n=== ИСХОДНЫЙ ТЕКСТ ===")
    print(f"Общая оценка E-E-A-T: {original_eeat['overall_eeat_score']:.2f}")
    print("Компоненты:")
    print(f"- Экспертиза: {original_eeat['expertise_score']:.2f}")
    print(f"- Авторитетность: {original_eeat['authority_score']:.2f}")
    print(f"- Доверие: {original_eeat['trust_score']:.2f}")
    print(f"- Структура: {original_eeat['structural_score']:.2f}")
    
    print("\nРекомендации от E-E-A-T анализатора:")
    for i, rec in enumerate(original_eeat['recommendations'][:5], 1):
        print(f"{i}. {rec}")
    
    # Анализ E-E-A-T для улучшенного текста
    improved_eeat = eeat_analyzer.analyze(improved_text, industry=industry)
    
    print("\n=== УЛУЧШЕННЫЙ ТЕКСТ ===")
    print(f"Общая оценка E-E-A-T: {improved_eeat['overall_eeat_score']:.2f}")
    print("Компоненты:")
    print(f"- Экспертиза: {improved_eeat['expertise_score']:.2f}")
    print(f"- Авторитетность: {improved_eeat['authority_score']:.2f}")
    print(f"- Доверие: {improved_eeat['trust_score']:.2f}")
    print(f"- Структура: {improved_eeat['structural_score']:.2f}")
    
    print("\nОставшиеся рекомендации:")
    for i, rec in enumerate(improved_eeat['recommendations'][:3], 1):
        print(f"{i}. {rec}")
    
    # Проверка влияния на ранжирование
    print("\n====== ВЛИЯНИЕ НА РАНЖИРОВАНИЕ ======")
    
    # Базовые метрики для обоих текстов (одинаковые для изоляции влияния E-E-A-T)
    base_features = {
        'keyword_density': 0.02,
        'content_length': 500,
        'readability_score': 70,
        'meta_tags_score': 0.6,
        'header_structure_score': 0.6,
        'multimedia_score': 0.4,
        'internal_linking_score': 0.4,
        'topic_relevance': 0.7,
        'semantic_depth': 0.6,
        'engagement_potential': 0.6
    }
    
    # Добавляем E-E-A-T метрики для исходного текста
    original_features = base_features.copy()
    original_features.update({
        'expertise_score': original_eeat['expertise_score'],
        'authority_score': original_eeat['authority_score'],
        'trust_score': original_eeat['trust_score'],
        'overall_eeat_score': original_eeat['overall_eeat_score']
    })
    
    # Добавляем E-E-A-T метрики для улучшенного текста
    improved_features = base_features.copy()
    improved_features.update({
        'expertise_score': improved_eeat['expertise_score'],
        'authority_score': improved_eeat['authority_score'],
        'trust_score': improved_eeat['trust_score'],
        'overall_eeat_score': improved_eeat['overall_eeat_score']
    })
    
    # Инициализация предиктора
    predictor = CalibratedRankPredictor(industry=industry)
    
    # Прогноз для исходного текста
    original_prediction = predictor.predict_position(original_features)
    
    # Прогноз для улучшенного текста
    improved_prediction = predictor.predict_position(improved_features)
    
    print(f"Исходная позиция: {original_prediction['position']:.1f}")
    print(f"Улучшенная позиция: {improved_prediction['position']:.1f}")
    print(f"Абсолютное улучшение: {original_prediction['position'] - improved_prediction['position']:.1f} позиций")
    print(f"Процентное улучшение: {((original_prediction['position'] - improved_prediction['position']) / original_prediction['position'] * 100):.1f}%")
    
    # Анализ изменений E-E-A-T
    print("\n====== ДЕТАЛЬНЫЙ АНАЛИЗ УЛУЧШЕНИЙ E-E-A-T ======")
    
    print("\nКомпоненты E-E-A-T (до → после):")
    
    # Рассчитываем улучшение по каждому компоненту
    expertise_improvement = improved_eeat['expertise_score'] - original_eeat['expertise_score']
    authority_improvement = improved_eeat['authority_score'] - original_eeat['authority_score']
    trust_improvement = improved_eeat['trust_score'] - original_eeat['trust_score']
    structure_improvement = improved_eeat['structural_score'] - original_eeat['structural_score']
    overall_improvement = improved_eeat['overall_eeat_score'] - original_eeat['overall_eeat_score']
    
    print(f"- Экспертиза: {original_eeat['expertise_score']:.2f} → {improved_eeat['expertise_score']:.2f} ({expertise_improvement:.2f}, {expertise_improvement/max(0.01, original_eeat['expertise_score'])*100:.0f}%)")
    print(f"- Авторитетность: {original_eeat['authority_score']:.2f} → {improved_eeat['authority_score']:.2f} ({authority_improvement:.2f}, {authority_improvement/max(0.01, original_eeat['authority_score'])*100:.0f}%)")
    print(f"- Доверие: {original_eeat['trust_score']:.2f} → {improved_eeat['trust_score']:.2f} ({trust_improvement:.2f}, {trust_improvement/max(0.01, original_eeat['trust_score'])*100:.0f}%)")
    print(f"- Структура: {original_eeat['structural_score']:.2f} → {improved_eeat['structural_score']:.2f} ({structure_improvement:.2f}, {structure_improvement/max(0.01, original_eeat['structural_score'])*100:.0f}%)")
    print(f"- Общий E-E-A-T: {original_eeat['overall_eeat_score']:.2f} → {improved_eeat['overall_eeat_score']:.2f} ({overall_improvement:.2f}, {overall_improvement/max(0.01, original_eeat['overall_eeat_score'])*100:.0f}%)")
    
    # Найдем маркеры, которые были добавлены в улучшенный текст
    print("\nДобавленные маркеры E-E-A-T:")
    
    # Для экспертизы
    original_exp_markers = set(original_eeat['component_details']['expertise']['found_markers'])
    improved_exp_markers = set(improved_eeat['component_details']['expertise']['found_markers'])
    new_exp_markers = improved_exp_markers - original_exp_markers
    
    print("Маркеры экспертизы:")
    for marker in new_exp_markers:
        print(f"- {marker}")
    
    # Для авторитетности
    original_auth_markers = set(original_eeat['component_details']['authority']['found_markers'])
    improved_auth_markers = set(improved_eeat['component_details']['authority']['found_markers'])
    new_auth_markers = improved_auth_markers - original_auth_markers
    
    print("\nМаркеры авторитетности:")
    for marker in new_auth_markers:
        print(f"- {marker}")
    
    # Для доверия
    print("\nУлучшения структуры:")
    orig_structure = original_eeat['component_details']['structure']['elements']
    improved_structure = improved_eeat['component_details']['structure']['elements']
    
    print(f"- Заголовки: {orig_structure.get('headers', 0)} → {improved_structure.get('headers', 0)}")
    print(f"- Списки: {orig_structure.get('lists', 0)} → {improved_structure.get('lists', 0)}")
    print(f"- Цитаты: {orig_structure.get('quotes', 0)} → {improved_structure.get('quotes', 0)}")
    print(f"- Параграфы: {orig_structure.get('paragraphs', 0)} → {improved_structure.get('paragraphs', 0)}")
    
    # Итоговые выводы
    print("\n====== ВЫВОДЫ ======")
    print(f"1. Улучшение E-E-A-T привело к повышению позиции на {original_prediction['position'] - improved_prediction['position']:.1f} пунктов ({((original_prediction['position'] - improved_prediction['position']) / original_prediction['position'] * 100):.1f}%)")
    print(f"2. Наибольшее улучшение произошло в компоненте {'экспертизы' if expertise_improvement >= max(authority_improvement, trust_improvement, structure_improvement) else 'авторитетности' if authority_improvement >= max(expertise_improvement, trust_improvement, structure_improvement) else 'доверия' if trust_improvement >= max(expertise_improvement, authority_improvement, structure_improvement) else 'структуры'}")
    
    # Определение наиболее эффективных изменений
    print("3. Наиболее эффективные улучшения:")
    
    effective_changes = []
    
    if len(new_exp_markers) > 0:
        effective_changes.append("- Добавление информации об авторе и его квалификации")
    
    if len(new_auth_markers) > 0:
        effective_changes.append("- Цитирование авторитетных источников и исследований")
    
    if improved_structure.get('quotes', 0) > orig_structure.get('quotes', 0):
        effective_changes.append("- Включение экспертных цитат")
    
    if improved_structure.get('lists', 0) > orig_structure.get('lists', 0):
        effective_changes.append("- Улучшение структуры с помощью нумерованных списков")
    
    if improved_structure.get('headers', 0) > orig_structure.get('headers', 0):
        effective_changes.append("- Добавление подзаголовков для лучшей структуры")
    
    if "дата" in improved_text.lower() and "дата" not in original_text.lower():
        effective_changes.append("- Указание дат публикации и обновления")
    
    if "методология" in improved_text.lower() and "методология" not in original_text.lower():
        effective_changes.append("- Описание методологии")
        
    for change in effective_changes:
        print(change)
    
    print("\n4. Рентабельность оптимизации E-E-A-T: высокая для YMYL контента среднего качества")

if __name__ == "__main__":
    test_eeat_improvement()
