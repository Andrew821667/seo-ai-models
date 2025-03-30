
import sys
sys.path.append('/content/seo-ai-models')
from models.seo_advisor.advisor import SEOAdvisor
from models.seo_advisor.enhanced_eeat_analyzer import EnhancedEEATAnalyzer

# Обновим конфигурацию SEO Advisor для использования нашей новой модели
class EnhancedSEOAdvisor(SEOAdvisor):
    def __init__(self, industry: str = 'default', use_eeat_model: bool = True):
        super().__init__(industry)
        # Инициализация улучшенного анализатора E-E-A-T
        if use_eeat_model:
            self.eeat_analyzer = EnhancedEEATAnalyzer(
                model_path='/content/seo-ai-models/models/checkpoints/eeat_best_model.joblib'
            )
            print("Инициализирован улучшенный анализатор E-E-A-T с моделью машинного обучения")
    
    def analyze_content(self, content: str, target_keywords: list):
        # Получаем базовый анализ от родительского класса
        report = super().analyze_content(content, target_keywords)
        
        # Добавляем расширенный E-E-A-T анализ
        eeat_analysis = self.eeat_analyzer.analyze(content, industry=self.industry)
        
        # Обновляем метрики в отчете
        report.content_metrics.update({
            "expertise_score": eeat_analysis['expertise_score'],
            "authority_score": eeat_analysis['authority_score'],
            "trust_score": eeat_analysis['trust_score'],
            "enhanced_eeat_score": eeat_analysis['overall_eeat_score']
        })
        
        # Добавляем E-E-A-T рекомендации
        if "eeat_improvement" not in report.recommendations:
            report.recommendations["eeat_improvement"] = []
        
        report.recommendations["eeat_improvement"].extend(
            eeat_analysis.get('recommendations', [])
        )
        
        return report

# Тестовый пример
if __name__ == "__main__":
    test_content = """
    # Как выбрать лучший смартфон в 2025 году
    
    Выбор нового смартфона может быть сложной задачей из-за огромного количества доступных моделей.
    В этой статье мы рассмотрим ключевые факторы, которые нужно учесть при покупке.
    
    ## Ключевые характеристики
    
    1. Производительность процессора
    2. Качество камеры и фото
    3. Время работы батареи
    4. Дисплей и его разрешение
    5. Объем памяти
    
    ## Рейтинг лучших моделей
    
    По данным независимых тестов, проведенных нашей лабораторией, лидерами стали:
    
    * Samsung Galaxy S25 Ultra
    * iPhone 16 Pro
    * Google Pixel 9
    
    ## Экспертное мнение
    
    Джон Смит, технический обозреватель с 15-летним опытом тестирования смартфонов, отмечает:
    "В 2025 году искусственный интеллект стал ключевым фактором при выборе смартфона".
    
    ## Источники
    
    - Consumer Reports, март 2025
    - Mobile Tech Review
    - Данные тестирования TechLab
    """
    
    # Целевые ключевые слова
    keywords = ["смартфон", "выбор смартфона", "лучший смартфон 2025"]
    
    # Инициализация улучшенного SEO Advisor
    advisor = EnhancedSEOAdvisor(industry='tech')
    
    # Анализ контента
    report = advisor.analyze_content(test_content, keywords)
    
    # Вывод основных метрик
    print("\nОсновные метрики контента:")
    print(f"Общий E-E-A-T скор: {report.content_metrics.get('enhanced_eeat_score', 'нет данных'):.4f}")
    print(f"Экспертность: {report.content_metrics.get('expertise_score', 'нет данных'):.4f}")
    print(f"Авторитетность: {report.content_metrics.get('authority_score', 'нет данных'):.4f}")
    print(f"Доверие: {report.content_metrics.get('trust_score', 'нет данных'):.4f}")
    
    # Вывод E-E-A-T рекомендаций
    print("\nE-E-A-T рекомендации:")
    for i, rec in enumerate(report.recommendations.get("eeat_improvement", [])[:5], 1):
        print(f"{i}. {rec}")
    
    # Вывод предсказанного рейтинга
    print(f"\nПрогнозируемая позиция: {report.predicted_position:.2f}")
