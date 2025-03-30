
import sys
sys.path.append('/content/seo-ai-models')

# Импортируем интегратор E-E-A-T
from models.seo_advisor.eeat_integrator import EEATIntegrator

# Пример класса SEOAdvisor с интегрированным E-E-A-T анализом
class EnhancedSEOAdvisor:
    """Пример интеграции E-E-A-T анализа в SEO Advisor"""
    
    def __init__(self, industry: str = 'default', use_eeat_model: bool = True):
        """
        Инициализация
        
        Args:
            industry: Отрасль для анализа
            use_eeat_model: Использовать ли модель E-E-A-T
        """
        self.industry = industry
        
        # Инициализация E-E-A-T интегратора
        if use_eeat_model:
            self.eeat_integrator = EEATIntegrator()
            print(f"Инициализирован E-E-A-T интегратор для отрасли: {industry}")
    
    def analyze_content(self, content: str, keywords: list) -> dict:
        """
        Анализ контента с интегрированным E-E-A-T
        
        Args:
            content: Текстовый контент
            keywords: Ключевые слова
            
        Returns:
            Результаты анализа
        """
        # Базовый анализ (здесь могла бы быть логика существующего SEOAdvisor)
        metrics = {
            'word_count': len(content.split()),
            'keyword_count': sum(content.lower().count(kw.lower()) for kw in keywords),
            'keyword_density': sum(content.lower().count(kw.lower()) for kw in keywords) / max(len(content.split()), 1)
        }
        
        # Расширяем метрики результатами E-E-A-T анализа
        enhanced_metrics = self.eeat_integrator.enhance_metrics(metrics, content, self.industry)
        
        # Получаем рекомендации по E-E-A-T
        eeat_recommendations = self.eeat_integrator.get_recommendations(content, self.industry)
        
        # Формируем финальный отчет
        report = {
            'metrics': enhanced_metrics,
            'recommendations': {
                'general': ["Улучшите плотность ключевых слов", "Расширьте контент"],
                'eeat': eeat_recommendations
            },
            'predicted_position': self._calculate_position(enhanced_metrics)
        }
        
        return report
    
    def _calculate_position(self, metrics: dict) -> float:
        """
        Расчет предсказанного положения с учетом E-E-A-T
        
        Args:
            metrics: Метрики контента, включая E-E-A-T
            
        Returns:
            Предсказанная позиция
        """
        # Реализация могла бы использовать CalibratedRankPredictor
        # Здесь простая иллюстративная формула
        
        # Базовый расчет
        base_score = 0.5  # Заглушка
        
        # Если E-E-A-T метрики доступны, используем их
        eeat_modifier = 1.0
        if 'overall_eeat_score' in metrics:
            eeat_score = metrics['overall_eeat_score']
            eeat_modifier = 1.5 - (eeat_score * 0.8)  # Чем выше E-E-A-T, тем лучше позиция
            
        # YMYL модификатор (более строгие требования для YMYL-контента)
        ymyl_modifier = 1.0
        if metrics.get('ymyl_status', 0) == 1:
            ymyl_modifier = 1.3
            
        # Расчет позиции (от 1 до 100)
        position = min(100, max(1, 50 * eeat_modifier * ymyl_modifier * (1 - base_score)))
        
        return position

# Тест интеграции с примерами контента
def test_integration():
    # Пример контента для анализа
    test_content = """
    # Инвестиционная стратегия на 2025 год
    
    В этой статье рассмотрим перспективные инвестиционные возможности на 2025 год.
    
    ## Ключевые тренды
    
    1. Зеленая энергетика
    2. Искусственный интеллект
    3. Биотехнологии
    
    ## Рекомендации
    
    Рекомендуется диверсифицировать портфель и рассмотреть ETF в сфере инноваций.
    
    Источник: финансовые отчеты компаний за 2024 год.
    """
    
    # Ключевые слова
    keywords = ["инвестиции", "стратегия", "2025", "тренды"]
    
    # Инициализация расширенного SEO Advisor
    advisor = EnhancedSEOAdvisor(industry='finance')
    
    # Анализ контента
    results = advisor.analyze_content(test_content, keywords)
    
    # Вывод результатов
    print("\n=== Результаты анализа контента ===")
    print(f"Слов: {results['metrics']['word_count']}")
    print(f"Ключевых слов: {results['metrics']['keyword_count']}")
    
    # E-E-A-T метрики
    print("\n=== E-E-A-T метрики ===")
    print(f"Expertise Score: {results['metrics']['expertise_score']:.4f}")
    print(f"Authority Score: {results['metrics']['authority_score']:.4f}")
    print(f"Trust Score: {results['metrics']['trust_score']:.4f}")
    print(f"Overall E-E-A-T Score: {results['metrics']['overall_eeat_score']:.4f}")
    
    # E-E-A-T рекомендации
    print("\n=== E-E-A-T рекомендации ===")
    for i, rec in enumerate(results['recommendations']['eeat'], 1):
        print(f"{i}. {rec}")
    
    # Предсказание позиции
    print(f"\nПрогнозируемая позиция: {results['predicted_position']:.2f}")

if __name__ == "__main__":
    test_integration()
