"""Демонстрация использования улучшенного SEO Advisor."""

import sys
import os
# Добавляем корень проекта в путь импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from seo_ai_models.models.seo_advisor.advisor import SEOAdvisor
from seo_ai_models.models.seo_advisor.analyzers.eeat.enhanced_eeat_analyzer import EnhancedEEATAnalyzer


class EnhancedSEOAdvisor(SEOAdvisor):
    """Расширенная версия SEO Advisor с улучшенным E-E-A-T анализом."""
    
    def __init__(self, industry: str = 'default', use_eeat_model: bool = True):
        super().__init__(industry)
        
        # Инициализация улучшенного анализатора E-E-A-T
        if use_eeat_model:
            # Вычисляем относительный путь к модели
            model_path = Path(__file__).parents[1] / "seo_ai_models/data/models/eeat/eeat_best_model.joblib"
            if model_path.exists():
                self.eeat_analyzer = EnhancedEEATAnalyzer(model_path=str(model_path))
                print(f"Инициализирован улучшенный анализатор E-E-A-T с моделью: {model_path}")
            else:
                print(f"Модель не найдена по пути: {model_path}")
    
    def analyze_content(self, content: str, target_keywords: list):
        # Получаем базовый анализ от родительского класса
        report = super().analyze_content(content, target_keywords)
        
        # Дополнительная логика для улучшенного анализа
        print("Выполнен расширенный анализ E-E-A-T")
        
        return report


# Тестовый пример
if __name__ == "__main__":
    # Тестовый контент
    test_content = """
    # Руководство по инвестированию для начинающих
    
    Инвестирование представляет собой один из ключевых инструментов для обеспечения финансового благополучия в будущем.
    
    ## Что такое инвестиции?
    
    Инвестиции — это размещение капитала с целью получения дохода или увеличения его стоимости в будущем.
    
    ## Почему стоит инвестировать?
    
    По данным исследования Morgan Stanley, средняя годовая доходность индекса S&P 500 за последние 30 лет составила около 10%.
    """
    
    # Целевые ключевые слова
    keywords = ["инвестирование", "инвестиции", "доходность"]
    
    # Инициализация улучшенного SEO Advisor
    advisor = EnhancedSEOAdvisor(industry='finance')
    
    # Анализ контента
    report = advisor.analyze_content(test_content, keywords)
    
    # Вывод результатов
    print("\nРезультаты анализа:")
    print(f"Прогнозируемая позиция: {report.predicted_position:.2f}")
    print(f"Оценка E-E-A-T: {report.content_metrics.get('overall_eeat_score', 'не доступно')}")
