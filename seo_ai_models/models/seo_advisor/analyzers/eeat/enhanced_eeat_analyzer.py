"""Улучшенный анализатор E-E-A-T с использованием машинного обучения.

Расширенная версия анализатора E-E-A-T, которая использует модель 
машинного обучения для более точной оценки контента и генерации
специализированных отраслевых рекомендаций.
"""

import logging
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

# Относительный импорт базового анализатора
from .eeat_analyzer import EEATAnalyzer

logger = logging.getLogger(__name__)

class EnhancedEEATAnalyzer(EEATAnalyzer):
    """Улучшенный анализатор E-E-A-T с поддержкой моделей машинного обучения.
    
    Расширяет базовый анализатор E-E-A-T возможностью использования
    моделей машинного обучения для оценки контента и предоставления
    более точных, отраслевых рекомендаций.
    
    Attributes:
        model: Загруженная модель машинного обучения или None.
        ml_model_used (bool): Флаг, указывающий на использование модели.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация анализатора
        
        Args:
            model_path: Путь к модели машинного обучения
        """
        super().__init__()
        self.model = None
        self.ml_model_used = False
        
        # Пытаемся загрузить модель, если указан путь
        if model_path:
            try:
                self.model = joblib.load(model_path)
                self.ml_model_used = True
                logger.info(f"Модель E-E-A-T загружена из {model_path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели E-E-A-T: {e}")
                logger.info("Продолжение работы с базовой версией анализатора")
    
    def analyze(self, content: str, industry: str = 'default') -> Dict[str, Union[float, Dict, List]]:
        """
        Анализ содержимого с использованием модели машинного обучения
        
        Args:
            content: Текстовое содержимое
            industry: Отрасль контента
            
        Returns:
            Словарь с оценками и рекомендациями
        """
        # Получаем базовую оценку от родительского класса
        base_results = super().analyze(content, industry)
        
        # Для YMYL отраслей важность E-E-A-T выше
        is_ymyl = industry in ['finance', 'health', 'legal', 'medical']
        
        # Если есть доступ к модели, используем ее для улучшения результатов
        if self.model and self.ml_model_used:
            try:
                # Подготовка данных для модели
                features = np.array([
                    base_results['expertise_score'],
                    base_results['authority_score'],
                    base_results['trust_score'],
                    base_results['structural_score'],
                    0.7,  # semanic_coherence_score (заглушка)
                    0.5,  # citation_score (заглушка)
                    0.6,  # external_links_score (заглушка)
                    1 if is_ymyl else 0
                ]).reshape(1, -1)
                
                # Предсказание с помощью модели
                overall_eeat_score = float(self.model.predict(features)[0])
                base_results['overall_eeat_score'] = overall_eeat_score
                logger.info(f"E-E-A-T оценка с использованием ML: {overall_eeat_score:.4f}")
                
                # Добавляем информацию об использовании модели
                base_results['ml_model_used'] = True
            except Exception as e:
                logger.error(f"Ошибка при использовании модели ML: {e}")
                logger.info("Возврат к базовому расчету оценки")
                base_results['ml_model_used'] = False
        else:
            base_results['ml_model_used'] = False
            
        # Дополняем рекомендации для отрасли
        industry_recommendations = self._get_industry_specific_recommendations(industry)
        if industry_recommendations:
            base_results['recommendations'].extend(industry_recommendations)
            
        return base_results
    
    def _get_industry_specific_recommendations(self, industry: str) -> List[str]:
        """
        Получение рекомендаций, специфичных для отрасли
        
        Args:
            industry: Отрасль контента
            
        Returns:
            Список рекомендаций
        """
        industry_recommendations = {
            'finance': [
                "Включите актуальные финансовые данные с указанием источников",
                "Добавьте дисклеймер о том, что материал не является индивидуальной консультацией",
                "Укажите профессиональную квалификацию автора в финансовой сфере"
            ],
            'health': [
                "Добавьте ссылки на медицинские исследования и рецензируемые источники",
                "Укажите медицинскую квалификацию автора или экспертов",
                "Включите предупреждение о необходимости консультации с врачом"
            ],
            'legal': [
                "Подчеркните, что материал не заменяет консультацию с юристом",
                "Укажите юридическую квалификацию автора",
                "Включите ссылки на законодательство и нормативные акты"
            ],
            'technology': [
                "Добавьте технические спецификации и сравнительные таблицы",
                "Включите результаты тестирования продуктов",
                "Укажите дату анализа технологий для сохранения актуальности"
            ],
            'education': [
                "Укажите педагогический опыт автора",
                "Добавьте ссылки на образовательные исследования",
                "Включите методические рекомендации с опорой на стандарты"
            ]
        }
        
        return industry_recommendations.get(industry, [])
