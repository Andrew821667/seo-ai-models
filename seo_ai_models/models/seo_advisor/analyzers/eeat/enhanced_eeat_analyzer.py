"""Улучшенный анализатор E-E-A-T с использованием машинного обучения."""

import logging
import joblib
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Импорт базового анализатора
from seo_ai_models.models.seo_advisor.analyzers.eeat.eeat_analyzer import EEATAnalyzer

class EnhancedEEATAnalyzer(EEATAnalyzer):
    """Улучшенный анализатор E-E-A-T с использованием машинного обучения."""
    
    def __init__(self, model_path: Optional[str] = None, language: str = 'ru'):
        """Инициализация анализатора с моделью машинного обучения.
        
        Args:
            model_path (str, optional): Путь к модели ML. Если None, будет использован
                стандартный путь к модели в проекте.
            language (str): Язык для анализа контента.
        """
        super().__init__(language=language)
        
        self.model = None
        self.ml_model_used = False
        
        # Если путь не указан, используем стандартный путь в проекте
        if model_path is None:
            # Попробуем найти модель в стандартном расположении
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
                'data', 'models', 'eeat', 'eeat_best_model.joblib'
            )
            if os.path.exists(default_path):
                model_path = default_path
        
        # Пытаемся загрузить модель
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.ml_model_used = True
                logging.info(f"EEAT ML модель успешно загружена: {type(self.model)}")
            except Exception as e:
                logging.error(f"Ошибка загрузки модели E-E-A-T: {e}")
                self.ml_model_used = False
        else:
            if model_path:
                logging.warning(f"Файл модели не найден: {model_path}")
            else:
                logging.info("Путь к модели не указан, будет использован базовый анализатор")
    
    def _prepare_features_for_model(self, text_lower: str, base_metrics: Dict[str, float]) -> np.ndarray:
        """Подготовка признаков для модели машинного обучения.
        
        Args:
            text_lower: Нормализованный текст (в нижнем регистре)
            base_metrics: Базовые метрики E-E-A-T анализа
            
        Returns:
            np.ndarray: Вектор признаков для модели
        """
        # Подготовка векторов признаков - точно 8 признаков
        features = []
        
        # 1. Основные EEAT метрики (4 признака)
        features.extend([
            base_metrics['experience_score'],
            base_metrics['expertise_score'],
            base_metrics['authority_score'],
            base_metrics['trust_score']
        ])
        
        # 2. Структурная оценка (1 признак)
        features.append(base_metrics['structural_score'])
        
        # 3. Дополнительные характеристики (3 признака)
        # Количество слов (нормализованное)
        word_count = len(text_lower.split())
        features.append(min(word_count / 2000, 1.0))  # Нормализация до 2000 слов
        
        # Наличие ссылок на источники
        has_sources = ('источник' in text_lower) or ('reference' in text_lower) or ('[' in text_lower and ']' in text_lower)
        features.append(float(has_sources))
        
        # Наличие авторства
        has_author = ('автор' in text_lower) or ('профессор' in text_lower) or ('доктор' in text_lower) or ('эксперт' in text_lower)
        features.append(float(has_author))
        
        # Проверка размерности вектора признаков
        assert len(features) == 8, f"Ошибка: вектор признаков должен содержать 8 элементов, сейчас: {len(features)}"
        
        # Возвращаем numpy массив
        return np.array([features])
    
    def analyze(self, content: str, industry: str = 'default', language: str = None,
               html_content: Optional[str] = None) -> Dict[str, Union[float, Dict, List]]:
        """Расширенный анализ контента с использованием ML-модели.
        
        Args:
            content: Текстовое содержимое
            industry: Отрасль контента
            language: Язык контента (если None, будет определен автоматически)
            html_content: HTML-версия контента (если доступна)
            
        Returns:
            Словарь с оценками и рекомендациями
        """
        # Получаем базовые результаты от родительского класса
        base_results = super().analyze(content, industry, language, html_content)
        
        # Если модель недоступна, возвращаем базовые результаты
        if not self.model or not self.ml_model_used:
            base_results['ml_model_used'] = False
            return base_results
        
        try:
            # Подготовка данных для модели
            text_lower = content.lower()
            features = self._prepare_features_for_model(text_lower, base_results)
            
            # Проверяем размерность
            logging.info(f"Размерность вектора признаков: {features.shape}")
            
            # Используем модель для предсказания улучшенной оценки E-E-A-T
            enhanced_score = float(self.model.predict(features)[0])
            
            # Ограничиваем значением от 0 до 1
            enhanced_score = max(min(enhanced_score, 1.0), 0.0)
            
            # Определение разницы между базовой и улучшенной оценкой
            original_score = base_results['overall_eeat_score']
            score_difference = enhanced_score - original_score
            
            # Добавляем дополнительные рекомендации на основе анализа ML
            model_recommendations = []
            if score_difference < -0.1:
                # Модель оценила контент хуже, чем базовый анализатор
                model_recommendations.append(
                    "ML-анализ показал, что контент имеет скрытые проблемы с E-E-A-T."
                )
                if enhanced_score < 0.4:
                    model_recommendations.append(
                        "Рекомендуется значительно улучшить авторитетность и экспертность контента."
                    )
            elif score_difference > 0.1:
                # Модель оценила контент лучше, чем базовый анализатор
                model_recommendations.append(
                    "ML-анализ показал дополнительные сильные стороны контента с точки зрения E-E-A-T."
                )
            
            # Специфические рекомендации на основе значения enhanced_score
            if enhanced_score < 0.3:
                model_recommendations.append(
                    "Критически низкая оценка E-E-A-T: добавьте цитаты экспертов и ссылки на авторитетные источники."
                )
            elif enhanced_score < 0.6:
                model_recommendations.append(
                    "Средняя оценка E-E-A-T: усильте акцент на квалификации автора и обоснованности утверждений."
                )
                
            # Обновляем результаты
            base_results['original_overall_eeat_score'] = original_score
            base_results['overall_eeat_score'] = enhanced_score
            base_results['ml_model_used'] = True
            base_results['ml_score_difference'] = score_difference
            
            # Добавляем дополнительные рекомендации от модели ML
            base_results['recommendations'].extend(model_recommendations)
            
        except Exception as e:
            logging.error(f"Ошибка при использовании модели ML: {e}")
            base_results['ml_model_used'] = False
        
        return base_results
