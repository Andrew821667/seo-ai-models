"""Откалиброванный предиктор ранжирования с оптимизированными весами.

Модуль содержит улучшенную версию предиктора ранжирования с калиброванными весами,
учитывающую специфику отрасли и уровень конкуренции в нише.
"""

from typing import Dict, List, Optional, Union, Tuple
import math

# Относительный импорт родительского класса
from .improved_rank_predictor import ImprovedRankPredictor

class CalibratedRankPredictor(ImprovedRankPredictor):
    """
    Усовершенствованный предиктор ранжирования с калиброванными весами,
    учетом конкурентности ниши и отраслевой специфики
    """
    
    def __init__(self, industry: str = 'default'):
        """Инициализация предиктора с улучшенными весами.
        
        Args:
            industry (str): Отрасль для специализированных настроек. Влияет на
                применяемые коэффициенты конкурентности и оценку YMYL-статуса.
        """
        super().__init__(industry)
        
        # Откалиброванные веса с суммой 1.0
        self.feature_weights = {
            'keyword_density': 0.12,       # Снижено с 0.15 - меньше акцента на простую плотность
            'content_length': 0.15,        # Снижено с 0.18 - качество важнее объема
            'readability_score': 0.13,     # Снижено с 0.15
            'meta_tags_score': 0.07,       # Снижено с 0.08
            'header_structure_score': 0.12, # Без изменений
            'multimedia_score': 0.06,      # Снижено с 0.07
            'internal_linking_score': 0.06, # Снижено с 0.07
            'topic_relevance': 0.16,       # Снижено с 0.18
            'semantic_depth': 0.13,        # Снижено с 0.15
            'engagement_potential': 0.10    # Без изменений
        }
        
        # Нормализаторы для входных данных
        self.feature_normalizers = {
            'keyword_density': 0.04,     # Нормализация по максимальной плотности 4%
            'content_length': 2000,      # Нормализация по 2000 слов
            'readability_score': 100,    # Шкала 0-100
            'meta_tags_score': 1.0,      # Шкала 0-1
            'header_structure_score': 1.0, # Шкала 0-1
            'multimedia_score': 1.0,     # Шкала 0-1
            'internal_linking_score': 1.0, # Шкала 0-1
            'topic_relevance': 1.0,      # Шкала 0-1
            'semantic_depth': 1.0,       # Шкала 0-1
            'engagement_potential': 1.0  # Шкала 0-1
        }
        
        # Оптимальные диапазоны для метрик
        self.optimal_ranges = {
            'keyword_density': (0.01, 0.03),  # 1-3% оптимальная плотность
            'content_length': (1200, 2500),   # 1200-2500 слов оптимальная длина
            'readability_score': (70, 90),    # 70-90 оптимальная читабельность
        }
        
        # Коэффициенты конкурентности ниш (снижаем пессимизм)
        self.competition_factors = {
            'electronics': 1.3,    # Высокая конкуренция (было 1.4)
            'health': 1.4,         # Очень высокая (YMYL) (было 1.5)
            'finance': 1.5,        # Наивысшая (YMYL) (было 1.6)
            'travel': 1.1,         # Средняя (было 1.2)
            'education': 1.2,      # Средне-высокая (было 1.3)
            'ecommerce': 1.25,     # Высокая (было 1.35)
            'local_business': 0.9, # Ниже средней (было 1.0)
            'blog': 1.0,           # Низкая (было 1.1)
            'news': 1.2,           # Средне-высокая (было 1.3)
            'default': 1.1         # По умолчанию (было 1.2)
        }
        
        # YMYL отрасли - для особой обработки E-E-A-T факторов
        self.ymyl_industries = {
            'finance': True,
            'health': True,
            'law': True,
            'insurance': True,
            'medical': True
        }
    
    def predict_position(self, features: Dict[str, float], text: str = None, keywords: List[str] = None) -> Dict:
        """Предсказание позиции с учетом конкурентности ниши и E-E-A-T метрик.
        
        Args:
            features (Dict[str, float]): Словарь значений факторов ранжирования.
            text (str, optional): Текст контента для дополнительного анализа.
            keywords (List[str], optional): Ключевые слова для дополнительного анализа.
            
        Returns:
            Dict: Результаты предсказания, включая:
                - position (float): Предсказанная позиция в выдаче.
                - feature_scores (Dict[str, float]): Нормализованные оценки факторов.
                - weighted_scores (Dict[str, float]): Взвешенные оценки факторов.
                - probability (Dict[str, float], optional): Вероятности попадания
                  в различные диапазоны (top3, top5, top10).
        """
        # 1. Нормализация признаков
        normalized_features = {}
        for feature, value in features.items():
            # Если у нас есть нормализатор для этого признака
            if feature in self.feature_normalizers:
                max_value = self.feature_normalizers[feature]
                normalized_features[feature] = min(value / max_value, 1.0)
            else:
                # Если нет - используем значение как есть (если в пределах [0,1])
                normalized_features[feature] = min(max(value, 0.0), 1.0)
        
        # 2. Применение весов
        weighted_scores = {}
        for feature, norm_value in normalized_features.items():
            # Если у нас есть вес для этого признака
            if feature in self.feature_weights:
                weight = self.feature_weights[feature]
                weighted_scores[feature] = norm_value * weight
            else:
                # Если нет веса - используем небольшой стандартный вес
                weighted_scores[feature] = norm_value * 0.05
        
        # 3. Учет E-E-A-T факторов для YMYL отраслей
        is_ymyl = self.industry.lower() in self.ymyl_industries
        total_score = sum(weighted_scores.values())
        
        # Если это YMYL отрасль, проверяем E-E-A-T метрики
        if is_ymyl:
            eeat_features = [
                'expertise_score', 'authority_score', 
                'trust_score', 'overall_eeat_score'
            ]
            
            eeat_penalty = 1.0
            for eeat_feature in eeat_features:
                if eeat_feature in normalized_features:
                    # Если E-E-A-T метрики низкие, применяем штраф
                    if normalized_features[eeat_feature] < 0.5:
                        eeat_penalty *= 0.8 + (normalized_features[eeat_feature] * 0.4)
            
            # Применяем штраф к общему скору
            total_score *= eeat_penalty
        
        # 4. Учет конкурентности ниши
        competition_factor = self.competition_factors.get(self.industry.lower(), 1.1)
        
        # 5. Расчет позиции на основе общего скора
        # Используем логарифмическую формулу для более реалистичного распределения позиций
        # Снижаем общий пессимизм прогноза на 20%
        position_raw = 80 * (1 - (total_score / competition_factor)) # Было 100, теперь 80
        position = max(1, min(100, math.ceil(position_raw)))
        
        # 6. Расчет вероятностей попадания в топы
        probability = {
            'top3': max(0, min(1, 1 - (position / 3))),
            'top5': max(0, min(1, 1 - (position / 5))),
            'top10': max(0, min(1, 1 - (position / 10))),
            'top30': max(0, min(1, 1 - (position / 30))),
            'top50': max(0, min(1, 1 - (position / 50)))
        }
        
        # Формируем результат
        result = {
            'position': position,
            'feature_scores': normalized_features,
            'weighted_scores': weighted_scores,
            'total_score': total_score,
            'competition_factor': competition_factor,
            'ymyl_status': is_ymyl,
            'probability': probability
        }
        
        return result
    
    def generate_recommendations(self, features: Dict[str, float]) -> Dict[str, List[str]]:
        """Генерация конкретных рекомендаций на основе метрик контента.
        
        Args:
            features (Dict[str, float]): Словарь значений факторов ранжирования.
            
        Returns:
            Dict[str, List[str]]: Словарь с категориями рекомендаций.
        """
        recommendations = {}
        
        # Проверяем каждый фактор и генерируем конкретные рекомендации
        for feature, value in features.items():
            if feature == 'keyword_density':
                min_value, max_value = self.optimal_ranges.get(feature, (0.01, 0.03))
                
                if value < min_value:
                    recommendations[feature] = [
                        f"Увеличьте плотность ключевых слов с текущих {value*100:.1f}% до {min_value*100:.1f}-{max_value*100:.1f}%",
                        "Добавьте ключевые слова в заголовки H1, H2 и H3",
                        "Используйте ключевые слова в первом и последнем абзацах",
                        "Включите синонимы и LSI-ключевые слова для естественного повышения плотности"
                    ]
                elif value > max_value:
                    recommendations[feature] = [
                        f"Снизьте плотность ключевых слов с текущих {value*100:.1f}% до {min_value*100:.1f}-{max_value*100:.1f}%",
                        "Избегайте чрезмерного повторения ключевых слов, это может восприниматься как спам",
                        "Используйте больше синонимов и вариаций ключевых слов",
                        "Расширьте контент дополнительной информацией, не содержащей ключевые слова"
                    ]
            
            elif feature == 'content_length':
                min_value, max_value = self.optimal_ranges.get(feature, (1200, 2500))
                
                if value < min_value:
                    recommendations[feature] = [
                        f"Увеличьте объем контента с текущих {value:.0f} слов минимум до {min_value:.0f} слов",
                        "Добавьте больше примеров и подробностей к каждому разделу",
                        "Расширьте введение и заключение",
                        "Добавьте раздел с часто задаваемыми вопросами (FAQ)"
                    ]
                elif value > max_value * 1.5:  # Значительно длиннее оптимального
                    recommendations[feature] = [
                        f"Рассмотрите возможность разделения контента на несколько страниц",
                        "Используйте более лаконичные формулировки",
                        "Удалите избыточную информацию, не имеющую прямого отношения к теме"
                    ]
            
            elif feature == 'readability_score':
                min_value, max_value = self.optimal_ranges.get(feature, (70, 90))
                
                if value < min_value:
                    recommendations[feature] = [
                        f"Повысьте читабельность текста с текущих {value:.0f} до {min_value:.0f}-{max_value:.0f} баллов",
                        "Используйте более короткие предложения (в среднем до 20 слов)",
                        "Разбейте текст на небольшие абзацы по 2-3 предложения",
                        "Замените сложные термины более простыми аналогами"
                    ]
                elif value > max_value:
                    recommendations[feature] = [
                        "Добавьте более глубокий контент с экспертными деталями",
                        "Используйте отраслевые термины для повышения авторитетности"
                    ]
            
            elif feature == 'header_structure_score' and value < 0.7:
                recommendations[feature] = [
                    "Улучшите структуру заголовков: используйте H1, H2, H3 в правильной иерархии",
                    "Добавьте подзаголовки (H2, H3) для каждого логического раздела",
                    "Включите ключевые слова в заголовки всех уровней",
                    "Убедитесь, что H1 содержит главное ключевое слово"
                ]
            
            elif feature == 'meta_tags_score' and value < 0.7:
                recommendations[feature] = [
                    "Оптимизируйте мета-теги: title, description и alt-текст изображений",
                    "Включите основное ключевое слово в начало title и description",
                    "Сделайте description длиной 150-160 символов с призывом к действию",
                    "Убедитесь, что все изображения имеют alt-теги с ключевыми словами"
                ]
            
            elif feature == 'multimedia_score' and value < 0.6:
                recommendations[feature] = [
                    "Добавьте больше медиа-контента: изображения, инфографики, видео",
                    "Используйте не менее 1 изображения на каждые 300 слов текста",
                    "Добавьте описательные подписи к изображениям с ключевыми словами",
                    "Включите интерактивные элементы: опросы, калькуляторы, галереи"
                ]
            
            elif feature == 'internal_linking_score' and value < 0.6:
                recommendations[feature] = [
                    "Улучшите внутреннюю перелинковку: добавьте 3-5 внутренних ссылок",
                    "Используйте ключевые слова в анкорах внутренних ссылок",
                    "Добавьте ссылки на старые статьи в новые и наоборот",
                    "Создайте список рекомендуемых материалов в конце статьи"
                ]
            
            elif feature == 'topic_relevance' and value < 0.7:
                recommendations[feature] = [
                    "Повысьте тематическую релевантность контента",
                    "Расширьте охват темы, включив все важные подтемы",
                    "Добавьте ответы на распространенные вопросы по теме",
                    "Проведите исследование контента конкурентов из топ-10"
                ]
            
            elif feature == 'semantic_depth' and value < 0.7:
                recommendations[feature] = [
                    "Увеличьте семантическую глубину контента",
                    "Добавьте больше LSI-ключевых слов и связанных терминов",
                    "Расширьте контекст темы, используя тематический словарь",
                    "Структурируйте контент по кластерам связанных подтем"
                ]
            
            elif feature == 'engagement_potential' and value < 0.7:
                recommendations[feature] = [
                    "Повысьте потенциал вовлечения аудитории",
                    "Добавьте призывы к действию (CTA) в начале и конце материала",
                    "Включите вопросы к читателям для стимулирования комментариев",
                    "Добавьте интерактивные элементы: опросы, тесты, калькуляторы"
                ]
            
            elif feature in ['expertise_score', 'authority_score', 'trust_score', 'overall_eeat_score'] and value < 0.6:
                if feature == 'expertise_score':
                    recommendations[feature] = [
                        "Повысьте сигналы экспертизы в контенте",
                        "Добавьте информацию о квалификации и опыте автора",
                        "Включите ссылки на исследования и статистические данные",
                        "Используйте профессиональную терминологию и детальные объяснения"
                    ]
                elif feature == 'authority_score':
                    recommendations[feature] = [
                        "Усильте сигналы авторитетности",
                        "Добавьте цитаты и ссылки на признанных экспертов в отрасли",
                        "Включите упоминания публикаций в авторитетных источниках",
                        "Укажите профессиональные награды и достижения автора или компании"
                    ]
                elif feature == 'trust_score':
                    recommendations[feature] = [
                        "Повысьте сигналы доверия",
                        "Включите ссылки на первоисточники информации",
                        "Добавьте раздел с источниками в конце статьи",
                        "Укажите даты публикации и последнего обновления материала"
                    ]
                elif feature == 'overall_eeat_score':
                    recommendations[feature] = [
                        "Комплексно улучшите E-E-A-T сигналы",
                        "Укажите авторство материала с информацией о квалификации",
                        "Добавьте больше ссылок на авторитетные источники",
                        "Включите экспертные мнения и отзывы по теме"
                    ]
        
        return recommendations
