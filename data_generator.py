"""
Генератор реалистичных данных для обучения модели E-E-A-T
"""

import random
import numpy as np
import pandas as pd
from typing import List, Dict, Union

def generate_realistic_dataset(count: int = 1000) -> List[Dict[str, Union[float, bool, str]]]:
    """
    Генерация реалистичного набора данных для обучения модели E-E-A-T
    
    Args:
        count: Количество примеров для генерации
        
    Returns:
        Список словарей с данными
    """
    data = []
    
    # Определяем отрасли и их особенности
    industries = {
        'finance': {
            'is_ymyl': True,
            'base_expertise': 0.7,      # Высокие требования к экспертизе
            'base_authority': 0.8,      # Очень высокие требования к авторитетности
            'base_trust': 0.8,          # Очень высокие требования к доверию
            'quality_levels': {
                'low': 0.2,             # % низкокачественного контента
                'medium': 0.3,          # % среднего контента 
                'high': 0.5             # % высококачественного контента
            }
        },
        'health': {
            'is_ymyl': True,
            'base_expertise': 0.75,     # Самые высокие требования к экспертизе
            'base_authority': 0.75,     # Высокие требования к авторитетности
            'base_trust': 0.85,         # Очень высокие требования к доверию
            'quality_levels': {
                'low': 0.25,
                'medium': 0.35,
                'high': 0.4
            }
        },
        'legal': {
            'is_ymyl': True,
            'base_expertise': 0.7,
            'base_authority': 0.75,
            'base_trust': 0.75,
            'quality_levels': {
                'low': 0.2,
                'medium': 0.35, 
                'high': 0.45
            }
        },
        'tech': {
            'is_ymyl': False,
            'base_expertise': 0.6,
            'base_authority': 0.65,
            'base_trust': 0.6,
            'quality_levels': {
                'low': 0.25,
                'medium': 0.45,
                'high': 0.3
            }
        },
        'ecommerce': {
            'is_ymyl': False,
            'base_expertise': 0.5,
            'base_authority': 0.6,
            'base_trust': 0.65,
            'quality_levels': {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.2
            }
        },
        'travel': {
            'is_ymyl': False,
            'base_expertise': 0.55,
            'base_authority': 0.5,
            'base_trust': 0.6,
            'quality_levels': {
                'low': 0.2,
                'medium': 0.6,
                'high': 0.2
            }
        },
        'blog': {
            'is_ymyl': False,
            'base_expertise': 0.4,
            'base_authority': 0.4,
            'base_trust': 0.5,
            'quality_levels': {
                'low': 0.4,
                'medium': 0.5,
                'high': 0.1
            }
        }
    }
    
    # Определяем профили качества контента (коэффициенты для каждого уровня)
    quality_profiles = {
        'low': {
            'expertise_factor': (0.2, 0.4),       # Диапазон (min, max)
            'authority_factor': (0.2, 0.4),
            'trust_factor': (0.2, 0.5),
            'structure_factor': (0.3, 0.7),
            'semantics_factor': (0.2, 0.5),
            'citations_factor': (0.1, 0.3),
            'external_links_factor': (0.1, 0.4)
        },
        'medium': {
            'expertise_factor': (0.4, 0.7),
            'authority_factor': (0.4, 0.7),
            'trust_factor': (0.5, 0.7),
            'structure_factor': (0.5, 0.8),
            'semantics_factor': (0.4, 0.7),
            'citations_factor': (0.3, 0.6),
            'external_links_factor': (0.3, 0.6)
        },
        'high': {
            'expertise_factor': (0.7, 0.95),
            'authority_factor': (0.7, 0.95),
            'trust_factor': (0.7, 0.95),
            'structure_factor': (0.8, 0.95),
            'semantics_factor': (0.7, 0.95),
            'citations_factor': (0.6, 0.95),
            'external_links_factor': (0.5, 0.95)
        }
    }
    
    # Генерация данных с учетом распределения
    for _ in range(count):
        # Выбираем случайную отрасль с учетом распределения
        industry_weights = {
            'finance': 0.15,
            'health': 0.15,
            'legal': 0.1,
            'tech': 0.2,
            'ecommerce': 0.15,
            'travel': 0.1,
            'blog': 0.15
        }
        industry = np.random.choice(
            list(industry_weights.keys()),
            p=list(industry_weights.values())
        )
        
        industry_profile = industries[industry]
        is_ymyl = industry_profile['is_ymyl']
        
        # Выбираем уровень качества контента с учетом распределения для отрасли
        quality_weights = list(industry_profile['quality_levels'].values())
        quality_levels = list(industry_profile['quality_levels'].keys())
        quality_level = np.random.choice(quality_levels, p=quality_weights)
        
        # Получаем профиль качества
        profile = quality_profiles[quality_level]
        
        # Генерируем значения метрик с учетом базовых значений для отрасли
        # и факторов качества, с добавлением некоторого шума
        
        # Основные E-E-A-T метрики с корреляцией
        # Генерируем базовое значение для корреляции
        base_quality = random.uniform(
            min(profile['expertise_factor'][0], profile['authority_factor'][0], profile['trust_factor'][0]),
            max(profile['expertise_factor'][1], profile['authority_factor'][1], profile['trust_factor'][1])
        )
        
        # Корреляция между E-E-A-T компонентами с некоторой вариацией
        expertise_score = min(1.0, max(0.0, 
            industry_profile['base_expertise'] * 
            random.uniform(profile['expertise_factor'][0], profile['expertise_factor'][1]) *
            (0.7 + 0.3 * base_quality) + 
            random.uniform(-0.07, 0.07)  # Добавляем шум
        ))
        
        authority_score = min(1.0, max(0.0, 
            industry_profile['base_authority'] * 
            random.uniform(profile['authority_factor'][0], profile['authority_factor'][1]) *
            (0.7 + 0.3 * base_quality) + 
            random.uniform(-0.07, 0.07)
        ))
        
        trust_score = min(1.0, max(0.0, 
            industry_profile['base_trust'] * 
            random.uniform(profile['trust_factor'][0], profile['trust_factor'][1]) *
            (0.7 + 0.3 * base_quality) + 
            random.uniform(-0.07, 0.07)
        ))
        
        # Вспомогательные метрики
        structural_score = min(1.0, max(0.0,
            random.uniform(profile['structure_factor'][0], profile['structure_factor'][1]) +
            random.uniform(-0.1, 0.1)
        ))
        
        semantic_score = min(1.0, max(0.0,
            random.uniform(profile['semantics_factor'][0], profile['semantics_factor'][1]) +
            random.uniform(-0.1, 0.1)
        ))
        
        citation_score = min(1.0, max(0.0,
            random.uniform(profile['citations_factor'][0], profile['citations_factor'][1]) +
            random.uniform(-0.1, 0.1)
        ))
        
        external_links_score = min(1.0, max(0.0,
            random.uniform(profile['external_links_factor'][0], profile['external_links_factor'][1]) +
            random.uniform(-0.1, 0.1)
        ))
        
        # Расчет общей оценки E-E-A-T с учетом YMYL-статуса
        if is_ymyl:
            weights = {
                'expertise': 0.20,
                'authority': 0.25,
                'trust': 0.30,
                'structure': 0.05,
                'semantics': 0.10,
                'citations': 0.05,
                'external_links': 0.05
            }
        else:
            weights = {
                'expertise': 0.25,
                'authority': 0.20,
                'trust': 0.20,
                'structure': 0.10,
                'semantics': 0.15,
                'citations': 0.05,
                'external_links': 0.05
            }
        
        overall_score = (
            expertise_score * weights['expertise'] +
            authority_score * weights['authority'] +
            trust_score * weights['trust'] +
            structural_score * weights['structure'] +
            semantic_score * weights['semantics'] +
            citation_score * weights['citations'] +
            external_links_score * weights['external_links']
        )
        
        # Добавляем небольшой случайный шум для реалистичности
        overall_score = min(1.0, max(0.0, overall_score + random.uniform(-0.03, 0.03)))
        
        # Создаем запись данных
        data_entry = {
            'expertise_score': expertise_score,
            'authority_score': authority_score,
            'trust_score': trust_score,
            'structural_score': structural_score,
            'semantic_coherence_score': semantic_score,
            'citation_score': citation_score,
            'external_links_score': external_links_score,
            'overall_eeat_score': overall_score,
            'ymyl_status': is_ymyl,
            'industry': industry,
            'quality_level': quality_level  # Для анализа
        }
        
        data.append(data_entry)
    
    return data

def analyze_dataset(data: List[Dict[str, Union[float, bool, str]]]) -> Dict:
    """
    Анализ распределения данных в наборе
    
    Args:
        data: Список словарей с данными
        
    Returns:
        Словарь с результатами анализа
    """
    # Конвертируем в DataFrame для удобства анализа
    df = pd.DataFrame(data)
    
    # Базовый анализ распределения
    analysis = {
        'count': len(df),
        'ymyl_distribution': df['ymyl_status'].value_counts().to_dict(),
        'industry_distribution': df['industry'].value_counts().to_dict(),
        'quality_distribution': df['quality_level'].value_counts().to_dict(),
        'metrics_stats': {
            column: {
                'min': df[column].min(),
                'max': df[column].max(),
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std()
            }
            for column in df.columns if column not in ['ymyl_status', 'industry', 'quality_level']
        },
        'ymyl_vs_non_ymyl': {
            'ymyl_mean_score': df[df['ymyl_status'] == True]['overall_eeat_score'].mean(),
            'non_ymyl_mean_score': df[df['ymyl_status'] == False]['overall_eeat_score'].mean()
        },
        'correlations': df[[
            'expertise_score', 'authority_score', 'trust_score', 
            'structural_score', 'semantic_coherence_score', 
            'citation_score', 'external_links_score', 'overall_eeat_score'
        ]].corr().to_dict()
    }
    
    return analysis

if __name__ == "__main__":
    # Тестирование генератора
    data = generate_realistic_dataset(100)
    analysis = analyze_dataset(data)
    print(f"Сгенерировано {analysis['count']} записей")
    print("Распределение по отраслям:", analysis['industry_distribution'])
    print("Распределение по уровням качества:", analysis['quality_distribution'])
    print("Средняя общая оценка:", analysis['metrics_stats']['overall_eeat_score']['mean'])
