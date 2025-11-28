#!/usr/bin/env python3
"""
Генерация SEO-рекомендаций на основе результатов анализа.
"""
import argparse
import json
import sys
from pathlib import Path
from seo_ai_models.models.seo_advisor.suggester.suggester import Suggester


def generate_recommendations(input_file: str, output_file: str) -> bool:
    """
    Генерирует SEO-рекомендации на основе результатов анализа.
    
    Args:
        input_file: Путь к JSON файлу с результатами анализа
        output_file: Путь к выходному Markdown файлу
    
    Returns:
        bool: True если успешно, False в противном случае
    """
    try:
        # Читаем результаты анализа
        print(f"📂 Чтение результатов анализа из {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        # Инициализируем генератор рекомендаций
        print("⚙️ Инициализация Suggester...")
        suggester = Suggester()
        
        # Генерируем рекомендации
        print("▶ Генерация SEO-рекомендаций...")
        
                # Извлекаем необходимые данные из analysis_data
        # Создаем basic_recommendations на основе результатов анализа
        basic_recommendations = {}
        
        # Генерируем базовые рекомендации на основе метрик
        
        # Анализ content_length
        if 'technical_seo' in analysis_data:
            tech = analysis_data['technical_seo']
            content_len = tech.get('content_length', 0)
            
            if content_len < 300:
                basic_recommendations.setdefault('content_length', []).append(
                    "Критически малый объем контента ({} слов). Рекомендуется минимум 500-1000 слов.".format(content_len)
                )
            elif content_len < 1000:
                basic_recommendations.setdefault('content_length', []).append(
                    "Объем контента ({} слов) ниже рекомендуемого. Добавьте детальное описание темы.".format(content_len)
                )
                
        # Анализ meta_tags
        if 'meta_tags' in analysis_data:
            meta = analysis_data['meta_tags']
            
            if not meta.get('title'):
                basic_recommendations.setdefault('meta_tags', []).append(
                    "Отсутствует meta title. Добавьте уникальный заголовок с ключевыми словами."
                )
            elif len(meta.get('title', '')) < 30:
                basic_recommendations.setdefault('meta_tags', []).append(
                    "Meta title слишком короткий. Рекомендуемая длина: 50-60 символов."
                )
            elif len(meta.get('title', '')) > 70:
                basic_recommendations.setdefault('meta_tags', []).append(
                    "Meta title слишком длинный. Сократите до 60 символов."
                )
                
            if not meta.get('description'):
                basic_recommendations.setdefault('meta_tags', []).append(
                    "Отсутствует meta description. Добавьте описание до 160 символов."
                )
                
        # Анализ readability
        if 'content_analysis' in analysis_data:
            content = analysis_data['content_analysis']
            if 'readability' in content:
                read_score = content['readability'].get('score', 0)
                if read_score < 0.4:
                    basic_recommendations.setdefault('readability', []).append(
                        "Низкая читабельность текста. Упростите предложения и добавьте подзаголовки."
                    )
                    
        # Если нет ни одной рекомендации, добавим общую
        if not basic_recommendations:
            basic_recommendations['general'] = ['Основные SEO-метрики в норме. Продолжайте работу над контентом.']
